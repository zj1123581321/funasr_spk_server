"""Spike: 单 worker, 跑一个长音频, 打 per-stage timing 报告.

用法:
    venv/bin/python spikes/qwen3_mac_hw_accel/profile_worker.py \
        <audio_path> <label> [--out-json <path>] [--num-threads N] [--provider cpu|coreml]

设计:
- 不走 server, 不走 file_based_process_pool. 直接 import Qwen3DiarizeTranscriber.
- 装 timing_hooks monkey-patch 测 per-stage 时间分布.
- N=2 并发由外层 spawn 两个独立 process 自然模拟生产 worker pool.
- 各 task 在自己 process 内独立 ASR engine + sherpa pipeline, 不共享状态.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio_path", type=str)
    p.add_argument("label", type=str)
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--provider", type=str, default="cpu", choices=["cpu", "coreml"])
    p.add_argument("--onnx-provider", type=str, default="CPU",
                   choices=["CPU", "COREML", "COREML_ANE_FE", "COREML_ANE_FULL"],
                   help="ASR encoder ONNX EP. COREML_ANE_FE=Phase 2 (fe ANE), "
                        "COREML_ANE_FULL=Phase 3 (fe ANE + be mlpackage ANE)")
    p.add_argument("--enable-coreml-asr-patch", action="store_true",
                   help="开启后, monkey-patch vendor encoder 强制走 CoreMLExecutionProvider")
    p.add_argument("--coreml-units", type=str, default="ALL",
                   choices=["ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"],
                   help="MLComputeUnits 路由")
    p.add_argument("--coreml-format", type=str, default="MLProgram",
                   choices=["MLProgram", "NeuralNetwork"],
                   help="CoreML 模型格式")
    p.add_argument("--coreml-static-shapes", action="store_true",
                   help="RequireStaticInputShapes=1")
    p.add_argument("--coreml-only-frontend", action="store_true",
                   help="仅 frontend 走 CoreML, backend 保持 CPU (绕开 backend op 兼容问题)")
    args = p.parse_args()

    # 装 timing patches (必须在 transcriber import 之前)
    from spikes.qwen3_mac_hw_accel import timing_hooks
    timing_hooks.install_patches()

    # 可选: 方向 1 — 替换 vendor encoder 的 provider 选择逻辑
    if args.enable_coreml_asr_patch:
        _install_coreml_asr_patch(
            units=args.coreml_units,
            fmt=args.coreml_format,
            static_shapes=args.coreml_static_shapes,
            only_frontend=args.coreml_only_frontend,
        )

    from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber
    from loguru import logger

    # 配置
    MODEL_ROOT = PROJECT_ROOT / "models" / "qwen3_diarize"
    asr_model_dir = str(MODEL_ROOT / "Qwen3-ASR-1.7B")
    segmentation_model = str(MODEL_ROOT / "sherpa" / "pyannote-segmentation-3.0" / "model.onnx")
    embedding_model = str(MODEL_ROOT / "sherpa" / "nemo-titanet-small" / "embedding.onnx")

    print(f"[{args.label}] PID {os.getpid()} start audio={args.audio_path}", flush=True)
    print(f"[{args.label}] num_threads={args.num_threads} sherpa_provider={args.provider} "
          f"asr_onnx_provider={args.onnx_provider} coreml_asr_patch={args.enable_coreml_asr_patch}",
          flush=True)

    transcriber = Qwen3DiarizeTranscriber(
        asr_model_dir=asr_model_dir,
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        num_speakers=None,
        cluster_threshold=0.9,
        num_threads=args.num_threads,
        provider=args.provider,
    )

    # 若指定 ASR ONNX provider != CPU 默认, 透到 engine config
    if args.onnx_provider != "CPU":
        # build_engine_config 默认 onnx_provider="CPU"; 这里通过 monkey-patch
        from src.core.qwen3 import asr as _qwen3_asr_mod
        _orig_build_cfg = _qwen3_asr_mod.build_engine_config

        def _patched_build_cfg(model_dir, **kw):
            kw["onnx_provider"] = args.onnx_provider
            return _orig_build_cfg(model_dir, **kw)

        _qwen3_asr_mod.build_engine_config = _patched_build_cfg

    async def _run():
        await transcriber.initialize()
        t0 = time.monotonic()
        result = await transcriber.transcribe(
            audio_path=args.audio_path,
            task_id=args.label,
            output_format="json",
        )
        wall = time.monotonic() - t0
        return result, wall

    result, wall = asyncio.run(_run())

    # 解 result (Tuple[TranscriptionResult, dict])
    if isinstance(result, tuple):
        tr, raw = result
        duration = getattr(tr, "duration", raw.get("asr_duration", 0))
        n_segments = len(getattr(tr, "segments", []))
        speakers = sorted({s.speaker for s in getattr(tr, "segments", [])})
    else:
        duration = result.get("duration", 0)
        n_segments = 0
        speakers = []
        raw = result.get("raw_result", {})

    print(f"\n[{args.label}] DONE wall={wall:.2f}s duration={duration:.2f}s "
          f"RTF={wall/max(duration, 0.001):.3f} segments={n_segments} speakers={speakers}",
          flush=True)
    print(timing_hooks.fmt_report(), flush=True)

    if args.out_json:
        rep = timing_hooks.get_timer().report()
        rep["meta"] = {
            "label": args.label,
            "audio_path": args.audio_path,
            "wall": wall,
            "duration": duration,
            "rtf": wall / max(duration, 0.001),
            "n_segments": n_segments,
            "speakers": speakers,
            "num_threads": args.num_threads,
            "provider": args.provider,
            "onnx_provider": args.onnx_provider,
            "coreml_asr_patch": args.enable_coreml_asr_patch,
            "asr_text_len": len(raw.get("asr_text", "")) if isinstance(raw, dict) else 0,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f"[{args.label}] timing saved to {args.out_json}", flush=True)


def _install_coreml_asr_patch(units="ALL", fmt="MLProgram", static_shapes=False,
                              only_frontend=False):
    """方向 1: 让 vendor QwenAudioEncoder 走 CoreML EP, 支持多种 config."""
    import onnxruntime as ort
    from src.core.vendor.qwen_asr_gguf.inference import encoder as _enc

    _OrigCls = _enc.QwenAudioEncoder
    cpu_provider = 'CPUExecutionProvider'

    def _build_providers():
        available = ort.get_available_providers()
        if 'CoreMLExecutionProvider' not in available:
            return [cpu_provider]
        return [
            ('CoreMLExecutionProvider', {
                'ModelFormat': fmt,
                'MLComputeUnits': units,
                'RequireStaticInputShapes': '1' if static_shapes else '0',
                'EnableOnSubgraphs': '0',
            }),
            cpu_provider,
        ]

    class _PatchedEncoder(_OrigCls):
        def __init__(self, frontend_path, backend_path, onnx_provider='CPU',
                     dml_pad_to=30, verbose=True):
            self.verbose = verbose
            self.onnx_provider = onnx_provider.upper()
            self.active_dml = False
            self.dml_pad_to = dml_pad_to
            self.h_target_len = self.dml_pad_to * 13

            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers_fe = _build_providers()
            providers_be = [cpu_provider] if only_frontend else _build_providers()
            if self.verbose:
                print(f"--- [PATCH-Encoder] CoreML units={units} fmt={fmt} "
                      f"static={static_shapes} only_frontend={only_frontend} ---", flush=True)
                print(f"    Frontend: {os.path.basename(frontend_path)}")
                print(f"    Backend:  {os.path.basename(backend_path)}")

            self.sess_fe = ort.InferenceSession(frontend_path, sess_options=sess_opts,
                                                providers=providers_fe)
            self.sess_be = ort.InferenceSession(backend_path, sess_options=sess_opts,
                                                providers=providers_be)
            print(f"    sess_fe providers: {self.sess_fe.get_providers()}", flush=True)
            print(f"    sess_be providers: {self.sess_be.get_providers()}", flush=True)

            from src.core.vendor.qwen_asr_gguf.inference.encoder import FastWhisperMel
            import numpy as np
            self.mel_extractor = FastWhisperMel()
            try:
                fe_input_type = self.sess_fe.get_inputs()[0].type
                self.input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
            except Exception:
                self.input_dtype = np.float32

            if self.verbose:
                print("--- [PATCH-Encoder] 预热 (2s 静音)... ---", flush=True)
            dummy_wav = np.zeros(int(16000 * 2.0)).astype(np.float32)
            _ = self.encode(dummy_wav)
            if self.verbose:
                print("--- [PATCH-Encoder] 预热完成 ---", flush=True)

    _enc.QwenAudioEncoder = _PatchedEncoder
    from src.core.vendor.qwen_asr_gguf.inference import asr as _asr_mod
    _asr_mod.QwenAudioEncoder = _PatchedEncoder


if __name__ == "__main__":
    main()
