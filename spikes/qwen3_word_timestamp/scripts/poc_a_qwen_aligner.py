#!/usr/bin/env python3
# coding=utf-8
"""PoC-A: vendor 自带 Qwen3-ForcedAligner (GGUF+ONNX split) 验证

目标:验证重路线 feasibility + 量 RTF + 出词级时间戳, 跟 fa-zh (PoC-B) 头对头比中文精度。
在远端 3060 box 上跑 (权重 + libllama-cuda.so 都在那)。

接入坑 (见记忆 reference_qwen3_aligner_weights):
- llm_fn 必须 override 为 q5_k (AlignerConfig 默认 q4_k, 实际权重是 q5_k)
- n_ctx=2048 → 单次 align ≈68s 上限; 本 PoC 用 60s clip, 单次喂得下
- encoder 走 CPU, GGUF 走 CUDA (项目文档的工作组合)

用法 (在 box 上):
    venv/bin/python spikes/qwen3_word_timestamp/scripts/poc_a_qwen_aligner.py \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav \
        --golden tests/fixtures/golden/podcast_2speakers_60s.golden.json \
        --model-dir /data/projects/CapsWriter-Offline-with-AI/models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B \
        --out spikes/qwen3_word_timestamp/outputs/poc_a_60s.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_reference_text(golden_path: Path) -> str:
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    return "".join(seg["text"] for seg in data["segments"])


def load_audio_16k_mono(audio_path: str) -> tuple[np.ndarray, float]:
    """加载音频为 16k mono float32, 返回 (samples, duration_sec)"""
    import librosa

    samples, sr = librosa.load(audio_path, sr=16000, mono=True)
    samples = samples.astype(np.float32)
    return samples, len(samples) / 16000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--llm-fn", default="qwen3_aligner_llm.q5_k.gguf")
    ap.add_argument("--onnx-provider", default="CPU", help="encoder EP: CPU / CUDA")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_text = args.text or load_reference_text(Path(args.golden))
    samples, dur = load_audio_16k_mono(args.audio)
    print(f"[PoC-A] audio dur={dur:.2f}s text_len={len(ref_text)}字 provider={args.onnx_provider}")

    from src.core.vendor.qwen_asr_gguf.inference.schema import AlignerConfig
    from src.core.vendor.qwen_asr_gguf.inference.aligner import QwenForcedAligner

    cfg = AlignerConfig(
        model_dir=args.model_dir,
        llm_fn=args.llm_fn,
        onnx_provider=args.onnx_provider,
        llm_use_gpu=True,
    )

    t_load0 = time.time()
    aligner = QwenForcedAligner(cfg)
    t_load = time.time() - t_load0
    print(f"[PoC-A] aligner 加载耗时 {t_load:.2f}s")

    t0 = time.time()
    res = aligner.align(samples, ref_text, language=args.language)
    elapsed = time.time() - t0
    rtf = elapsed / dur if dur > 0 else None

    items = [{"text": it.text, "start": it.start_time, "end": it.end_time} for it in res.items]
    n = len(items)
    print(f"[PoC-A] 对齐耗时 {elapsed:.2f}s  RTF={rtf}  items={n}  perf={res.performance}")
    print("[PoC-A] 前 12 项 (text: start-end s):")
    for it in items[:12]:
        print(f"    {it['text']!r}: {it['start']:.3f}-{it['end']:.3f}")

    Path(args.out).write_text(
        json.dumps(
            {
                "audio": args.audio,
                "duration_sec": dur,
                "text_len": len(ref_text),
                "provider": args.onnx_provider,
                "load_sec": t_load,
                "align_sec": elapsed,
                "rtf": rtf,
                "n_items": n,
                "performance": res.performance,
                "items": items,
                "reference_text": ref_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PoC-A] 结果落 {args.out}")


if __name__ == "__main__":
    main()
