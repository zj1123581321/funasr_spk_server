"""
Phase 3 Path B: PyTorch -> coremltools -> .mlpackage 转换 Qwen3-ASR backend.

input:   HF Qwen/Qwen3-ASR-1.7B (snapshot cache)
output:  models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage

设计要点 (依据 Phase 3 prompt 第 1.6 节硬警告):
- static shape (1, 390, 1024)   : Phase 2 frontend 13 fps × 30s = 390 帧
- 2D key_padding_mask 输入       : 内部转 [B,1,1,T] additive mask (issue #19887 ANE 友好)
- compute_units=CPU_AND_NE      : 不要 ALL (抢 llama.cpp Metal)
- compute_precision=FLOAT16     : INT4 palettization ANE 慢 3.3x (Ivan), 留 FP16 baseline
- convert_to=mlprogram          : 不要 NeuralNetwork
- minimum_deployment_target=mac15

使用:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/export_backend_coreml.py \
        --time 390 --precision fp16 --output models/.../qwen3_asr_encoder_backend.mlpackage
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 让 vendor modeling 可 import
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-ASR-1.7B", help="HF repo id")
    ap.add_argument("--time", type=int, default=390, help="static time dim (13 fps × 30s = 390)")
    ap.add_argument("--dim", type=int, default=1024, help="audio d_model")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--units", choices=["CPU_AND_NE", "CPU_AND_GPU", "ALL", "CPU_ONLY"], default="CPU_AND_NE")
    ap.add_argument(
        "--output",
        default=str(ROOT / "models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage"),
    )
    ap.add_argument("--int8", action="store_true", help="加 INT8 linear_symmetric 量化")
    ap.add_argument("--check-only", action="store_true", help="只 trace + 数值 sanity, 不 convert")
    return ap.parse_args()


def load_audio_tower(repo: str):
    """加载 audio_tower (来自 HF Qwen3-ASR-1.7B snapshot)。

    HF repo 没 auto_map, transformers 不识别 model_type=qwen3_asr。
    所以手动: vendor `Qwen3ASRAudioEncoder._from_config()` + 从 safetensors 抽 `thinker.audio_tower.*` keys 加载。
    """
    import glob
    import json

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
    )
    from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
    )

    print(f"[load] snapshot_download({repo!r}) ...")
    snap_dir = snapshot_download(
        repo,
        allow_patterns=["config.json", "model-*.safetensors", "model.safetensors.index.json"],
    )
    print(f"[load] snapshot: {snap_dir}")

    # 1. audio_config
    cfg_path = Path(snap_dir) / "config.json"
    full_cfg = json.loads(cfg_path.read_text())
    audio_cfg_dict = full_cfg["thinker_config"]["audio_config"]
    audio_cfg = Qwen3ASRAudioEncoderConfig(**audio_cfg_dict)
    print(
        f"[load] audio_config: d_model={audio_cfg.d_model}, "
        f"layers={audio_cfg.encoder_layers}, "
        f"heads={audio_cfg.encoder_attention_heads}, "
        f"ffn={audio_cfg.encoder_ffn_dim}, "
        f"output_dim={audio_cfg.output_dim}"
    )

    # 2. 实例化空 audio_tower
    audio_tower = Qwen3ASRAudioEncoder._from_config(audio_cfg).to(torch.float32).eval()
    print(f"[load] empty audio_tower instantiated, params = {sum(p.numel() for p in audio_tower.parameters()) / 1e6:.1f}M")

    # 3. 从 safetensors 抽 thinker.audio_tower.* 加载
    idx = json.loads((Path(snap_dir) / "model.safetensors.index.json").read_text())
    shard_to_keys: dict[str, list[str]] = {}
    for k, shard in idx["weight_map"].items():
        if k.startswith("thinker.audio_tower."):
            shard_to_keys.setdefault(shard, []).append(k)
    print(f"[load] 抽 {sum(len(v) for v in shard_to_keys.values())} audio_tower keys from {len(shard_to_keys)} shards")

    new_state: dict[str, torch.Tensor] = {}
    PREFIX = "thinker.audio_tower."
    for shard_fn in sorted(shard_to_keys.keys()):
        shard_path = Path(snap_dir) / shard_fn
        print(f"[load]   load {shard_fn} ...")
        t0 = time.time()
        # 只加载这个 shard 内 audio_tower keys (省内存)
        all_in_shard = load_file(str(shard_path))
        for k in shard_to_keys[shard_fn]:
            short = k[len(PREFIX):]
            new_state[short] = all_in_shard[k].to(torch.float32)
        # 释放非 audio_tower 部分 ASAP
        del all_in_shard
        print(f"[load]   shard loaded in {time.time()-t0:.1f}s")

    missing, unexpected = audio_tower.load_state_dict(new_state, strict=False)
    # SinusoidsPositionEmbedding.positional_embedding 是 persistent=False 的 buffer, 不在 state_dict
    expected_missing = {"positional_embedding.positional_embedding"}
    real_missing = [k for k in missing if k not in expected_missing]
    if real_missing:
        print(f"[load] WARN missing keys ({len(real_missing)}):", real_missing[:10])
    if unexpected:
        print(f"[load] WARN unexpected keys ({len(unexpected)}):", unexpected[:10])

    print(
        f"[load] audio_tower loaded ✅: "
        f"layers={len(audio_tower.layers)}, embed_dim={audio_tower.layers[0].embed_dim}, "
        f"d_model={audio_cfg.d_model}"
    )
    return audio_tower


class BackendCoreMLWrapper(nn.Module):
    """
    包 Qwen3ASRBackendOnnx, forward 接 2D key_padding_mask, 内部转 [B,1,1,T] additive。

    issue #19887 (CoreML EP 4D mask fallback CPU) 警告 → 2D mask 是 ANE 推荐做法。
    实际 attention 内部 attn_weights [B,H,T_q,T_k] + additive [B,1,1,T_k] broadcast 等价。
    """

    def __init__(self, audio_tower):
        super().__init__()
        from src.core.vendor.qwen_asr_gguf.export.qwen3_asr_custom.modeling_qwen3_asr_onnx import (
            Qwen3ASRBackendOnnx,
        )
        # Qwen3ASRBackendOnnx 会就地修改 audio_tower.layers 的 self_attn (替成 Qwen3ASRAudioAttentionOnnx)
        self.backend = Qwen3ASRBackendOnnx(audio_tower)

    def forward(self, hidden_states: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states:     [B, T, D]   fp32 (trace) / fp16 (CoreML 内部)
        key_padding_mask:  [B, T]      int32, 1 = 有效 token, 0 = pad
        return:            [B, T, output_dim]
        """
        b, t = key_padding_mask.shape
        # (1 - mask) * -10000 → real=0, pad=-10000.0
        additive = (1.0 - key_padding_mask.to(hidden_states.dtype)).view(b, 1, 1, t) * -10000.0
        return self.backend(hidden_states, attention_mask=additive)


def trace_wrapper(audio_tower, time_dim: int, dim: int) -> torch.jit.ScriptModule:
    """trace BackendCoreMLWrapper 到 TorchScript。"""
    print(f"[trace] 包 BackendCoreMLWrapper + trace, T={time_dim} D={dim}")
    wrapper = BackendCoreMLWrapper(audio_tower).eval()
    example_h = torch.randn(1, time_dim, dim, dtype=torch.float32)
    example_mask = torch.ones(1, time_dim, dtype=torch.int32)

    with torch.no_grad():
        # 先跑一次 sanity check
        out = wrapper(example_h, example_mask)
        print(f"[trace] sanity out shape = {tuple(out.shape)}, dtype = {out.dtype}")
        traced = torch.jit.trace(wrapper, (example_h, example_mask), strict=False)
    return traced


def convert_to_mlpackage(traced, time_dim: int, dim: int, precision: str, units: str, output: str):
    """coremltools.convert 到 mlpackage。"""
    import coremltools as ct

    units_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    precision_map = {
        "fp16": ct.precision.FLOAT16,
        "fp32": ct.precision.FLOAT32,
    }

    print(f"[convert] ct.convert: units={units}, precision={precision}, T={time_dim}, D={dim}")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, time_dim, dim), dtype=np.float32),
            ct.TensorType(name="key_padding_mask", shape=(1, time_dim), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=units_map[units],
        compute_precision=precision_map[precision],
        convert_to="mlprogram",
    )
    print(f"[convert] done in {time.time()-t0:.1f}s")
    return mlmodel


def main():
    args = parse_args()

    audio_tower = load_audio_tower(args.repo)
    traced = trace_wrapper(audio_tower, args.time, args.dim)

    if args.check_only:
        print("[done] --check-only: trace OK, 不 convert")
        return

    mlmodel = convert_to_mlpackage(traced, args.time, args.dim, args.precision, args.units, args.output)

    if args.int8:
        print("[int8] 跑 linear_quantize_weights ...")
        from coremltools.optimize.coreml import (
            linear_quantize_weights,
            OpLinearQuantizerConfig,
            OptimizationConfig,
        )
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        mlmodel = linear_quantize_weights(mlmodel, config=cfg)
        # 改输出文件名: ...backend_int8.mlpackage
        out_path = Path(args.output)
        out_int8 = out_path.with_name(out_path.stem + "_int8" + out_path.suffix)
        mlmodel.save(str(out_int8))
        print(f"[int8] saved: {out_int8}")
    else:
        mlmodel.save(args.output)
        print(f"[save] saved: {args.output}")


if __name__ == "__main__":
    main()
