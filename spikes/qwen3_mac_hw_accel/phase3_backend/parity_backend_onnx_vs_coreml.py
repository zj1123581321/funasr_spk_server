"""
Step 4: parity 测试 - 同一 hidden_states 输入下,
ONNX backend (CPU) vs mlpackage backend (ANE) 输出对比。

用法:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/parity_backend_onnx_vs_coreml.py \
        --onnx models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx \
        --mlpackage models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage \
        --frontend models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav

逻辑:
  1. 加载真实音频, 用 frontend ONNX (CPU 或 ANE) 跑出 hidden_states (变长 T)
  2. Padding 到 fixed time = 390 (符合 mlpackage static shape)
  3. ONNX backend (CPU) 用 [B,1,T,T] additive mask 跑
  4. mlpackage backend (ANE) 用 [B,T] key_padding_mask 跑
  5. 比较两者 last_hidden_state 的有效部分 (前 seq_len 帧)
  6. 阈值: cos_sim > 0.999, max_abs_diff < 1e-2 (FP16 数值噪声)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="ONNX backend 路径")
    ap.add_argument("--mlpackage", required=True, help="CoreML mlpackage 路径")
    ap.add_argument("--frontend", required=True, help="ONNX frontend 路径")
    ap.add_argument("--audio", required=True, help="测试音频 (wav)")
    ap.add_argument("--time", type=int, default=390)
    ap.add_argument("--dim", type=int, default=1024)
    return ap.parse_args()


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """加载音频, 重采样到 16k mono float32"""
    import soundfile as sf

    audio, file_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        # 简单线性插值重采样 (足够 parity)
        import math
        ratio = sr / file_sr
        new_len = int(math.floor(len(audio) * ratio))
        idx = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def main():
    args = parse_args()
    from src.core.vendor.qwen_asr_gguf.inference.encoder import (
        QwenAudioEncoder, FastWhisperMel, get_feat_extract_output_lengths,
    )
    import onnxruntime as ort

    # 1. 加载音频 + mel
    print(f"[audio] {args.audio}")
    audio = load_audio(args.audio)
    audio_30s = audio[: 16000 * 30]  # 截 30s 对齐 time=390
    print(f"[audio] shape={audio_30s.shape}, dur={len(audio_30s)/16000:.1f}s")

    # 2. 跑 frontend ONNX → hidden_states (CPU 确定数值)
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    fe_sess = ort.InferenceSession(args.frontend, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # 探测 frontend dtype (production ONNX 是 fp16)
    fe_dtype_str = fe_sess.get_inputs()[0].type
    fe_dtype = np.float16 if "float16" in fe_dtype_str else np.float32
    print(f"[fe] input dtype = {fe_dtype}")

    mel_extractor = FastWhisperMel()
    mel = mel_extractor(audio_30s, dtype=fe_dtype)
    print(f"[mel] shape={mel.shape}, dtype={mel.dtype}")

    T = mel.shape[1]
    pad_len = (100 - (T % 100)) % 100
    if pad_len > 0:
        mel = np.pad(mel, ((0, 0), (0, pad_len)), mode="constant")
    mel_input = mel[np.newaxis, ...]
    num_chunks = mel_input.shape[2] // 100
    fe_outputs = []
    for i in range(num_chunks):
        chunk = mel_input[:, :, i * 100 : (i + 1) * 100].astype(fe_dtype)
        out = fe_sess.run(None, {"chunk_mel": chunk})[0]
        fe_outputs.append(out)
    hidden_states = np.concatenate(fe_outputs, axis=1)
    seq_len = get_feat_extract_output_lengths(T)
    hidden_states = hidden_states[:, :seq_len, :]  # (1, seq_len, 1024)
    print(f"[fe] hidden_states shape={hidden_states.shape}, valid seq_len={seq_len}")

    # 3. Padding 到 time=390 (mlpackage static shape)
    target_t = args.time
    if seq_len > target_t:
        print(f"[warn] seq_len={seq_len} > target={target_t}, 截断")
        hidden_states = hidden_states[:, :target_t, :]
        seq_len = target_t
        pad_width = 0
    else:
        pad_width = target_t - seq_len
        hidden_states_padded = np.pad(hidden_states, ((0, 0), (0, pad_width), (0, 0)), mode="constant")

    # 4. ONNX backend (CPU) — 4D additive mask
    print(f"[onnx_be] 加载 {Path(args.onnx).name} ...")
    be_sess = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    be_dtype_str = be_sess.get_inputs()[0].type
    be_dtype = np.float16 if "float16" in be_dtype_str else np.float32
    print(f"[onnx_be] input dtype = {be_dtype}")
    additive_mask_4d = np.zeros((1, 1, target_t, target_t), dtype=be_dtype)
    additive_mask_4d[:, :, :, seq_len:] = -10000.0
    t0 = time.time()
    onnx_out = be_sess.run(None, {
        "hidden_states": hidden_states_padded.astype(be_dtype),
        "attention_mask": additive_mask_4d,
    })[0]
    t_onnx = time.time() - t0
    print(f"[onnx_be] out shape={onnx_out.shape}, time={t_onnx*1000:.1f}ms")

    # 5. CoreML mlpackage — 2D key_padding_mask
    import coremltools as ct
    print(f"[coreml_be] 加载 {Path(args.mlpackage).name} ...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)
    t_load = time.time() - t0
    print(f"[coreml_be] load {t_load:.1f}s")

    key_padding_mask = np.zeros((1, target_t), dtype=np.int32)
    key_padding_mask[:, :seq_len] = 1
    # mlpackage 内部 fp16, 但 input 接口是 fp32; cast 一下
    t0 = time.time()
    cm_pred = mlmodel.predict({
        "hidden_states": hidden_states_padded.astype(np.float32),
        "key_padding_mask": key_padding_mask,
    })
    t_cm = time.time() - t0
    cm_out = cm_pred["last_hidden_state"]
    print(f"[coreml_be] out shape={cm_out.shape}, dtype={cm_out.dtype}, time={t_cm*1000:.1f}ms")

    # 6. 对比有效部分 (前 seq_len 帧)
    onnx_valid = onnx_out[:, :seq_len, :].astype(np.float32)
    cm_valid = cm_out[:, :seq_len, :].astype(np.float32)

    diff = np.abs(onnx_valid - cm_valid)
    max_abs = diff.max()
    mean_abs = diff.mean()
    # cosine sim 每帧每特征展平
    a = onnx_valid.flatten()
    b = cm_valid.flatten()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"\n=== parity ===")
    print(f"  ONNX vs CoreML on valid {seq_len} frames:")
    print(f"  max_abs_diff = {max_abs:.4e}")
    print(f"  mean_abs_diff = {mean_abs:.4e}")
    print(f"  cosine_sim = {cos:.6f}")
    print(f"  threshold: cos > 0.999, max_abs_diff < 1e-2")
    ok = cos > 0.999 and max_abs < 1e-2
    print(f"  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
