"""验证 ORT CUDA Python API 直跑 pyannote-seg + TitaNet 是否跟 llama.cpp CUDA 撞.

跑两种顺序:
  --order=llm_first    LLM 先 init, 然后 ORT CUDA session
  --order=ort_first    ORT CUDA session 先 init, 然后 LLM
  --order=interleaved  LLM init + ORT session + ORT inference + LLM inference 交错

每个阶段都打印 "[step:ok] <name>"; segfault 会让 process 死, 缺哪行就是哪步崩.

成功 → 候选 1 可行, 接着写 FastClustering Python 版接入项目.
失败 → 走候选 2 (pyannote-audio) 或 候选 3 (进程隔离).
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step(name: str):
    print(f"[step:ok] {name}", flush=True)


def load_audio_np():
    """加载 60s podcast 转 16k mono float32."""
    import soundfile as sf
    import numpy as np
    audio_path = ROOT / "tests/fixtures/audio/podcast_2speakers_60s.wav"
    y, sr = sf.read(str(audio_path))
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def init_llm():
    """加载 + warmup llama.cpp CUDA, 不做真实 decode."""
    from src.core.vendor.qwen_asr_gguf.inference.asr import QwenASREngine
    from src.core.qwen3.asr import build_engine_config
    cfg = build_engine_config(
        model_dir=str(ROOT / "models/qwen3_diarize/Qwen3-ASR-1.7B"),
        llm_use_gpu=1,
        onnx_provider="CPU",  # 只测 LLM CUDA 跟 ORT CUDA 撞, encoder 走 CPU
    )
    engine = QwenASREngine(config=cfg)
    return engine


def init_ort_pyannote():
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(ROOT / "models/qwen3_diarize/sherpa/pyannote-segmentation-3.0/model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return sess


def init_ort_titanet():
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(ROOT / "models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return sess


def run_ort_pyannote(sess, audio):
    """pyannote-seg 输入: (batch=1, channels=1, samples_chunk).
    模型是固定 10s chunk(160000 samples @ 16k). 跑第一个 10s chunk."""
    import numpy as np
    chunk = audio[:160000]
    if len(chunk) < 160000:
        chunk = np.pad(chunk, (0, 160000 - len(chunk)))
    x = chunk.reshape(1, 1, -1).astype(np.float32)
    t0 = time.time()
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    dt = time.time() - t0
    print(f"  pyannote-seg output shape: {out[0].shape}, elapsed={dt*1000:.1f}ms")


def run_ort_titanet(sess, audio):
    """TitaNet embedding. NeMo 期望 rank-3 (batch, channels, samples). 喂 5s 片段."""
    import numpy as np
    # 探测真实 input rank, 兼容 (B, C, T) 或 (B, T) variant
    in_meta = sess.get_inputs()[0]
    print(f"  titanet input: {in_meta.name} shape={in_meta.shape}")
    rank = len(in_meta.shape)
    if rank == 3:
        chunk = audio[:80000].reshape(1, 1, -1).astype(np.float32)
    else:
        chunk = audio[:80000].reshape(1, -1).astype(np.float32)
    inputs = {in_meta.name: chunk}
    if len(sess.get_inputs()) > 1:
        length_meta = sess.get_inputs()[1]
        inputs[length_meta.name] = np.array([chunk.shape[-1]], dtype=np.int64)
    t0 = time.time()
    out = sess.run(None, inputs)
    dt = time.time() - t0
    print(f"  titanet output shape: {out[0].shape}, elapsed={dt*1000:.1f}ms")


def run_llm_dummy(engine):
    """跑完整 ASR pipeline (encoder + LLM), 用项目现成的 run_asr 接口."""
    from src.core.qwen3.asr import run_asr
    audio_path = ROOT / "tests/fixtures/audio/podcast_2speakers_60s.wav"
    t0 = time.time()
    res = engine_or_run_asr_path(engine, str(audio_path))
    elapsed = time.time() - t0
    txt = getattr(res, "text", "") or ""
    print(f"  asr full pipeline elapsed={elapsed:.2f}s text={txt[:40]!r}")


def engine_or_run_asr_path(engine, audio_path):
    """复用 src.core.qwen3.asr.run_asr 但要传 engine 不是 audio_file 唯一参数."""
    from src.core.qwen3.asr import run_asr
    return run_asr(audio_path, engine, language=None, temperature=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--order",
        choices=["llm_first", "ort_first", "interleaved"],
        default="llm_first",
    )
    args = parser.parse_args()

    step(f"start, order={args.order}")
    audio, sr = load_audio_np()
    step(f"audio loaded: {len(audio)} samples @ {sr}Hz")

    if args.order == "llm_first":
        engine = init_llm()
        step("llm CUDA init")
        run_llm_dummy(engine)
        step("llm CUDA inference")

        sess_seg = init_ort_pyannote()
        step("ort pyannote-seg CUDA init")
        run_ort_pyannote(sess_seg, audio)
        step("ort pyannote-seg inference")

        sess_emb = init_ort_titanet()
        step("ort titanet CUDA init")
        run_ort_titanet(sess_emb, audio)
        step("ort titanet inference")

    elif args.order == "ort_first":
        sess_seg = init_ort_pyannote()
        step("ort pyannote-seg CUDA init")
        run_ort_pyannote(sess_seg, audio)
        step("ort pyannote-seg inference")

        sess_emb = init_ort_titanet()
        step("ort titanet CUDA init")
        run_ort_titanet(sess_emb, audio)
        step("ort titanet inference")

        engine = init_llm()
        step("llm CUDA init")
        run_llm_dummy(engine)
        step("llm CUDA inference")

    else:  # interleaved
        engine = init_llm()
        step("llm CUDA init")
        sess_seg = init_ort_pyannote()
        step("ort pyannote-seg CUDA init")
        sess_emb = init_ort_titanet()
        step("ort titanet CUDA init")
        run_ort_pyannote(sess_seg, audio)
        step("ort pyannote-seg inference")
        run_llm_dummy(engine)
        step("llm CUDA inference")
        run_ort_titanet(sess_emb, audio)
        step("ort titanet inference")

    step("ALL CLEAR — no segfault, ORT CUDA + llama.cpp CUDA can coexist")


if __name__ == "__main__":
    main()
