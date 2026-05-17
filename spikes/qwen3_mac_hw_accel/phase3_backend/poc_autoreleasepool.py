"""
PoC: PyObjC autorelease_pool 包 mlpackage predict, 看能否避开 SIGSEGV.

理论: MLE5ExecutionStream lingering reset 是 dispatch_async 调度,
跟 ObjC autorelease 不是同一机制 — 但试一下, 万一 work 比子进程方案轻 100x。

跑法:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/poc_autoreleasepool.py \
        --mlpackage models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage \
        --embedding models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx
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
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--time", type=int, default=520)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()

    import objc
    print("pyobjc:", objc.__version__)

    # 加载 mlpackage 在 autoreleasepool 内
    import coremltools as ct
    print(f"[load] mlpackage from {args.mlpackage}", flush=True)
    t0 = time.time()
    with objc.autorelease_pool():
        mlmodel = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] mlpackage in {time.time()-t0:.1f}s", flush=True)

    # 加载 sherpa-onnx (跟生产同顺序: ASR engine init 后才加载 sherpa)
    print("[load] sherpa-onnx", flush=True)
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.embedding, num_threads=4, provider="cpu", debug=False,
    )
    assert cfg.validate()
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    print("[load] sherpa OK", flush=True)

    # 交替推理
    h = np.random.randn(1, args.time, args.dim).astype(np.float32)
    m = np.ones((1, args.time), dtype=np.int32)
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)

    print(f"[bench] {args.runs} iters with autoreleasepool", flush=True)
    t0 = time.time()
    for i in range(args.runs):
        # sherpa
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=dummy_audio)
        stream.input_finished()
        _ = extractor.compute(stream)

        # mlpackage predict 在 autoreleasepool 里
        with objc.autorelease_pool():
            pred = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
            out = np.asarray(pred["last_hidden_state"]).copy()
            del pred

        if (i + 1) % 10 == 0:
            print(f"  iter {i+1}/{args.runs} OK out={out.shape}", flush=True)

    elapsed = time.time() - t0
    print(f"\n✅ {args.runs} iters in {elapsed:.1f}s, avg {elapsed/args.runs*1000:.1f} ms/iter")
    print("autoreleasepool 方案 PASS (没崩)")


if __name__ == "__main__":
    main()
