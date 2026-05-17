"""
PoC v2: 精确复现生产路径加载顺序, 验证 autoreleasepool 是否能避开 SIGSEGV.

生产顺序 (跟 N=2 长音频 crash 一致):
  1. import coremltools, 加载 mlpackage
  2. warmup predict (这里曾经注册 lingering buffer)
  3. import sherpa_onnx (这里曾 SIGSEGV)
  4. 加载 sherpa embedding extractor
  5. 跑 N 次 predict + embedding 交替

变量: 在 step 2 warmup 用 / 不用 autoreleasepool, 看是否避开崩。
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
    ap.add_argument("--no-autoreleasepool", action="store_true",
                    help="不用 autoreleasepool, 应该复现 SIGSEGV")
    return ap.parse_args()


def main():
    args = parse_args()
    use_pool = not args.no_autoreleasepool
    print(f"[mode] use_autoreleasepool = {use_pool}", flush=True)

    import objc
    import coremltools as ct

    # ----- Step 1: 加载 mlpackage -----
    print(f"[step1] load mlpackage", flush=True)
    t0 = time.time()
    if use_pool:
        with objc.autorelease_pool():
            mlmodel = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)
    else:
        mlmodel = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[step1] mlpackage in {time.time()-t0:.1f}s", flush=True)

    # ----- Step 2: warmup predict (生产 encoder __init__ encode dummy_wav 触发这一步) -----
    h = np.random.randn(1, args.time, args.dim).astype(np.float32)
    m = np.ones((1, args.time), dtype=np.int32)
    print(f"[step2] warmup predict", flush=True)
    t0 = time.time()
    if use_pool:
        with objc.autorelease_pool():
            pred = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
            warmup_out = np.asarray(pred["last_hidden_state"]).copy()
            del pred
    else:
        pred = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
        warmup_out = pred["last_hidden_state"]
    print(f"[step2] warmup done in {time.time()-t0:.1f}s, out shape={warmup_out.shape}", flush=True)

    # ----- Step 3-4: import + 加载 sherpa-onnx (生产里 SIGSEGV 触发点) -----
    print(f"[step3] import sherpa_onnx ... ⚠️ 之前 SIGSEGV 触发点", flush=True)
    t0 = time.time()
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.embedding, num_threads=4, provider="cpu", debug=False,
    )
    assert cfg.validate()
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    print(f"[step3] sherpa loaded in {time.time()-t0:.1f}s ✅ 通过 SIGSEGV 点", flush=True)

    # ----- Step 5: 循环 predict + embedding (跟生产 ASR loop 同) -----
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
    print(f"[step5] {args.runs} iters predict + embedding", flush=True)
    t0 = time.time()
    for i in range(args.runs):
        # sherpa
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=dummy_audio)
        stream.input_finished()
        _ = extractor.compute(stream)

        # mlpackage
        if use_pool:
            with objc.autorelease_pool():
                pred = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
                _out = np.asarray(pred["last_hidden_state"]).copy()
                del pred
        else:
            pred = mlmodel.predict({"hidden_states": h, "key_padding_mask": m})
            _out = pred["last_hidden_state"]

        if (i + 1) % 10 == 0:
            print(f"  iter {i+1}/{args.runs} OK", flush=True)

    elapsed = time.time() - t0
    print(f"\n✅ {args.runs} iters in {elapsed:.1f}s, avg {elapsed/args.runs*1000:.1f} ms/iter")
    print(f"autoreleasepool {'WORK' if use_pool else 'OFF (baseline)'} — 没崩, 生产顺序 reproduce OK")


if __name__ == "__main__":
    main()
