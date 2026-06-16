"""
Step 3: 加载 mlpackage + 推理 + RSS profiling。

用法:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/verify_mlpackage_load.py \
        --mlpackage models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage \
        --time 390 --runs 20

ANE 占用率测量 (并行另开 shell):
    sudo powermetrics --samplers ane_power -i 1000 -n 30
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np


def rss_mb() -> float:
    """ru_maxrss 在 macOS 上单位是字节, Linux 上是 KB。按平台判别。"""
    import sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / 1024 / 1024
    return r / 1024


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--time", type=int, default=390)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=20, help="inference 次数 (warmup=2 + measure)")
    ap.add_argument("--units", choices=["CPU_AND_NE", "CPU_AND_GPU", "ALL", "CPU_ONLY"], default="CPU_AND_NE")
    return ap.parse_args()


def main():
    args = parse_args()
    p = Path(args.mlpackage)
    assert p.exists(), f"mlpackage not found: {p}"

    rss0 = rss_mb()
    print(f"[rss] baseline (Python imports already done): {rss0:.0f} MB")

    import coremltools as ct

    print(f"[load] ct.models.MLModel({p}, compute_units={args.units}) ...")
    t0 = time.time()
    units_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    mlmodel = ct.models.MLModel(str(p), compute_units=units_map[args.units])
    load_s = time.time() - t0
    rss_after_load = rss_mb()
    print(f"[load] done in {load_s:.1f}s, ΔRSS={rss_after_load - rss0:.0f} MB (total {rss_after_load:.0f} MB)")

    # Dummy inputs
    h = np.random.randn(1, args.time, args.dim).astype(np.float32)
    m = np.ones((1, args.time), dtype=np.int32)
    inputs = {"hidden_states": h, "key_padding_mask": m}

    # Cold + warmup
    print(f"[warmup] cold start prediction ...")
    t0 = time.time()
    out = mlmodel.predict(inputs)
    cold_s = time.time() - t0
    rss_after_first = rss_mb()
    print(f"[warmup] cold = {cold_s:.2f}s, ΔRSS (load → first inference) = {rss_after_first - rss_after_load:.0f} MB")
    print(f"[warmup] output keys: {list(out.keys())}")
    for k, v in out.items():
        print(f"        {k}: shape={v.shape}, dtype={v.dtype}")

    print(f"[bench] {args.runs} runs ...")
    t0 = time.time()
    for _ in range(args.runs):
        _ = mlmodel.predict(inputs)
    elapsed = time.time() - t0
    rss_after_bench = rss_mb()
    print(f"[bench] {args.runs} runs in {elapsed:.2f}s, avg = {elapsed/args.runs*1000:.1f} ms/run")
    print(f"[rss] final = {rss_after_bench:.0f} MB (Δ from baseline = {rss_after_bench - rss0:.0f} MB)")

    print("\n=== summary ===")
    print(f"  mlpackage      : {p.name}")
    print(f"  size on disk   : {sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1024 / 1024:.0f} MB")
    print(f"  load time      : {load_s:.1f}s")
    print(f"  cold predict   : {cold_s:.2f}s")
    print(f"  warm avg       : {elapsed/args.runs*1000:.1f} ms")
    print(f"  RSS Δ          : {rss_after_bench - rss0:.0f} MB")
    print(f"  compute_units  : {args.units}")
    print(f"  (run powermetrics -s ane_power 并行测 ANE peak/mean)")


if __name__ == "__main__":
    main()
