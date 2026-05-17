"""
PoC: 子进程隔离 mlpackage backend.

假设: SIGSEGV 根因是同一 Python 进程内 CoreML libdispatch worker 跟 sherpa-onnx 抢资源。
做法: 把 mlpackage backend 隔离到独立子进程, 父进程只跟它 IPC, 父进程加载 sherpa-onnx。

最小验证:
1. 父进程 import sherpa_onnx + load embedding model (跟生产路径一样)
2. 父进程 spawn 子进程, 子进程加载 mlpackage + 跑 N 次 dummy predict
3. 父进程跑 sherpa embedding 推理
4. 父进程通过 Pipe send numpy 给子进程, recv 输出
5. 看是否 SIGSEGV

跑法:
    venv/bin/python spikes/qwen3_mac_hw_accel/phase3_backend/poc_subprocess_isolation.py \
        --mlpackage models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.mlpackage \
        --embedding models/qwen3_diarize/sherpa/nemo-titanet-small/embedding.onnx
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", required=True)
    ap.add_argument("--embedding", required=True, help="sherpa nemo-titanet embedding.onnx")
    ap.add_argument("--time", type=int, default=520)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=50, help="父子各跑多少次推理")
    return ap.parse_args()


def backend_worker_loop(mlpackage_path: str, conn):
    """子进程: 加载 mlpackage, 循环等输入 → predict → 发输出。"""
    import coremltools as ct

    print(f"[child] load mlpackage from {mlpackage_path} ...", flush=True)
    t0 = time.time()
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[child] mlmodel loaded in {time.time()-t0:.1f}s", flush=True)
    conn.send({"event": "ready"})

    while True:
        msg = conn.recv()
        if msg is None:
            print("[child] received None, exit", flush=True)
            break
        try:
            pred = mlmodel.predict({
                "hidden_states": msg["hidden_states"],
                "key_padding_mask": msg["key_padding_mask"],
            })
            conn.send({"event": "result", "output": np.asarray(pred["last_hidden_state"])})
        except Exception as e:
            conn.send({"event": "error", "error": str(e)})
    print("[child] bye", flush=True)


def main():
    args = parse_args()
    mp.set_start_method("spawn")  # spawn = clean process, no fork-inherited state

    # ----- 父进程: 加载 sherpa-onnx 跟生产路径一致 -----
    print("[parent] import sherpa_onnx + load embedding ...", flush=True)
    import sherpa_onnx

    t0 = time.time()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.embedding,
        num_threads=4,
        provider="cpu",
        debug=False,
    )
    assert cfg.validate(), "sherpa cfg invalid"
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    print(f"[parent] sherpa loaded in {time.time()-t0:.1f}s", flush=True)

    # ----- 父进程: spawn backend 子进程 -----
    parent_conn, child_conn = mp.Pipe()
    child = mp.Process(target=backend_worker_loop, args=(args.mlpackage, child_conn))
    child.start()
    print(f"[parent] spawned child pid={child.pid}", flush=True)

    # 等子进程 ready
    msg = parent_conn.recv()
    assert msg["event"] == "ready", f"unexpected: {msg}"
    print("[parent] child ready, start parallel work", flush=True)

    # ----- 主循环: 父跑 sherpa embedding + 子跑 backend predict, 交替 -----
    h = np.random.randn(1, args.time, args.dim).astype(np.float32)
    m = np.ones((1, args.time), dtype=np.int32)
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s 静音

    t_total = time.time()
    for i in range(args.runs):
        # 父: sherpa embedding (跟生产里 cluster_centroid_merge 一样)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=dummy_audio)
        stream.input_finished()
        emb = extractor.compute(stream)

        # 子: mlpackage predict
        parent_conn.send({"hidden_states": h, "key_padding_mask": m})
        result = parent_conn.recv()
        if result["event"] != "result":
            print(f"[parent] child error: {result}", flush=True)
            break

        if (i + 1) % 10 == 0:
            print(f"[parent] iter {i+1}/{args.runs} OK (sherpa emb len={len(emb)}, be out shape={result['output'].shape})", flush=True)

    elapsed = time.time() - t_total
    print(f"\n[parent] {args.runs} iters in {elapsed:.1f}s, avg {elapsed/args.runs*1000:.1f} ms/iter")

    # ----- 收尾 -----
    parent_conn.send(None)
    child.join(timeout=5)
    print(f"[parent] child exit code = {child.exitcode}")

    if child.exitcode == 0:
        print("\n✅ PoC PASS: 子进程隔离避开 SIGSEGV")
        sys.exit(0)
    else:
        print(f"\n❌ PoC FAIL: child rc={child.exitcode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
