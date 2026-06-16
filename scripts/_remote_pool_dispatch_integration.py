"""Integration probe: 真实 get_qwen3_pool_transcriber() dispatch 走 InProcPool 跑 1800s × 2.

跟 _remote_single_proc_concurrent_probe.py 的差异:
- 那个: 直接 new Qwen3DiarizeTranscriber × 2 (绕过 dispatch, 证明算法可行)
- 这个: 走 get_qwen3_pool_transcriber() 真实路径, 验证:
  1. detect_runtime() 在 cuda 上返回 CudaRuntime
  2. dispatch 选 Qwen3InProcPool 而非 multi-process
  3. pool.initialize + 2 并发 transcribe 跟 probe 同样数据 (parity)

需要 env:
  FUNASR_QWEN3_POOL_SIZE=2  # 强制 pool=2
  FUNASR_QWEN3_DIARIZE_BACKEND=ort_cuda  # 强制 ORT CUDA backend
  FUNASR_QWEN3_ASR_ENCODER_PROVIDER=cuda  # ASR encoder 走 CUDA
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("FUNASR_QWEN3_POOL_SIZE", "2")
os.environ.setdefault("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")
os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")


async def main() -> None:
    audio_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "tests/fixtures/audio/podcast_2speakers_1800s.wav"
    )
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    from src.core.qwen3_pool_transcriber import (
        get_qwen3_pool_transcriber,
        reset_qwen3_pool_singleton,
    )
    from src.core.qwen3_inproc_pool import Qwen3InProcPool
    from src.core.runtime import describe_runtime, detect_runtime

    runtime = detect_runtime()
    runtime.validate()
    print(f"[probe] runtime = {describe_runtime(runtime)}", flush=True)

    reset_qwen3_pool_singleton()
    pool = get_qwen3_pool_transcriber()
    print(f"[probe] pool class = {type(pool).__name__}", flush=True)
    if not isinstance(pool, Qwen3InProcPool):
        print(
            f"[probe] ERROR: cuda 路径应返回 Qwen3InProcPool, 但 dispatch 返回了 {type(pool).__name__}",
            flush=True,
        )
        sys.exit(1)
    print(f"[probe] pool.pool_size = {pool.pool_size}", flush=True)

    print(f"[probe] initialize pool (build + init {pool.pool_size} transcribers)...", flush=True)
    t_init = time.time()
    await pool.initialize()
    print(f"[probe] pool ready in {time.time() - t_init:.2f}s", flush=True)

    print(
        f"[probe] kickoff {count} concurrent pool.transcribe on {audio_path}",
        flush=True,
    )

    async def run_one(idx: int) -> dict:
        t0 = time.time()
        result, _ = await pool.transcribe(
            audio_path=audio_path,
            task_id=f"dispatch-{idx}",
            progress_callback=None,
            output_format="json",
        )
        wall = time.time() - t0
        rtf = wall / result.duration if result.duration else 0
        return {
            "task_id": f"dispatch-{idx}",
            "wall": wall,
            "rtf": rtf,
            "speakers": list(result.speakers),
            "segs": len(result.segments),
        }

    t_kick = time.time()
    results = await asyncio.gather(
        *(run_one(i + 1) for i in range(count)), return_exceptions=True
    )
    total_wall = time.time() - t_kick

    print(f"[probe] TOTAL_WALL={total_wall:.2f}s", flush=True)
    for r in results:
        if isinstance(r, Exception):
            print(f"[probe] TASK_FAILED: {type(r).__name__}: {r}", flush=True)
        else:
            print(
                f"[probe] TASK_OK {r['task_id']} wall={r['wall']:.2f}s "
                f"RTF={r['rtf']:.3f} speakers={r['speakers']} segs={r['segs']}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
