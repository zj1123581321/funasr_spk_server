"""Worker for concurrent transcribe probe.

调用: python scripts/_remote_concurrent_worker.py <worker_id> <audio_path> <barrier_prefix>

每个 worker 独立加载 model, 写 <barrier_prefix>.ready.<worker_id> 标记 ready,
等 <barrier_prefix>.go 出现后同时开始 transcribe, 输出 WORKER=<id> wall=... RTF=... 行.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 让 ASR encoder 走 CUDA, 跟单任务 probe 一致
os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")
os.environ.setdefault("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")


async def main() -> None:
    if len(sys.argv) != 4:
        print("usage: worker.py <worker_id> <audio_path> <barrier_prefix>", file=sys.stderr)
        sys.exit(2)
    worker_id = sys.argv[1]
    audio_path = sys.argv[2]
    barrier = sys.argv[3]

    from src.core.qwen3.diarize_ort import reset_session_cache
    from src.core.qwen3_transcriber import (
        get_qwen3_transcriber,
        reset_qwen3_transcriber_singleton,
    )

    reset_qwen3_transcriber_singleton()
    reset_session_cache()
    tx = get_qwen3_transcriber()
    await tx.initialize()

    # 标记 ready, 等 go signal 同时启动 (避免一个先开跑 5s 的偏差)
    ready_path = f"{barrier}.ready.{worker_id}"
    go_path = f"{barrier}.go"
    Path(ready_path).touch()
    while not Path(go_path).exists():
        await asyncio.sleep(0.1)

    t0 = time.time()
    result, _ = await tx.transcribe(
        audio_path=audio_path,
        task_id=f"concurrent-{worker_id}",
        progress_callback=None,
        output_format="json",
    )
    wall = time.time() - t0
    rtf = wall / result.duration if result.duration else 0
    print(
        f"WORKER={worker_id} wall={wall:.2f}s RTF={rtf:.3f} "
        f"speakers={result.speakers} segs={len(result.segments)}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
