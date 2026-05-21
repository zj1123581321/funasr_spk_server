"""远端 e2e (ASR + diarize gather) wall RTF probe.

跑 60s/300s/1800s × {sherpa, ort_cuda} 两个 backend, 通过 Qwen3DiarizeTranscriber
端到端 transcribe() 调用, 含全部后处理 (spurious filter / cluster_centroid_merge /
short_segment_guard / silence_align).

用法 (远端):
  export LD_LIBRARY_PATH=...; python scripts/_remote_e2e_wall_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 启用 CUDA encoder 走 GPU (跟 8 vCPU 实测脚本对齐)
os.environ.setdefault("FUNASR_QWEN3_ASR_ENCODER_PROVIDER", "cuda")


async def run_once(audio_path: str, backend: str):
    from src.core.config import config
    from src.core.qwen3.diarize_ort import reset_session_cache
    from src.core.qwen3_transcriber import (
        get_qwen3_transcriber,
        reset_qwen3_transcriber_singleton,
    )

    reset_qwen3_transcriber_singleton()
    reset_session_cache()
    os.environ["FUNASR_QWEN3_DIARIZE_BACKEND"] = backend

    tx = get_qwen3_transcriber()
    await tx.initialize()

    t0 = time.time()
    result, raw = await tx.transcribe(
        audio_path=audio_path,
        task_id=f"probe-{backend}-{Path(audio_path).stem}",
        progress_callback=None,
        output_format="json",
    )
    wall = time.time() - t0
    return wall, result


async def main() -> None:
    audios = [
        ("60s", "tests/fixtures/audio/podcast_2speakers_60s.wav"),
        ("300s", "tests/fixtures/audio/podcast_2speakers_300s.wav"),
        ("1800s", "tests/fixtures/audio/podcast_2speakers_1800s.wav"),
    ]
    for label, audio in audios:
        if not Path(audio).exists():
            print(f"== {label}: 缺 {audio}", flush=True)
            continue
        print(f"\n========== {label} ==========", flush=True)
        for backend in ("sherpa", "ort_cuda"):
            try:
                wall, result = await run_once(audio, backend)
                dur = result.duration
                rtf = wall / dur if dur else 0
                print(
                    f"  {backend}: wall={wall:.2f}s RTF={rtf:.3f} "
                    f"speakers={result.speakers} segments={len(result.segments)}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  {backend}: ERROR {exc!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
