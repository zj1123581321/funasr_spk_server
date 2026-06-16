"""远端 diarize parity 探测脚本 — sherpa vs ort_cuda 在不同 cluster_threshold 下的对比.

用法 (远端):
  bash scripts/_remote_run_provider.sh ort_cuda parity-probe  # 不适用, 它跑死命令
  # 推荐:
  export LD_LIBRARY_PATH=...; python scripts/_remote_diarize_parity_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import config
from src.core.qwen3.diarize import run_diarization
from src.core.qwen3.diarize_ort import (
    reset_session_cache,
    run_diarization_ort_cuda,
)


def _summarize(label: str, turns: list[dict]) -> None:
    speakers = sorted({t["speaker"] for t in turns})
    total = sum(t["end"] - t["start"] for t in turns)
    print(
        f"  {label}: turns={len(turns)} speakers={speakers} active={total:.2f}s"
    )
    for t in turns[:5]:
        print(
            f"    start={t['start']:.2f} end={t['end']:.2f} spk={t['speaker']}"
        )
    if len(turns) > 5:
        print(f"    ... (+{len(turns) - 5})")


def main() -> None:
    audios = [
        ("60s", "tests/fixtures/audio/podcast_2speakers_60s.wav"),
        ("300s", "tests/fixtures/audio/podcast_2speakers_300s.wav"),
        ("1800s", "tests/fixtures/audio/podcast_2speakers_1800s.wav"),
    ]

    # Step 1: 跳过 threshold sweep (上轮已确认 thr=0.9 + complete linkage 在 60s 上 2 spk)
    # Step 2: 60s/300s/1800s 各跑 sherpa + ort_cuda 拿 wall (threshold=0.9 跟 sherpa config 一致)
    print("\n========== full-length perf (sherpa baseline vs ort_cuda thr=0.9) ==========", flush=True)
    best_thr = 0.9
    for label, audio in audios:
        if not Path(audio).exists():
            print(f"== {label}: 缺音频, 跳过 ==", flush=True)
            continue
        print(f"\n----- {label} -----", flush=True)
        t0 = time.time()
        sh = run_diarization(
            audio,
            segmentation_model=config.qwen3.segmentation_model,
            embedding_model=config.qwen3.embedding_model,
            num_speakers=None,
            cluster_threshold=config.qwen3.cluster_threshold,
            num_threads=4,
            provider="cpu",
        )
        sh_wall = time.time() - t0
        _summarize(f"sherpa cpu wall={sh_wall:.2f}s", sh)

        reset_session_cache()
        t0 = time.time()
        ot = run_diarization_ort_cuda(
            audio,
            segmentation_model=config.qwen3.segmentation_model,
            embedding_model=config.qwen3.embedding_model,
            num_speakers=None,
            cluster_threshold=best_thr,
        )
        wall = time.time() - t0
        _summarize(f"ort_cuda thr={best_thr} wall={wall:.2f}s", ot)


if __name__ == "__main__":
    main()
