#!/usr/bin/env python3
"""sherpa-onnx 说话人分离 benchmark 脚本。

用法:
    python benchmark/diarize_bench.py <audio_path> [--num-speakers 2]
    python benchmark/diarize_bench.py audio.wav --cluster-threshold 0.7

输出:
    - stdout: JSON 包含 turns + 指标
    - stderr: 进度/汇总日志 (音频时长, diarize 耗时, RTF, 峰值 RSS, turns/speakers 数)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# 让 src/ 在 import path 上,直接复用 diarize 模块
_SPIKE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPIKE_ROOT))

import soundfile as sf  # noqa: E402

from src.diarize import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEGMENTATION_MODEL,
    run_diarization,
)


def _audio_duration_sec(path: str) -> float:
    """soundfile 直接读元数据拿时长,避免把全部 PCM load 进内存。"""
    info = sf.info(path)
    return info.frames / info.samplerate


def _peak_rss_bytes() -> int:
    """跨平台峰值 RSS。

    macOS: getrusage().ru_maxrss 单位是 bytes。
    Linux: getrusage().ru_maxrss 单位是 KB,需要 *1024。
    fallback: psutil 当前 RSS (非峰值)。
    """
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(ru)  # bytes
        return int(ru) * 1024  # KB -> bytes (Linux)
    except Exception:
        try:
            import psutil

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="sherpa-onnx diarization benchmark")
    parser.add_argument("audio_path", help="输入音频文件")
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="已知说话人数则锁定;传 -1 表示未知,启用 cluster_threshold 自动聚类",
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--segmentation-model", default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    num_speakers = None if args.num_speakers < 0 else args.num_speakers

    print(f"[bench] audio_path = {args.audio_path}", file=sys.stderr)
    print(
        f"[bench] num_speakers = {num_speakers}, "
        f"cluster_threshold = {args.cluster_threshold}",
        file=sys.stderr,
    )

    audio_duration = _audio_duration_sec(args.audio_path)
    print(f"[bench] audio_duration = {audio_duration:.3f}s", file=sys.stderr)

    t0 = time.perf_counter()
    turns = run_diarization(
        audio_path=args.audio_path,
        num_speakers=num_speakers,
        cluster_threshold=args.cluster_threshold,
        segmentation_model=args.segmentation_model,
        embedding_model=args.embedding_model,
        num_threads=args.num_threads,
        provider=args.provider,
    )
    diarize_time = time.perf_counter() - t0

    rtf = diarize_time / audio_duration if audio_duration > 0 else float("inf")
    peak_rss = _peak_rss_bytes()
    peak_rss_mb = peak_rss / (1024 * 1024) if peak_rss > 0 else -1.0
    unique_speakers = len({t["speaker"] for t in turns})

    summary = {
        "audio_path": args.audio_path,
        "audio_duration_sec": round(audio_duration, 3),
        "diarize_time_sec": round(diarize_time, 3),
        "rtf": round(rtf, 4),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "num_turns": len(turns),
        "num_unique_speakers": unique_speakers,
        "num_speakers_arg": num_speakers,
        "cluster_threshold": args.cluster_threshold,
        "turns": turns,
    }

    # stderr 人类可读
    print(f"[bench] diarize_time = {diarize_time:.3f}s", file=sys.stderr)
    print(f"[bench] RTF = {rtf:.4f}", file=sys.stderr)
    print(f"[bench] peak_rss = {peak_rss_mb:.2f} MB", file=sys.stderr)
    print(
        f"[bench] turns = {len(turns)}, unique_speakers = {unique_speakers}",
        file=sys.stderr,
    )

    # stdout 机读 JSON
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
