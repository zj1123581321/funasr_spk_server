"""Dump intermediate data (ASR chunks + diarize turns + ffmpeg speech_regions) once.

跑一次, 后续所有 spike 迭代直接读 JSON, 不用再加载 ~2GB 模型.

Usage:
    venv/bin/python spikes/qwen3_silence_align/scripts/dump_intermediate.py \
        --audio tests/fixtures/audio/podcast_2speakers_60s.wav \
        --out spikes/qwen3_silence_align/data/podcast_60s.intermediate.json

Output JSON 结构:
{
  "audio_path": "tests/fixtures/audio/podcast_2speakers_60s.wav",
  "audio_duration": 59.96,
  "asr": {
    "text": "...",
    "chunks": [{"text": "...", "start": 0.0, "end": 40.0, "index": 0}, ...],
    "rtf": 0.12
  },
  "diarize_turns": [{"start": 0.0, "end": 9.2, "speaker": 0}, ...],
  "speech_regions": [{"start": 0.0, "end": 9.1}, ...]
}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# 让脚本能 import src.core.*
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def ffmpeg_speech_regions(
    audio_path: Path,
    audio_duration: float,
    noise_db: str = "-35dB",
    min_silence_sec: float = 0.8,
) -> list[dict]:
    """Use ffmpeg silencedetect as a lightweight VAD boundary proxy.

    抄自 tests/manual/server/qwen3_long_audio_poc.py:70, spike 内自包含避免循环依赖.
    返回 speech_regions (说话区间), 不是 silence regions.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={noise_db}:d={min_silence_sec}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    stderr = proc.stderr or ""

    silence_starts: list[float] = []
    silences: list[tuple[float, float]] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start: ([0-9.]+)", line)
        if m_start:
            silence_starts.append(float(m_start.group(1)))
            continue
        m_end = re.search(r"silence_end: ([0-9.]+)", line)
        if m_end and silence_starts:
            start = silence_starts.pop(0)
            end = float(m_end.group(1))
            silences.append((start, end))
    for start in silence_starts:
        silences.append((start, audio_duration))

    regions: list[dict] = []
    cursor = 0.0
    for start, end in silences:
        if start > cursor:
            regions.append({"start": cursor, "end": start})
        cursor = max(cursor, end)
    if cursor < audio_duration:
        regions.append({"start": cursor, "end": audio_duration})
    return [r for r in regions if r["end"] - r["start"] > 0.1]


def get_audio_duration(audio_path: Path) -> float:
    """ffprobe 拿音频时长."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--asr-model-dir",
        type=Path,
        default=_REPO_ROOT / "models" / "qwen3_diarize" / "Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=_REPO_ROOT / "models" / "qwen3_diarize" / "sherpa" / "pyannote-segmentation-3.0" / "model.onnx",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=_REPO_ROOT / "models" / "qwen3_diarize" / "sherpa" / "nemo-titanet-small" / "embedding.onnx",
    )
    parser.add_argument("--noise-db", default="-35dB")
    parser.add_argument("--min-silence-sec", type=float, default=0.8)
    parser.add_argument("--cluster-threshold", type=float, default=0.9)
    args = parser.parse_args()

    audio_path: Path = args.audio.resolve()
    if not audio_path.exists():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dump] audio = {audio_path}")
    t0 = time.time()
    duration = get_audio_duration(audio_path)
    print(f"[dump] duration = {duration:.2f}s")

    # 1. VAD speech regions
    t1 = time.time()
    speech_regions = ffmpeg_speech_regions(
        audio_path,
        audio_duration=duration,
        noise_db=args.noise_db,
        min_silence_sec=args.min_silence_sec,
    )
    print(f"[dump] speech_regions = {len(speech_regions)} (elapsed {time.time() - t1:.2f}s)")

    # 2. ASR
    from src.core.qwen3.asr import build_engine, run_asr
    t1 = time.time()
    print(f"[dump] loading ASR engine from {args.asr_model_dir} ...")
    engine = build_engine(str(args.asr_model_dir))
    print(f"[dump] engine loaded in {time.time() - t1:.1f}s")
    t1 = time.time()
    asr_result = run_asr(str(audio_path), engine)
    print(f"[dump] ASR done in {time.time() - t1:.1f}s, rtf={asr_result.rtf:.3f}, chunks={len(asr_result.chunks)}")

    # 3. Diarize
    from src.core.qwen3.diarize import run_diarization
    t1 = time.time()
    turns = run_diarization(
        str(audio_path),
        num_speakers=None,
        cluster_threshold=args.cluster_threshold,
        segmentation_model=str(args.segmentation_model),
        embedding_model=str(args.embedding_model),
    )
    print(f"[dump] diarize done in {time.time() - t1:.1f}s, turns={len(turns)}")

    # 4. dump
    payload = {
        "audio_path": str(audio_path.relative_to(_REPO_ROOT)) if audio_path.is_relative_to(_REPO_ROOT) else str(audio_path),
        "audio_duration": duration,
        "vad_params": {
            "noise_db": args.noise_db,
            "min_silence_sec": args.min_silence_sec,
        },
        "diarize_params": {
            "num_speakers": None,
            "cluster_threshold": args.cluster_threshold,
        },
        "asr": {
            "text": asr_result.text,
            "rtf": asr_result.rtf,
            "chunks": [
                {
                    "text": c.text,
                    "start": c.start,
                    "end": c.end,
                    "index": c.index,
                }
                for c in asr_result.chunks
            ],
        },
        "diarize_turns": [
            {
                "start": float(t["start"]),
                "end": float(t["end"]),
                "speaker": int(t["speaker"]),
            }
            for t in turns
        ],
        "speech_regions": speech_regions,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[dump] wrote {args.out} (total {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
