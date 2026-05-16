"""
Qwen3 long-audio PoC runner.

This script is intentionally outside the server path.  It validates the PR4
long-audio design with:

- quality-first macro segment planning (default target 12min, soft max 15min)
- optional ffmpeg silence boundary detection for VAD-like natural cuts
- global diarization once, preserving speaker IDs across ASR windows
- per-window Qwen3 ASR with one reused engine
- final JSON/SRT output plus RTF/quality-warning metrics

Examples:
    # Fast planning only
    python tests/manual/server/qwen3_long_audio_poc.py tmp_long_audio/audio_149min.mp3 --plan-only

    # Smoke first segment with real ASR, no diarization
    python tests/manual/server/qwen3_long_audio_poc.py tmp_long_audio/audio_83min.m4a --max-segments 1 --skip-diarize

    # Full PoC
    python tests/manual/server/qwen3_long_audio_poc.py tmp_long_audio/audio_149min.mp3
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import config
from src.core.qwen3.asr import build_engine, run_asr_window
from src.core.qwen3.asr import ASRChunkItem
from src.core.qwen3.diarize import run_diarization
from src.core.qwen3.merge import (
    Segment,
    filter_spurious_speakers,
    merge_asr_chunks_and_diarize,
    merge_asr_and_diarize,
    segments_to_srt,
)
from src.core.qwen3_segment import (
    clip_turns_to_window,
    detect_repetition_warnings,
    detect_segment_similarity_warnings,
    plan_macro_segments,
)
from src.core.qwen3_postprocess import apply_tech_podcast_glossary
from src.utils.file_utils import convert_to_wav, get_audio_duration


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def ffmpeg_speech_regions(
    audio_path: Path,
    audio_duration: float,
    noise_db: str = "-35dB",
    min_silence_sec: float = 0.8,
) -> list[dict]:
    """Use ffmpeg silencedetect as a lightweight VAD boundary proxy."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_db}:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(stderr[-2000:])

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


def offset_segments(segments: Iterable[Segment], offset: float) -> list[Segment]:
    return [
        Segment(
            start=round(seg.start + offset, 2),
            end=round(seg.end + offset, 2),
            speaker=seg.speaker,
            text=seg.text,
        )
        for seg in segments
    ]


def trim_segments_to_window(
    segments: Iterable[Segment],
    window_start: float,
    window_end: float,
) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        start = max(seg.start, window_start)
        end = min(seg.end, window_end)
        if end - start <= 0.05:
            continue
        out.append(
            Segment(start=round(start, 2), end=round(end, 2), speaker=seg.speaker, text=seg.text)
        )
    return out


def _turn_to_payload(turn: dict) -> dict:
    speaker = int(turn["speaker"])
    return {
        "start": round(float(turn["start"]), 2),
        "end": round(float(turn["end"]), 2),
        "speaker": f"Speaker{speaker + 1}",
    }


def write_outputs(
    out_dir: Path,
    audio_path: Path,
    segments: list[Segment],
    summary: dict,
    diarization_turns: list[dict] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    json_path = out_dir / f"{stem}.qwen3_long_poc.json"
    srt_path = out_dir / f"{stem}.qwen3_long_poc.srt"

    payload = {
        "summary": summary,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": f"Speaker{seg.speaker + 1}",
                "text": seg.text,
            }
            for seg in segments
        ],
    }
    if diarization_turns:
        payload["diarization_turns"] = [_turn_to_payload(turn) for turn in diarization_turns]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
    print(f"[out] JSON {json_path}")
    print(f"[out] SRT  {srt_path}")


def load_turns_from_poc_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_turns = payload.get("diarization_turns") or payload.get("segments", [])
    turns = []
    for seg in source_turns:
        speaker = str(seg["speaker"])
        if speaker.startswith("Speaker"):
            speaker_id = int(speaker.removeprefix("Speaker")) - 1
        else:
            speaker_id = int(speaker)
        turns.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": speaker_id,
            }
        )
    return sorted(turns, key=lambda t: t["start"])


def postprocess_text(text: str, mode: str) -> str:
    if mode == "tech-podcast":
        return apply_tech_podcast_glossary(text)
    return text


def postprocess_chunks(chunks: list[ASRChunkItem], mode: str) -> list[ASRChunkItem]:
    if mode == "none":
        return chunks
    return [
        ASRChunkItem(
            text=postprocess_text(chunk.text, mode),
            start=chunk.start,
            end=chunk.end,
            index=chunk.index,
        )
        for chunk in chunks
    ]


def prepare_diarization_audio(audio_path: Path, out_dir: Path) -> Path:
    """Return an audio path libsndfile/sherpa can read."""
    if audio_path.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}:
        return audio_path

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{audio_path.stem}.diarize.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        print(f"[diarize] reuse converted wav: {wav_path}")
        return wav_path

    print(f"[diarize] convert for sherpa: {audio_path.name} -> {wav_path.name}")
    convert_to_wav(str(audio_path), output_path=str(wav_path))
    return wav_path


def sample_processes(stop: threading.Event, samples: list[dict]) -> None:
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    while not stop.is_set():
        rss_mb = proc.memory_info().rss / 1024 / 1024
        cpu = proc.cpu_percent(interval=1.0)
        samples.append({"t": time.time(), "rss_mb": rss_mb, "cpu_pct": cpu})


async def run_poc(args: argparse.Namespace) -> dict:
    audio_path = args.audio.resolve()
    audio_duration = get_audio_duration(str(audio_path))
    if audio_duration <= 0:
        raise RuntimeError(f"Unable to read audio duration: {audio_path}")

    t0 = time.time()
    speech_regions: list[dict] = []
    if args.boundary_source == "ffmpeg-silence":
        t_vad = time.time()
        speech_regions = ffmpeg_speech_regions(
            audio_path,
            audio_duration=audio_duration,
            noise_db=args.silence_noise,
            min_silence_sec=args.min_silence_sec,
        )
        print(f"[vad] regions={len(speech_regions)} elapsed={time.time() - t_vad:.1f}s")

    macro_segments = plan_macro_segments(
        audio_duration=audio_duration,
        speech_regions=speech_regions,
        target_segment_sec=args.target_min * 60,
        soft_max_segment_sec=args.soft_max_min * 60,
        hard_max_segment_sec=args.hard_max_min * 60,
        min_segment_sec=args.min_segment_min * 60,
        overlap_sec=args.overlap_sec,
        boundary_search_sec=args.boundary_search_sec,
        min_silence_sec=args.min_silence_sec,
    )
    if args.max_segments:
        macro_segments = macro_segments[: args.max_segments]

    print(f"[plan] audio={audio_duration:.1f}s ({audio_duration/60:.1f}min)")
    print(
        f"[plan] segments={len(macro_segments)} target={args.target_min:.1f}min "
        f"soft_max={args.soft_max_min:.1f}min hard_max={args.hard_max_min:.1f}min "
        f"overlap={args.overlap_sec:.1f}s"
    )
    for seg in macro_segments:
        print(
            f"  #{seg.idx:02d} output={seg.start:.1f}-{seg.end:.1f}s "
            f"asr={seg.asr_start:.1f}-{seg.asr_end:.1f}s "
            f"dur={seg.duration/60:.2f}min"
        )

    if args.plan_only:
        return {
            "audio": str(audio_path),
            "audio_duration": audio_duration,
            "macro_segments": [seg.to_dict() for seg in macro_segments],
            "speech_region_count": len(speech_regions),
            "elapsed": time.time() - t0,
        }

    stop = threading.Event()
    samples: list[dict] = []
    sampler = threading.Thread(target=sample_processes, args=(stop, samples), daemon=True)
    sampler.start()

    turns: list[dict] = []
    filtered_turns: list[dict] = []
    t_diarize = 0.0
    if args.reuse_turns_json:
        filtered_turns = load_turns_from_poc_json(args.reuse_turns_json)
        turns = filtered_turns
        print(
            f"[diarize] reuse turns={len(filtered_turns)} "
            f"speakers={sorted({t['speaker'] for t in filtered_turns})} "
            f"from={args.reuse_turns_json}"
        )
    elif not args.skip_diarize:
        t_dia0 = time.time()
        q = config.qwen3
        diarize_audio = prepare_diarization_audio(audio_path, args.out_dir)
        turns = run_diarization(
            str(diarize_audio),
            segmentation_model=q.segmentation_model,
            embedding_model=q.embedding_model,
            num_speakers=q.num_speakers,
            cluster_threshold=q.cluster_threshold,
            num_threads=args.diarize_threads or q.num_threads,
            provider=q.provider,
        )
        filtered_turns = filter_spurious_speakers(
            turns,
            audio_duration=audio_duration,
        )
        t_diarize = time.time() - t_dia0
        print(
            f"[diarize] turns={len(turns)} filtered={len(filtered_turns)} "
            f"speakers={sorted({t['speaker'] for t in filtered_turns})} "
            f"elapsed={t_diarize:.1f}s rtf={t_diarize/audio_duration:.3f}"
        )

    q = config.qwen3
    print(f"[asr] loading engine: {q.asr_model_dir}")
    engine = build_engine(q.asr_model_dir)

    all_segments: list[Segment] = []
    window_results: list[dict] = []
    asr_text_parts: list[str] = []
    t_asr_total = 0.0

    for seg in macro_segments:
        win_duration = seg.asr_duration
        if args.show_asr_stream:
            asr_result = run_asr_window(
                str(audio_path),
                engine=engine,
                start_second=seg.asr_start,
                duration=win_duration,
                language=q.language,
                temperature=q.temperature,
            )
        else:
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                asr_result = run_asr_window(
                    str(audio_path),
                    engine=engine,
                    start_second=seg.asr_start,
                    duration=win_duration,
                    language=q.language,
                    temperature=q.temperature,
                )
        t_asr_total += asr_result.elapsed
        asr_text = postprocess_text(asr_result.text, args.text_postprocess)
        asr_chunks = postprocess_chunks(asr_result.chunks, args.text_postprocess)
        asr_text_parts.append(asr_text)

        if filtered_turns:
            # Use output window for final timeline; ASR overlap is only context.
            relative_turns = clip_turns_to_window(
                filtered_turns,
                window_start=seg.start,
                window_end=seg.end,
                relative=True,
            )
            if asr_chunks:
                merged = merge_asr_chunks_and_diarize(asr_chunks, relative_turns)
            else:
                merged = merge_asr_and_diarize(asr_text, relative_turns)
            merged = offset_segments(merged, seg.start)
            merged = trim_segments_to_window(merged, seg.start, seg.end)
        else:
            merged = [
                Segment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    speaker=0,
                    text=asr_text,
                )
            ]

        all_segments.extend(merged)
        window_warning = detect_repetition_warnings(asr_text)
        window_results.append(
            {
                "idx": seg.idx,
                "segment": seg.to_dict(),
                "asr_duration": asr_result.duration,
                "asr_elapsed": asr_result.elapsed,
                "asr_rtf": asr_result.rtf,
                "text_chars": len(asr_text),
                "asr_chunks": len(asr_chunks),
                "warnings": window_warning,
            }
        )
        print(
            f"[asr] #{seg.idx:02d} elapsed={asr_result.elapsed:.1f}s "
            f"rtf={asr_result.rtf:.3f} chars={len(asr_result.text)} "
            f"merged_segments={len(merged)} warnings={len(window_warning)}"
        )

    stop.set()
    sampler.join(timeout=3.0)

    segment_warnings = detect_segment_similarity_warnings([s.text for s in all_segments])
    text_warnings = detect_repetition_warnings("".join(asr_text_parts))
    quality_warnings = text_warnings + segment_warnings
    quality_warnings.extend(
        {"type": "window", "window": r["idx"], "detail": w}
        for r in window_results
        for w in r["warnings"]
    )

    total_elapsed = time.time() - t0
    processed_duration = sum(seg.duration for seg in macro_segments)
    summary = {
        "audio": str(audio_path),
        "audio_duration": audio_duration,
        "processed_duration": processed_duration,
        "total_elapsed": total_elapsed,
        "total_rtf_full_audio": total_elapsed / audio_duration,
        "total_rtf_processed_audio": total_elapsed / processed_duration if processed_duration else 0.0,
        "asr_elapsed_total": t_asr_total,
        "asr_rtf_full_audio": t_asr_total / audio_duration,
        "asr_rtf_processed_audio": t_asr_total / processed_duration if processed_duration else 0.0,
        "diarize_elapsed": t_diarize,
        "diarize_rtf": t_diarize / audio_duration if t_diarize else 0.0,
        "macro_segments": [seg.to_dict() for seg in macro_segments],
        "window_results": window_results,
        "segment_count": len(all_segments),
        "diarization_turn_count": len(filtered_turns),
        "speakers": sorted({f"Speaker{s.speaker + 1}" for s in all_segments}),
        "quality_warnings": quality_warnings,
        "text_postprocess": args.text_postprocess,
        "reuse_turns_json": str(args.reuse_turns_json) if args.reuse_turns_json else "",
        "rss_peak_mb": max((s["rss_mb"] for s in samples), default=0.0),
        "cpu_pct_peak": max((s["cpu_pct"] for s in samples), default=0.0),
    }
    print(
        f"[summary] elapsed={total_elapsed:.1f}s "
        f"rtf_processed={summary['total_rtf_processed_audio']:.3f} "
        f"asr_rtf_processed={summary['asr_rtf_processed_audio']:.3f} "
        f"diarize_rtf={summary['diarize_rtf']:.3f} "
        f"segments={len(all_segments)} speakers={summary['speakers']} "
        f"quality_warnings={len(quality_warnings)} rss_peak={summary['rss_peak_mb']:.0f}MB"
    )

    if not args.no_write_outputs:
        write_outputs(args.out_dir, audio_path, all_segments, summary, filtered_turns)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 long-audio PoC runner")
    parser.add_argument("audio", type=Path, help="Audio file path")
    parser.add_argument("--plan-only", action="store_true", help="Only print macro segment plan")
    parser.add_argument("--skip-diarize", action="store_true", help="Skip diarization and output one speaker")
    parser.add_argument("--max-segments", type=int, default=0, help="Run only first N macro segments")
    parser.add_argument("--target-min", type=float, default=12.0)
    parser.add_argument("--soft-max-min", type=float, default=15.0)
    parser.add_argument("--hard-max-min", type=float, default=20.0)
    parser.add_argument("--min-segment-min", type=float, default=3.0)
    parser.add_argument("--overlap-sec", type=float, default=0.0)
    parser.add_argument("--boundary-search-sec", type=float, default=60.0)
    parser.add_argument("--min-silence-sec", type=float, default=0.8)
    parser.add_argument("--boundary-source", choices=["none", "ffmpeg-silence"], default="ffmpeg-silence")
    parser.add_argument("--silence-noise", default="-35dB")
    parser.add_argument("--diarize-threads", type=int, default=0)
    parser.add_argument("--reuse-turns-json", type=Path, default=None)
    parser.add_argument("--text-postprocess", choices=["none", "tech-podcast"], default="none")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "tmp_long_audio" / "poc_outputs")
    parser.add_argument("--no-write-outputs", action="store_true")
    parser.add_argument("--show-asr-stream", action="store_true", help="Do not suppress vendor streaming text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_poc(args))


if __name__ == "__main__":
    main()
