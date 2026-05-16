"""
Use Qwen3-ForcedAligner timestamps to re-split existing Qwen3 PoC text by time.

This is an experimental offline helper: it does NOT rerun ASR or diarization.
It takes existing segment text, aligns concatenated text inside fixed time windows
(default: 40s ASR chunk windows), then assigns each aligned char/word item to the
speaker interval active at that timestamp.  The output JSON keeps the same text
content but can move speaker boundaries to punctuation/char timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.qwen3.merge import Segment, segments_to_srt


def _speaker_id(label: str) -> int:
    label = str(label)
    if label.startswith("Speaker"):
        return int(label.removeprefix("Speaker")) - 1
    return int(label)


def load_audio_window(path: Path, start: float, end: float, sr: int = 16000) -> tuple[np.ndarray, int]:
    import librosa

    audio, _sr = librosa.load(str(path), sr=sr, mono=True, offset=start, duration=max(0.01, end - start))
    return audio.astype(np.float32), sr


def window_index(t: float, size: float) -> int:
    return int(math.floor(max(0.0, t) / size))


def group_segments(segments: list[dict], window_sec: float, start_sec: float | None, end_sec: float | None) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for idx, seg in enumerate(segments):
        s = float(seg["start"]); e = float(seg["end"])
        if start_sec is not None and e <= start_sec:
            continue
        if end_sec is not None and s >= end_sec:
            continue
        wi = window_index(s, window_sec)
        groups.setdefault(wi, []).append({**seg, "_idx": idx})
    return groups


def choose_speaker(item_start: float, item_end: float, intervals: list[dict]) -> tuple[str, float, int | None]:
    mid = (item_start + item_end) / 2.0
    best = None
    best_score = -1.0
    dur = max(0.001, item_end - item_start)
    for it in intervals:
        s = float(it["start"]); e = float(it["end"])
        ov = min(item_end, e) - max(item_start, s)
        if ov > best_score:
            best_score = ov
            best = it
    if best is not None and best_score > 0:
        return str(best["speaker"]), min(1.0, best_score / dur), int(best.get("_idx", -1))

    # zero-duration punctuation or small timestamp gaps: choose containing/nearest interval by midpoint
    containing = [it for it in intervals if float(it["start"]) <= mid <= float(it["end"])]
    if containing:
        it = min(containing, key=lambda x: float(x["end"]) - float(x["start"]))
        return str(it["speaker"]), 0.5, int(it.get("_idx", -1))
    it = min(intervals, key=lambda x: min(abs(float(x["start"]) - mid), abs(float(x["end"]) - mid)))
    return str(it["speaker"]), 0.0, int(it.get("_idx", -1))


def flush_segment(out: list[dict], speaker: str | None, text_parts: list[str], start: float | None, end: float | None, debug: dict[str, Any] | None = None) -> None:
    text = "".join(text_parts)
    if not speaker or start is None or end is None or not text:
        return
    out.append({
        "start": round(float(start), 2),
        "end": round(float(max(end, start)), 2),
        "speaker": speaker,
        "text": text,
        **({"debug": debug} if debug else {}),
    })


def align_window(model, audio_path: Path, window_start: float, window_end: float, intervals: list[dict], language: str) -> tuple[list[dict], dict]:
    text = "".join(str(s.get("text", "")) for s in intervals)
    if not text.strip():
        return [], {"skipped": "empty_text"}
    audio, sr = load_audio_window(audio_path, window_start, window_end)
    t0 = time.time()
    result = model.align((audio, sr), text, language)[0]
    elapsed = time.time() - t0

    out: list[dict] = []
    cur_speaker: str | None = None
    cur_parts: list[str] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_debug = {"source_window_start": round(window_start, 2), "source_window_end": round(window_end, 2), "method": "qwen3_forced_aligner_interval_join"}
    low_conf = 0

    for item in result.items:
        txt = str(item.text)
        istart = window_start + float(item.start_time)
        iend = window_start + float(item.end_time)
        # clamp small model overrun
        istart = min(max(istart, window_start), window_end)
        iend = min(max(iend, istart), window_end)
        speaker, conf, src_idx = choose_speaker(istart, iend, intervals)
        if conf < 0.5:
            low_conf += 1
        if cur_speaker is None:
            cur_speaker = speaker; cur_start = istart; cur_end = iend; cur_parts = [txt]
        elif speaker == cur_speaker:
            cur_parts.append(txt); cur_end = max(cur_end if cur_end is not None else iend, iend)
        else:
            flush_segment(out, cur_speaker, cur_parts, cur_start, cur_end, dict(cur_debug))
            cur_speaker = speaker; cur_start = istart; cur_end = iend; cur_parts = [txt]
    flush_segment(out, cur_speaker, cur_parts, cur_start, cur_end, dict(cur_debug))
    return out, {"aligned_items": len(result.items), "elapsed": elapsed, "low_conf_items": low_conf, "chars": len(text)}


def write_srt(payload: dict, path: Path) -> None:
    segments = [Segment(float(s["start"]), float(s["end"]), _speaker_id(str(s["speaker"])), str(s.get("text", ""))) for s in payload["segments"]]
    path.write_text(segments_to_srt(segments), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 forced-align existing PoC segments")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--aligner-model", type=Path, default=PROJECT_ROOT / "models/qwen3_diarize/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=40.0)
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device-map", default="mps")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    args = parser.parse_args()

    import torch
    from qwen_asr import Qwen3ForcedAligner

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_segments = payload["segments"]
    groups = group_segments(source_segments, args.window_sec, args.start_sec, args.end_sec)
    items = sorted(groups.items())
    if args.max_windows:
        items = items[: args.max_windows]

    print(f"[align] load model={args.aligner_model} device={args.device_map} dtype={args.dtype} windows={len(items)}")
    t_model = time.time()
    model = Qwen3ForcedAligner.from_pretrained(str(args.aligner_model), dtype=dtype, device_map=args.device_map)
    print(f"[align] model loaded elapsed={time.time() - t_model:.1f}s device={model.device} dtype={model.model.dtype}")

    new_segments: list[dict] = []
    stats: list[dict] = []
    aligned_window_ids = {wi for wi, _ in items}
    for wi, intervals in items:
        wstart = wi * args.window_sec
        wend = wstart + args.window_sec
        print(f"[align] window={wi} {wstart:.1f}-{wend:.1f}s intervals={len(intervals)} chars={sum(len(s.get('text','')) for s in intervals)}")
        aligned, st = align_window(model, args.audio, wstart, wend, intervals, args.language)
        stats.append({"window": wi, "start": wstart, "end": wend, "input_segments": len(intervals), "output_segments": len(aligned), **st})
        new_segments.extend(aligned)

    # If aligning only a subset, preserve unaligned segments around it.
    if len(aligned_window_ids) != len(groups) or args.start_sec is not None or args.end_sec is not None or args.max_windows:
        preserved = []
        for seg in source_segments:
            if window_index(float(seg["start"]), args.window_sec) not in aligned_window_ids:
                preserved.append(seg)
        new_segments = preserved + new_segments
    new_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))

    out_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    out_payload["segments"] = new_segments
    out_payload.setdefault("summary", {})["forced_alignment_postprocess"] = {
        "strategy": "qwen3_forced_aligner_interval_join_v1",
        "source_json": str(args.input_json),
        "audio": str(args.audio),
        "aligner_model": str(args.aligner_model),
        "window_sec": args.window_sec,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "aligned_windows": len(items),
        "source_segment_count": len(source_segments),
        "output_segment_count": len(new_segments),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "window_stats": stats,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.input_json.name
    srt_path = json_path.with_suffix(".srt")
    json_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(out_payload, srt_path)
    print(f"[out] JSON {json_path}")
    print(f"[out] SRT  {srt_path}")


if __name__ == "__main__":
    main()
