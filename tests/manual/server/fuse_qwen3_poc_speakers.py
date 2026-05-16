"""
Fuse Qwen3 PoC text/timing from one JSON with speaker labels from another.

This is a manual analysis helper for PR4 long-audio experiments.  It keeps the
text and segment boundaries from the text JSON, then assigns each segment's
speaker by overlap majority against a speaker JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.qwen3.merge import Segment, segments_to_srt
from src.core.qwen3_postprocess import apply_tech_podcast_glossary


def _speaker_to_id(speaker: str) -> int:
    if speaker.startswith("Speaker"):
        return int(speaker.removeprefix("Speaker")) - 1
    return int(speaker)


def choose_speaker(target: dict, speaker_segments: list[dict]) -> str:
    start = float(target["start"])
    end = float(target["end"])
    mid = (start + end) / 2.0
    weights: dict[str, float] = {}
    for src in speaker_segments:
        overlap = min(end, float(src["end"])) - max(start, float(src["start"]))
        if overlap > 0:
            weights[src["speaker"]] = weights.get(src["speaker"], 0.0) + overlap
    if weights:
        return max(weights.items(), key=lambda item: (item[1], item[0]))[0]

    nearest = min(
        speaker_segments,
        key=lambda src: abs(((float(src["start"]) + float(src["end"])) / 2.0) - mid),
    )
    return nearest["speaker"]


def write_srt(payload: dict, path: Path) -> None:
    segments = [
        Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            speaker=_speaker_to_id(str(seg["speaker"])),
            text=str(seg["text"]),
        )
        for seg in payload["segments"]
    ]
    path.write_text(segments_to_srt(segments), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Qwen3 PoC text with speaker labels")
    parser.add_argument("--text-json", required=True, type=Path)
    parser.add_argument("--speaker-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--text-postprocess", choices=["none", "tech-podcast"], default="none")
    args = parser.parse_args()

    text_payload = json.loads(args.text_json.read_text(encoding="utf-8"))
    speaker_payload = json.loads(args.speaker_json.read_text(encoding="utf-8"))
    speaker_segments = speaker_payload["segments"]

    out_payload = json.loads(json.dumps(text_payload, ensure_ascii=False))
    changed = 0
    for seg in out_payload["segments"]:
        if args.text_postprocess == "tech-podcast":
            seg["text"] = apply_tech_podcast_glossary(seg.get("text", ""))
        new_speaker = choose_speaker(seg, speaker_segments)
        if new_speaker != seg["speaker"]:
            changed += 1
        seg["speaker"] = new_speaker

    out_payload.setdefault("summary", {})["fusion_strategy"] = (
        "text_segments_plus_speaker_overlap_majority"
    )
    out_payload["summary"]["source_text_json"] = str(args.text_json)
    out_payload["summary"]["source_speaker_json"] = str(args.speaker_json)
    out_payload["summary"]["speaker_reassigned_segments"] = changed
    out_payload["summary"]["text_postprocess"] = args.text_postprocess

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.text_json.name
    srt_path = json_path.with_suffix(".srt")
    json_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(out_payload, srt_path)
    print(f"[out] JSON {json_path}")
    print(f"[out] SRT  {srt_path}")


if __name__ == "__main__":
    main()
