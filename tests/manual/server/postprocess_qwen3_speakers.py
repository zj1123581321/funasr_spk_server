"""
Post-process Qwen3 long-audio PoC speaker labels without rerunning ASR.

Experimental helper for PR4 149min speaker analysis.  It keeps text/timing
stable, adds mixed-speaker debug flags, and can apply a conservative lexical
role smoother for two-person interview-style audio.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.qwen3.merge import Segment, segments_to_srt

QUESTION_PATTERNS = [
    r"你觉得", r"你们", r"可以讲讲", r"能不能", r"会不会", r"有没有",
    r"是不是", r"是什么", r"为什么", r"商业模式", r"接下来", r"同意.{0,8}吗",
    r"吗[？?]?$", r"[？?]",
]
ANSWER_PATTERNS = [
    r"我觉得", r"在我看来", r"我相信", r"我们公司", r"我们自己", r"我是",
    r"我的", r"我刚", r"对我来讲", r"龙虾", r"章鱼",
]


def _speaker_id(label: str) -> int:
    label = str(label)
    if label.startswith("Speaker"):
        return int(label.removeprefix("Speaker")) - 1
    return int(label)


def _score(patterns: list[str], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def infer_roles(segments: list[dict]) -> tuple[str | None, str | None]:
    """Infer (answer_dominant, question_dominant) speaker labels by text mass."""
    char_counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        char_counts[str(seg.get("speaker", ""))] += len(str(seg.get("text", "")))
    if len(char_counts) != 2:
        return None, None
    ordered = sorted(char_counts.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[0][0], ordered[1][0]


def apply_lexical_role_smoothing(
    payload: dict,
    *,
    max_question_sec: float = 10.0,
    max_answer_sec: float = 12.0,
    question_threshold: int = 1,
    answer_threshold: int = 1,
    neighbor_guard: bool = False,
) -> int:
    """Relabel obvious short Q/A role inversions.

    Assumption: for a two-person interview, the dominant-text speaker is the
    answerer/guest and the minority speaker is the questioner/host.  This is
    intentionally conservative and records every change in ``debug``.
    """
    segments = payload.get("segments", [])
    answer_speaker, question_speaker = infer_roles(segments)
    if not answer_speaker or not question_speaker:
        return 0

    changes = []
    for idx, seg in enumerate(segments):
        text = str(seg.get("text", ""))
        old = str(seg.get("speaker", ""))
        duration = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        q_score = _score(QUESTION_PATTERNS, text)
        a_score = _score(ANSWER_PATTERNS, text)
        new = old
        reason = ""
        prev_speaker = str(segments[idx - 1].get("speaker", "")) if idx else ""
        next_speaker = str(segments[idx + 1].get("speaker", "")) if idx + 1 < len(segments) else ""

        if old == answer_speaker and duration <= max_question_sec and q_score >= question_threshold and q_score >= a_score:
            new = question_speaker
            reason = "short_answer_speaker_segment_has_question_cues"
        elif old == question_speaker and duration <= max_answer_sec and a_score >= answer_threshold and a_score > q_score:
            if (not neighbor_guard) or prev_speaker == answer_speaker or next_speaker == answer_speaker:
                new = answer_speaker
                reason = "short_question_speaker_segment_has_answer_cues"

        if new != old:
            seg["speaker"] = new
            debug = seg.setdefault("debug", {})
            debug["speaker_smoothed_from"] = old
            debug["speaker_smoothing_reason"] = reason
            debug["speaker_role_scores"] = {"question": q_score, "answer": a_score}
            changes.append({
                "idx": idx,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "from": old,
                "to": new,
                "reason": reason,
                "question_score": q_score,
                "answer_score": a_score,
                "text_preview": text[:80],
            })

    payload.setdefault("debug", {})["speaker_smoothing_changes"] = changes
    payload.setdefault("summary", {})["speaker_smoothing"] = {
        "strategy": "lexical_role_short_turn_v1",
        "assumptions": [
            "two-speaker interview/conversation",
            "dominant text speaker is answerer; minority text speaker is questioner",
        ],
        "answer_speaker": answer_speaker,
        "question_speaker": question_speaker,
        "changes": len(changes),
        "params": {
            "max_question_sec": max_question_sec,
            "max_answer_sec": max_answer_sec,
            "question_threshold": question_threshold,
            "answer_threshold": answer_threshold,
            "neighbor_guard": neighbor_guard,
        },
    }
    return len(changes)


def add_mixed_speaker_debug(payload: dict, *, boundary_gap_sec: float = 0.25, short_sec: float = 1.0) -> int:
    """Add non-breaking mixed/low-confidence debug hints based on local topology."""
    segments = payload.get("segments", [])
    marked = 0
    for idx, seg in enumerate(segments):
        start = float(seg.get("start", 0.0)); end = float(seg.get("end", 0.0))
        duration = end - start
        prev = segments[idx - 1] if idx else None
        nxt = segments[idx + 1] if idx + 1 < len(segments) else None
        hints: list[str] = []
        if prev and str(prev.get("speaker")) != str(seg.get("speaker")):
            gap = start - float(prev.get("end", start))
            if abs(gap) <= boundary_gap_sec:
                hints.append("tight_left_speaker_boundary")
        if nxt and str(nxt.get("speaker")) != str(seg.get("speaker")):
            gap = float(nxt.get("start", end)) - end
            if abs(gap) <= boundary_gap_sec:
                hints.append("tight_right_speaker_boundary")
        if duration <= short_sec:
            hints.append("very_short_turn_low_confidence")
        if hints:
            seg.setdefault("debug", {})["speaker_confidence_hints"] = hints
            if any(h.startswith("tight_") for h in hints):
                seg["mixed_speaker"] = True
                marked += 1
    payload.setdefault("summary", {})["mixed_speaker_debug"] = {
        "strategy": "topology_tight_boundary_flags_v1",
        "boundary_gap_sec": boundary_gap_sec,
        "short_sec": short_sec,
        "marked_segments": marked,
    }
    return marked


def write_srt(payload: dict, path: Path) -> None:
    segments = [
        Segment(float(s["start"]), float(s["end"]), _speaker_id(str(s["speaker"])), str(s.get("text", "")))
        for s in payload.get("segments", [])
    ]
    path.write_text(segments_to_srt(segments), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process Qwen3 PoC speaker labels")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--strategy", choices=["debug-only", "lexical-role"], default="lexical-role")
    parser.add_argument("--max-question-sec", type=float, default=10.0)
    parser.add_argument("--max-answer-sec", type=float, default=12.0)
    parser.add_argument("--question-threshold", type=int, default=1)
    parser.add_argument("--answer-threshold", type=int, default=1)
    parser.add_argument("--neighbor-guard", action="store_true")
    parser.add_argument("--mixed-boundary-gap-sec", type=float, default=0.25)
    parser.add_argument("--mixed-short-sec", type=float, default=1.0)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    marked = add_mixed_speaker_debug(
        payload,
        boundary_gap_sec=args.mixed_boundary_gap_sec,
        short_sec=args.mixed_short_sec,
    )
    changed = 0
    if args.strategy == "lexical-role":
        changed = apply_lexical_role_smoothing(
            payload,
            max_question_sec=args.max_question_sec,
            max_answer_sec=args.max_answer_sec,
            question_threshold=args.question_threshold,
            answer_threshold=args.answer_threshold,
            neighbor_guard=args.neighbor_guard,
        )

    payload.setdefault("summary", {})["speaker_postprocess_source_json"] = str(args.input_json)
    payload["summary"]["speaker_postprocess_strategy"] = args.strategy

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.input_json.name
    srt_path = json_path.with_suffix(".srt")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(payload, srt_path)
    print(f"[postprocess] strategy={args.strategy} changed={changed} mixed_marked={marked}")
    print(f"[out] JSON {json_path}")
    print(f"[out] SRT  {srt_path}")


if __name__ == "__main__":
    main()
