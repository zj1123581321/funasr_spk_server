"""
Select high-risk 40s windows for Qwen3 speaker refinement without reference.

This manual PR4 helper scores existing Qwen3 long-audio PoC JSON segments using
only hypothesis-side topology, lexical Q/A cues, and debug hints.  It writes a
ranked JSON report and a comma-separated window-id file that can be passed to
``align_qwen3_poc_with_window_diarize.py --window-ids``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


QUESTION_PATTERNS = [
    r"你觉得", r"你们", r"可以讲讲", r"能不能", r"会不会", r"有没有",
    r"是不是", r"是什么", r"为什么", r"商业模式", r"接下来", r"同意.{0,8}吗",
    r"吗[？?]?", r"[？?]",
]
ANSWER_PATTERNS = [
    r"我觉得", r"在我看来", r"我相信", r"我们公司", r"我们自己", r"我是",
    r"我的", r"我刚", r"对我来讲", r"龙虾", r"章鱼", r"明略", r"开源",
]


def score_patterns(patterns: list[str], text: str) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def group_segments(segments: list[dict], window_sec: float) -> dict[int, list[tuple[int, dict]]]:
    groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        start = float(seg["start"])
        wi = int(math.floor(max(0.0, start) / window_sec))
        groups[wi].append((idx, seg))
    return groups


def has_aba_short_turn(items: list[tuple[int, dict]], max_mid_sec: float) -> bool:
    for (_i0, a), (_i1, b), (_i2, c) in zip(items, items[1:], items[2:]):
        if str(a.get("speaker")) != str(c.get("speaker")):
            continue
        if str(a.get("speaker")) == str(b.get("speaker")):
            continue
        if float(b["end"]) - float(b["start"]) <= max_mid_sec:
            return True
    return False


def compute_window_features(
    wi: int,
    items: list[tuple[int, dict]],
    *,
    window_sec: float,
    expected_end: float,
    aba_short_sec: float,
    tight_gap_sec: float,
) -> dict:
    speakers = [str(s.get("speaker")) for _idx, s in items]
    texts = [str(s.get("text", "")) for _idx, s in items]
    q_scores = [score_patterns(QUESTION_PATTERNS, t) for t in texts]
    a_scores = [score_patterns(ANSWER_PATTERNS, t) for t in texts]
    switches = sum(1 for a, b in zip(speakers, speakers[1:]) if a != b)
    very_short = 0
    tight = 0
    mixed_hint = 0
    abnormal_rate = 0
    long_cross = 0
    qa_mixed_segments = 0
    suspicious_single_label_qa = 0
    for pos, (_idx, seg) in enumerate(items):
        start = float(seg["start"]); end = float(seg["end"])
        dur = max(1e-6, end - start)
        chars = len(str(seg.get("text", "")))
        if dur <= 1.0:
            very_short += 1
        # Segment spans a substantial part of this window and crosses a 40s boundary.
        wstart = wi * window_sec; wend = min(wstart + window_sec, expected_end)
        if start < wend - 5.0 and end > wend and dur >= 12.0:
            long_cross += 1
        cps = chars / dur
        if cps < 1.0 or cps > 8.0:
            abnormal_rate += 1
        if q_scores[pos] and a_scores[pos]:
            qa_mixed_segments += 1
        dbg = seg.get("debug") or {}
        hints = dbg.get("speaker_confidence_hints") or []
        if seg.get("mixed_speaker") or hints:
            mixed_hint += 1
        if pos:
            prev = items[pos - 1][1]
            if str(prev.get("speaker")) != str(seg.get("speaker")):
                gap = start - float(prev.get("end", start))
                if abs(gap) <= tight_gap_sec:
                    tight += 1
    if len(set(speakers)) == 1 and sum(q_scores) and sum(a_scores):
        suspicious_single_label_qa = 1
    return {
        "window": wi,
        "start": round(wi * window_sec, 2),
        "end": round(min((wi + 1) * window_sec, expected_end), 2),
        "segments": len(items),
        "chars": sum(len(t) for t in texts),
        "speakers": dict(Counter(speakers)),
        "speaker_switches": switches,
        "question_cues": sum(q_scores),
        "answer_cues": sum(a_scores),
        "qa_mixed_segments": qa_mixed_segments,
        "suspicious_single_label_qa": suspicious_single_label_qa,
        "aba_short_turn": has_aba_short_turn(items, aba_short_sec),
        "very_short_turns": very_short,
        "tight_boundaries": tight,
        "mixed_debug_hints": mixed_hint,
        "long_cross_boundary_segments": long_cross,
        "abnormal_char_rate_segments": abnormal_rate,
        "text_preview": "".join(texts)[:120],
    }


def score_features(f: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    def add(points: float, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(f"{reason}+{points:g}")

    if f["suspicious_single_label_qa"]:
        add(5.0, "single_speaker_with_question_and_answer_cues")
    if f["qa_mixed_segments"]:
        add(3.5 * min(2, f["qa_mixed_segments"]), "same_segment_question_answer_cues")
    if f["mixed_debug_hints"]:
        add(2.0 * min(3, f["mixed_debug_hints"]), "mixed_or_confidence_debug")
    if f["tight_boundaries"]:
        add(1.5 * min(4, f["tight_boundaries"]), "tight_speaker_boundaries")
    if f["very_short_turns"]:
        add(1.0 * min(4, f["very_short_turns"]), "very_short_turns")
    if f["aba_short_turn"]:
        add(3.0, "aba_short_turn")
    if f["long_cross_boundary_segments"]:
        add(2.5 * f["long_cross_boundary_segments"], "long_segment_crosses_window_boundary")
    if f["segments"] <= 1 and f["chars"] >= 120 and (f["question_cues"] or f["answer_cues"]):
        add(3.0, "long_single_segment_with_role_cues")
    if f["segments"] >= 5:
        add(1.5, "many_segments")
    if f["speaker_switches"] >= 3:
        add(1.5, "many_speaker_switches")
    if f["abnormal_char_rate_segments"]:
        add(0.75 * min(3, f["abnormal_char_rate_segments"]), "abnormal_chars_per_second")
    # Small nudge for windows with both Q and A cues across separate segments.
    if f["question_cues"] and f["answer_cues"] and len(f["speakers"]) <= 2:
        add(1.0, "question_answer_cues_in_window")
    return score, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Select no-reference high-risk windows")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=40.0)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=0, help="Keep top K after threshold; 0 keeps all threshold hits")
    parser.add_argument("--aba-short-sec", type=float, default=2.0)
    parser.add_argument("--tight-gap-sec", type=float, default=0.25)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = payload["segments"]
    expected_end = float(payload.get("summary", {}).get("audio_duration") or max(float(s["end"]) for s in segments))
    rows = []
    for wi, items in sorted(group_segments(segments, args.window_sec).items()):
        f = compute_window_features(
            wi, items,
            window_sec=args.window_sec,
            expected_end=expected_end,
            aba_short_sec=args.aba_short_sec,
            tight_gap_sec=args.tight_gap_sec,
        )
        score, reasons = score_features(f)
        f["risk_score"] = round(score, 3)
        f["reasons"] = reasons
        rows.append(f)

    selected = [r for r in rows if r["risk_score"] >= args.threshold]
    selected.sort(key=lambda r: (-r["risk_score"], r["window"]))
    if args.top_k:
        selected = selected[: args.top_k]
    selected_ids = sorted(r["window"] for r in selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_json": str(args.input_json),
        "strategy": "hypothesis_topology_lexical_debug_v1",
        "params": vars(args) | {"input_json": str(args.input_json), "out_dir": str(args.out_dir)},
        "selected_count": len(selected_ids),
        "selected_window_ids": selected_ids,
        "selected_ranked": selected,
        "all_windows_ranked": sorted(rows, key=lambda r: (-r["risk_score"], r["window"])),
    }
    (args.out_dir / "high_risk_windows.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "high_risk_window_ids.txt").write_text(",".join(map(str, selected_ids)), encoding="utf-8")
    print(f"[select] selected={len(selected_ids)} threshold={args.threshold} top_k={args.top_k}")
    print("[select] ids=" + ",".join(map(str, selected_ids)))
    print(f"[out] {args.out_dir / 'high_risk_windows.json'}")
    print(f"[out] {args.out_dir / 'high_risk_window_ids.txt'}")


if __name__ == "__main__":
    main()
