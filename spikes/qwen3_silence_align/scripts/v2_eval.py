"""跑 silence-aware merge_v2 + 算 metric, 对比 baseline.

Usage:
    venv/bin/python spikes/qwen3_silence_align/scripts/v2_eval.py \
        --intermediate spikes/qwen3_silence_align/data/podcast_60s.intermediate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "spikes" / "qwen3_silence_align"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--intermediate", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("spikes/qwen3_silence_align/output/v2.json"))
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--min-segment-dur", type=float, default=0.1)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("spikes/qwen3_silence_align/output/baseline.json"),
        help="baseline json (跑过 baseline_eval.py 后存在), 用于对比",
    )
    args = p.parse_args()

    data = json.loads(args.intermediate.read_text())
    chunks = [SimpleNamespace(**c) for c in data["asr"]["chunks"]]
    turns = data["diarize_turns"]
    speech_regions = data["speech_regions"]
    audio_duration = data["audio_duration"]

    from align_lib.merge_v2 import merge_v2
    from align_lib.metric import evaluate_alignment, format_metric_report
    from src.core.qwen3.merge import segments_to_srt

    segments, snap_stats = merge_v2(
        chunks, turns, speech_regions, audio_duration,
        tolerance=args.tolerance,
        min_segment_dur=args.min_segment_dur,
    )
    seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in segments]

    metric = evaluate_alignment(seg_dicts, speech_regions, audio_duration, tolerance=0.3)
    print(format_metric_report(metric, f"V2 silence-aware (snap tolerance={args.tolerance}s)"))
    print(f"  snap stats: {snap_stats}")
    print()

    # 对比 baseline (如果存在)
    if args.baseline.exists():
        b = json.loads(args.baseline.read_text())
        b_metric = b["metric"]
        print("=== COMPARISON ===")
        print(f"  baseline align_ratio: {b_metric['align_ratio']*100:6.2f}%  ({b_metric['aligned_boundaries']}/{b_metric['total_boundaries']})")
        print(f"  v2       align_ratio: {metric.align_ratio*100:6.2f}%  ({metric.aligned_boundaries}/{metric.total_boundaries})")
        delta = metric.align_ratio - b_metric["align_ratio"]
        print(f"  delta:                {delta*100:+6.2f}pp")
    else:
        print(f"  (baseline.json not found at {args.baseline}, skip comparison)")

    # dump
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "v2",
        "tolerance": args.tolerance,
        "metric": {
            "total_boundaries": metric.total_boundaries,
            "aligned_boundaries": metric.aligned_boundaries,
            "align_ratio": metric.align_ratio,
            "in_silence_count": metric.in_silence_count,
            "near_boundary_count": metric.near_boundary_count,
        },
        "snap_stats": snap_stats,
        "segments": seg_dicts,
        "boundary_details": [
            {
                "timestamp": d.timestamp,
                "in_silence": d.in_silence,
                "dist_to_nearest_silence_boundary": round(d.dist_to_nearest_silence_boundary, 3),
            }
            for d in metric.details
        ],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    srt_out = args.out.with_suffix(".srt")
    srt_out.write_text(segments_to_srt(segments))
    print(f"\n  wrote {args.out}")
    print(f"  wrote {srt_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
