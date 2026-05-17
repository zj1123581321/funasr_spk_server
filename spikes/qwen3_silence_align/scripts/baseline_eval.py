"""跑现有 merge 逻辑 + 算 metric, 作为 baseline.

Usage:
    venv/bin/python spikes/qwen3_silence_align/scripts/baseline_eval.py \
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


def load_intermediate(path: Path) -> dict:
    return json.loads(path.read_text())


def chunks_to_objects(chunks_dicts: list[dict]):
    """JSON dict → 鸭子类型对象, merge.py 用 getattr 读 text/start/end."""
    return [SimpleNamespace(**c) for c in chunks_dicts]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--intermediate", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("spikes/qwen3_silence_align/output/baseline.json"))
    args = p.parse_args()

    data = load_intermediate(args.intermediate)
    chunks = chunks_to_objects(data["asr"]["chunks"])
    turns = data["diarize_turns"]
    speech_regions = data["speech_regions"]
    audio_duration = data["audio_duration"]

    # 跑现有 production merge
    from src.core.qwen3.merge import merge_asr_chunks_and_diarize, segments_to_srt
    segments = merge_asr_chunks_and_diarize(chunks, turns)
    seg_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in segments]

    # 算 metric
    from align_lib.metric import evaluate_alignment, format_metric_report
    metric = evaluate_alignment(seg_dicts, speech_regions, audio_duration, tolerance=0.3)
    print(format_metric_report(metric, "BASELINE (production merge_asr_chunks_and_diarize)"))

    # dump 段 + SRT 供人工对照
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "baseline",
        "metric": {
            "total_boundaries": metric.total_boundaries,
            "aligned_boundaries": metric.aligned_boundaries,
            "align_ratio": metric.align_ratio,
            "in_silence_count": metric.in_silence_count,
            "near_boundary_count": metric.near_boundary_count,
        },
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
    print(f"  wrote {args.out}")
    print(f"  wrote {srt_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
