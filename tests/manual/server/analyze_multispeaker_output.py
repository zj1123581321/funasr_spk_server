"""无 reference 的多人 PoC 输出分析.

针对不知道 speaker 数的实际场景, 不能算 speaker_acc, 但可以看:
- 检测到的 speaker 数 + 每个 speaker 总时长 / 占比
- 段长分布
- speaker 切换频率
- 每个 speaker 的代表性文本片段(供人工判断)
- 短段 (<1s) 比例
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--show-samples", type=int, default=3, help="每个 speaker 展示几条样本")
    args = ap.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    summary = payload.get("summary") or {}
    audio_duration = float(summary.get("audio_duration", 0)) or sum(
        max(0.0, float(s["end"]) - float(s["start"])) for s in segments
    )

    print(f"=== {args.json_path.name} ===")
    print(f"audio_duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)")
    print(f"total segments: {len(segments)}")
    print()

    # Speaker stats
    by_spk = defaultdict(list)
    for s in segments:
        by_spk[str(s.get("speaker"))].append(s)

    spk_dur = {sp: sum(float(s["end"]) - float(s["start"]) for s in segs) for sp, segs in by_spk.items()}
    total_dur = sum(spk_dur.values()) or 1.0
    print(f"detected speakers: {len(by_spk)}")
    print(f"{'speaker':12} | {'segs':>5} | {'dur (s)':>9} | {'dur (min)':>9} | {'share':>7}")
    print("-" * 60)
    for sp, dur in sorted(spk_dur.items(), key=lambda kv: -kv[1]):
        share = dur / total_dur * 100
        print(f"{sp:12} | {len(by_spk[sp]):>5} | {dur:>9.1f} | {dur/60:>9.2f} | {share:>6.1f}%")
    print()

    # Segment length distribution
    durs = sorted(float(s["end"]) - float(s["start"]) for s in segments)
    if durs:
        buckets = {"<1s": 0, "1-3s": 0, "3-10s": 0, "10-30s": 0, ">=30s": 0}
        for d in durs:
            if d < 1: buckets["<1s"] += 1
            elif d < 3: buckets["1-3s"] += 1
            elif d < 10: buckets["3-10s"] += 1
            elif d < 30: buckets["10-30s"] += 1
            else: buckets[">=30s"] += 1
        n = len(durs)
        print("segment length distribution:")
        for k, v in buckets.items():
            bar = "#" * int(v / n * 50)
            print(f"  {k:8}: {v:>4} ({v/n*100:5.1f}%) {bar}")
        print(f"  median: {durs[n//2]:.2f}s   mean: {sum(durs)/n:.2f}s   max: {durs[-1]:.2f}s")
    print()

    # Speaker switch frequency
    switches = sum(1 for a, b in zip(segments, segments[1:]) if str(a.get("speaker")) != str(b.get("speaker")))
    print(f"speaker switches: {switches} ({switches/audio_duration*60:.1f} per minute)")
    print()

    # Sample text per speaker
    print(f"=== Sample segments per speaker (top {args.show_samples} by duration) ===")
    for sp in sorted(by_spk.keys()):
        segs = sorted(by_spk[sp], key=lambda s: float(s["end"]) - float(s["start"]), reverse=True)[: args.show_samples]
        print(f"\n[{sp}] (total {len(by_spk[sp])} segs, {spk_dur[sp]:.0f}s)")
        for s in segs:
            dur = float(s["end"]) - float(s["start"])
            text = (s.get("text") or "").strip()[:120]
            print(f"  {s['start']:>7.1f}-{s['end']:>7.1f} ({dur:>5.1f}s) {text}")

    # First/last segments (sanity check)
    print(f"\n=== First 5 segments ===")
    for s in segments[:5]:
        dur = float(s["end"]) - float(s["start"])
        text = (s.get("text") or "").strip()[:100]
        print(f"  {s['start']:>7.1f}-{s['end']:>7.1f} ({dur:>5.1f}s) [{s.get('speaker')}] {text}")
    print(f"\n=== Last 5 segments ===")
    for s in segments[-5:]:
        dur = float(s["end"]) - float(s["start"])
        text = (s.get("text") or "").strip()[:100]
        print(f"  {s['start']:>7.1f}-{s['end']:>7.1f} ({dur:>5.1f}s) [{s.get('speaker')}] {text}")


if __name__ == "__main__":
    main()
