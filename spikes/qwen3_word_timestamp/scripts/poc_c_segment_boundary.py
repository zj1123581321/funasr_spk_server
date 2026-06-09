#!/usr/bin/env python3
# coding=utf-8
"""PoC-C: 现状段级边界 vs 词级时间戳 — 词级多买了什么.

golden json 的 segment start/end = 现状 pipeline 产出的段边界(标点+字符比例+静音吸附近似)。
拿 fa-zh 逐字时间戳当参照, 看每个段的【首字实际起点】【末字实际终点】跟 golden 段边界差多少。
差得大 → 现状段切点漂, 词级能纠; 差得小 → 现状段边界已准, 词级只多买段内字级粒度。
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--b", required=True, help="PoC-B fa-zh 输出(逐字 timestamp)")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))

    # fa-zh: reference_text 的非空白字符序列 ↔ timestamp[i] = [start_ms,end_ms]
    ref_chars = [c for c in b["reference_text"] if not c.isspace()]
    ts = b["timestamp"]

    idx = 0  # 累计非空白字符指针
    start_diffs = []
    end_diffs = []
    print("段 | golden起-止 | 词级首字起-末字止 | 起差 止差 (ms)")
    for k, seg in enumerate(golden["segments"]):
        seg_chars = [c for c in seg["text"] if not c.isspace()]
        if not seg_chars:
            continue
        first_i = idx
        last_i = idx + len(seg_chars) - 1
        idx += len(seg_chars)
        if last_i >= len(ts):
            break
        w_start = ts[first_i][0] / 1000.0
        w_end = ts[last_i][1] / 1000.0
        sd = abs(w_start - seg["start_time"])
        ed = abs(w_end - seg["end_time"])
        start_diffs.append(sd)
        end_diffs.append(ed)
        if k < 12:
            print(f"{k} | {seg['start_time']:.2f}-{seg['end_time']:.2f} | "
                  f"{w_start:.2f}-{w_end:.2f} | {sd*1000:.0f} {ed*1000:.0f}")

    def stat(name, ds):
        if not ds:
            print(f"{name}: 无数据")
            return
        ds_sorted = sorted(ds)
        aas = sum(ds) / len(ds)
        p90 = ds_sorted[int(len(ds) * 0.9)]
        mx = max(ds)
        print(f"{name}: AAS={aas*1000:.0f}ms  p90={p90*1000:.0f}ms  max={mx*1000:.0f}ms  n={len(ds)}")

    print("---")
    stat("段【起点】偏移 (golden start vs 词级首字)", start_diffs)
    stat("段【终点】偏移 (golden end vs 词级末字)", end_diffs)


if __name__ == "__main__":
    main()
