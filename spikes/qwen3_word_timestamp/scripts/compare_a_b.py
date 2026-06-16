#!/usr/bin/env python3
# coding=utf-8
"""头对头:比 PoC-A (Qwen aligner) vs PoC-B (fa-zh) 逐字时间戳一致度.

两个独立对齐器对同一音频, 若逐字时间戳接近 → 互证都准.
口径:对每个非标点汉字, 取两边 start 的绝对差, 报 AAS(start) + 命中率.
"""
import argparse
import json
import unicodedata
from pathlib import Path


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def a_char_starts(a: dict) -> list:
    """PoC-A items: [{text,start,end}] (秒), 含标点; 取 CJK 字的 start(秒)"""
    out = []
    for it in a["items"]:
        t = it["text"]
        for ch in t:
            if is_cjk(ch):
                out.append((ch, it["start"]))
    return out


def b_char_starts(b: dict) -> list:
    """PoC-B: timestamp [[start_ms,end_ms]] 对应 reference_text 的非空白字符序列; 取 CJK"""
    chars = [c for c in b["reference_text"] if not c.isspace()]
    ts = b["timestamp"]
    out = []
    for i, ch in enumerate(chars):
        if i < len(ts) and is_cjk(ch):
            out.append((ch, ts[i][0] / 1000.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))

    sa = a_char_starts(a)
    sb = b_char_starts(b)
    print(f"A(Qwen) CJK字={len(sa)}  B(fa-zh) CJK字={len(sb)}")

    n = min(len(sa), len(sb))
    diffs = []
    mismatch_char = 0
    for i in range(n):
        ca, ta = sa[i]
        cb, tb = sb[i]
        if ca != cb:
            mismatch_char += 1
            continue
        diffs.append(abs(ta - tb))

    if not diffs:
        print("无可比字 (字符序列错位)")
        return
    diffs.sort()
    aas = sum(diffs) / len(diffs)
    p50 = diffs[len(diffs) // 2]
    p90 = diffs[int(len(diffs) * 0.9)]
    hit_100 = sum(1 for d in diffs if d <= 0.1) / len(diffs)
    hit_200 = sum(1 for d in diffs if d <= 0.2) / len(diffs)
    hit_500 = sum(1 for d in diffs if d <= 0.5) / len(diffs)
    print(f"对齐字数={len(diffs)} 字符错位跳过={mismatch_char}")
    print(f"AAS(start差) = {aas*1000:.0f} ms")
    print(f"p50={p50*1000:.0f}ms  p90={p90*1000:.0f}ms")
    print(f"命中率: ≤100ms={hit_100:.1%}  ≤200ms={hit_200:.1%}  ≤500ms={hit_500:.1%}")


if __name__ == "__main__":
    main()
