#!/usr/bin/env python3
# coding=utf-8
"""通用逐字时间戳一致度对比 (任意两个对齐器输出).

支持两种输出格式:
- items: {"items":[{"text","start","end"}]}  (PoC-A Qwen / PoC-MMS, text 可能含标点/多字)
- fazh : {"reference_text", "timestamp":[[start_ms,end_ms]]}  (PoC-B fa-zh)

口径: 每个 CJK 字取 start(秒), 按字序对齐, 报 AAS + 命中率.
"""
import argparse
import json
from pathlib import Path


def is_cjk(ch):
    return "一" <= ch <= "鿿"


def extract(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    if "timestamp" in d:  # fa-zh
        chars = [c for c in d["reference_text"] if not c.isspace()]
        ts = d["timestamp"]
        for i, ch in enumerate(chars):
            if i < len(ts) and is_cjk(ch):
                out.append((ch, ts[i][0] / 1000.0))
    else:  # items
        for it in d["items"]:
            for ch in it["text"]:
                if is_cjk(ch):
                    out.append((ch, it["start"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--xname", default="X")
    ap.add_argument("--yname", default="Y")
    args = ap.parse_args()

    sx, sy = extract(args.x), extract(args.y)
    print(f"{args.xname} CJK字={len(sx)}  {args.yname} CJK字={len(sy)}")
    n = min(len(sx), len(sy))
    diffs, mism = [], 0
    for i in range(n):
        if sx[i][0] != sy[i][0]:
            mism += 1
            continue
        diffs.append(abs(sx[i][1] - sy[i][1]))
    if not diffs:
        print("无可比字 (序列错位)")
        return
    diffs.sort()
    aas = sum(diffs) / len(diffs)
    p50 = diffs[len(diffs) // 2]
    p90 = diffs[int(len(diffs) * 0.9)]
    h = lambda t: sum(1 for d in diffs if d <= t) / len(diffs)
    print(f"对齐字={len(diffs)} 错位跳过={mism}")
    print(f"AAS(start差)={aas*1000:.0f}ms  p50={p50*1000:.0f}ms  p90={p90*1000:.0f}ms")
    print(f"命中: ≤100ms={h(0.1):.1%}  ≤200ms={h(0.2):.1%}  ≤500ms={h(0.5):.1%}")


if __name__ == "__main__":
    main()
