#!/usr/bin/env python3
# coding=utf-8
"""把 LexGoGo calibrated 参考文本清洗成连续纯文本(去 --- 头块 + 说话人前缀).

用于喂 fa-zh 做长音频对齐压力测试.
"""
import argparse
import re
from pathlib import Path


def clean(raw: str) -> str:
    lines = raw.splitlines()
    # 去掉开头 --- ... --- 之间的 YAML 头块
    body_start = 0
    dash_seen = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "---":
            dash_seen += 1
            if dash_seen == 2:
                body_start = i + 1
                break
    body = lines[body_start:] if dash_seen >= 2 else lines

    out = []
    for ln in body:
        ln = ln.strip()
        if not ln:
            continue
        # 去 "说话人：" / "说话人:" 前缀
        ln = re.sub(r"^[^　-鿿\w]{0,2}[一-鿿A-Za-z]{1,8}[：:]", "", ln, count=1)
        out.append(ln)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.inp).read_text(encoding="utf-8")
    cleaned = clean(raw)
    Path(args.out).write_text(cleaned, encoding="utf-8")
    print(f"cleaned: {len(raw)} -> {len(cleaned)} 字, 落 {args.out}")


if __name__ == "__main__":
    main()
