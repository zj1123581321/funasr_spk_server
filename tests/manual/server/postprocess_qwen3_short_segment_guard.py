"""Short-segment guard postprocess for Qwen3 PoC speaker labels.

诊断 v8e_gate vs v7: refine 流程在短段上引入了 20 个 regression, 其中:
- 7 个 < 0.3s "幽灵段" (ForcedAligner token 间隙产物)
- 13 个 0.3-3s 短回应/碎片, speaker 被分错

本脚本对已有 PoC JSON 做无 GPU 后处理:
1. 合并 < short_drop_sec 的孤立微短段到时间相邻段 (按相邻段 speaker)
2. 1-3s 短段如果是 A-B-A 抖动 (前后段同 speaker, 中间不同) 且文本特征像回应
   (短文本 / 含 "对/嗯/啊/是" 类 backchannel), 强制改回 A
3. 短段如果与前段 gap <= merge_gap_sec 且 speaker 不同, 计算"该并入前 or 后", 倾向于
   文本相邻 punctuation 一致 + 与前段 speaker 一致 (避免被 forced-aligner 切碎)

输入:已有 v8e/v9 等 *.qwen3_long_poc.json
输出:同 schema 的新 JSON + 处理统计 summary
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BACKCHANNEL = re.compile(r"^[，。?!?,.\s]*(对|嗯|啊|是|哦|噢|呃|嗯嗯|对啊|对对|是的|好的)[啊嗯，。!?,.\s]*$")
# 短问句尾巴 — 不一定是 backchannel, 但常出现在转 turn 处
QUESTION_TAIL = re.compile(r"(对吗|是吧|是不是|有没有|可以吗|好吗|是吗)$")


def is_backchannel(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if BACKCHANNEL.match(t):
        return True
    # 极短 (<=4 字) 且全是常见短词
    if len(t) <= 4 and re.match(r"^[对嗯啊是哦噢呃好的吗，。?!吧呢呀\s]+$", t):
        return True
    return False


def is_question_tail(text: str) -> bool:
    t = (text or "").strip()
    return bool(QUESTION_TAIL.search(t)) and len(t) <= 14


def drop_tiny_segments(segments: list[dict], min_sec: float) -> tuple[list[dict], dict]:
    """合并 < min_sec 的微短段到时间最近邻段."""
    out: list[dict] = []
    dropped = 0
    merged_into_prev = 0
    merged_into_next = 0
    skip = set()
    for i, seg in enumerate(segments):
        if i in skip:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur >= min_sec or not (seg.get("text") or "").strip():
            out.append(seg)
            continue
        # 微短: 合并到 prev / next 中文本更接近 punctuation 边界的一边
        prev_seg = out[-1] if out else None
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        # 优先并到 next (常见: 短段是下一句开头碎片)
        target = None
        if next_seg is not None:
            gap_next = float(next_seg["start"]) - float(seg["end"])
            if prev_seg is not None:
                gap_prev = float(seg["start"]) - float(prev_seg["end"])
                if gap_prev <= gap_next:
                    target = "prev"
                else:
                    target = "next"
            else:
                target = "next"
        elif prev_seg is not None:
            target = "prev"
        if target == "prev" and prev_seg is not None:
            prev_seg["text"] = (prev_seg.get("text") or "") + (seg.get("text") or "")
            prev_seg["end"] = max(float(prev_seg["end"]), float(seg["end"]))
            merged_into_prev += 1
            dropped += 1
        elif target == "next" and next_seg is not None:
            # 改 next 的 start 与 text (修改原 list 不影响外层, 但要把 next 当作 seg 重新写到 out)
            new_next = dict(next_seg)
            new_next["text"] = (seg.get("text") or "") + (next_seg.get("text") or "")
            new_next["start"] = min(float(seg["start"]), float(next_seg["start"]))
            # 标记 i+1 跳过, 我们写新的 new_next 替代
            skip.add(i + 1)
            out.append(new_next)
            merged_into_next += 1
            dropped += 1
        else:
            out.append(seg)
    return out, {
        "dropped_total": dropped,
        "merged_into_prev": merged_into_prev,
        "merged_into_next": merged_into_next,
    }


def aba_smoothing(segments: list[dict], max_mid_sec: float) -> tuple[list[dict], dict]:
    """A-B-A 抖动平滑: 短中间段强制改回 A speaker (不动 text)."""
    changed = 0
    backchannel_changes = 0
    out = [dict(s) for s in segments]
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if str(a.get("speaker")) != str(c.get("speaker")):
            continue
        if str(a.get("speaker")) == str(b.get("speaker")):
            continue
        dur = float(b["end"]) - float(b["start"])
        if dur > max_mid_sec:
            continue
        # 接受规则:
        # 1) 文本是 backchannel  → 改
        # 2) 文本是 question_tail 且短  → 改
        # 3) 极短 (<= 1.0s) 且 char/sec >= 5 (高密度短碎片)  → 改 (含混合切碎)
        text = (b.get("text") or "").strip()
        cps = (len(text) / dur) if dur > 0 else 0
        accept = False
        if is_backchannel(text):
            accept = True
            backchannel_changes += 1
        elif is_question_tail(text):
            accept = True
        elif dur <= 1.0 and cps >= 5.0:
            accept = True
        if accept:
            out[i]["speaker"] = a["speaker"]
            changed += 1
    return out, {
        "aba_changed": changed,
        "aba_backchannel_changes": backchannel_changes,
    }


def merge_consecutive_same_speaker(segments: list[dict], merge_gap_sec: float = 0.05) -> tuple[list[dict], int]:
    """合并连续同 speaker 段, 减少碎片."""
    if not segments:
        return segments, 0
    out = [dict(segments[0])]
    merged = 0
    for s in segments[1:]:
        prev = out[-1]
        if str(prev.get("speaker")) == str(s.get("speaker")) and float(s["start"]) - float(prev["end"]) <= merge_gap_sec:
            prev["text"] = (prev.get("text") or "") + (s.get("text") or "")
            prev["end"] = max(float(prev["end"]), float(s["end"]))
            merged += 1
        else:
            out.append(dict(s))
    return out, merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Short-segment guard for Qwen3 PoC speaker labels")
    ap.add_argument("input_json", type=Path)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--short-drop-sec", type=float, default=0.3,
                    help="< 这个秒数的微短段并入邻居")
    ap.add_argument("--aba-max-mid-sec", type=float, default=1.5,
                    help="A-B-A 抖动平滑的中间段最大秒数")
    ap.add_argument("--no-merge-same", action="store_true",
                    help="禁用合并连续同 speaker (默认开)")
    args = ap.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    stats_log: dict = {"input_segments": len(segments)}

    segs1, stat_drop = drop_tiny_segments(segments, args.short_drop_sec)
    stats_log.update(stat_drop)
    stats_log["after_drop_segments"] = len(segs1)

    segs2, stat_aba = aba_smoothing(segs1, args.aba_max_mid_sec)
    stats_log.update(stat_aba)
    stats_log["after_aba_segments"] = len(segs2)

    if not args.no_merge_same:
        segs3, merged = merge_consecutive_same_speaker(segs2)
        stats_log["merge_same_count"] = merged
        stats_log["after_merge_segments"] = len(segs3)
    else:
        segs3 = segs2

    new_payload = dict(payload)
    new_payload["segments"] = segs3
    summary = dict(new_payload.get("summary") or {})
    summary["short_segment_guard"] = {
        "source_json": str(args.input_json),
        "short_drop_sec": args.short_drop_sec,
        "aba_max_mid_sec": args.aba_max_mid_sec,
        **stats_log,
    }
    new_payload["summary"] = summary

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(new_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats_log, ensure_ascii=False, indent=2))
    print(f"[out] {args.out_json}")


if __name__ == "__main__":
    main()
