"""
Qwen3-Diarize 后处理 (PR2 short-segment guard).

按 TDD 红绿循环逐步实现, 每次只为当前红测试加最少代码.
"""
from __future__ import annotations

import re


_BACKCHANNEL_TOKENS = {"对", "嗯嗯", "好的", "是的"}
_PURE_PUNCT_RE = re.compile(r"^[，。！？!?,.\s]+$")
_QUESTION_TAIL_RE = re.compile(r"(对吗|是吧|是不是|有没有|可以吗|好吗|是吗)$")


def is_backchannel(text: str) -> bool:
    """判断文本是否为 backchannel (空/单字短语气词/多字 backchannel/纯标点).

    Args:
        text: segment 的 ASR 文本.
    """
    if not text:
        return True
    stripped = text.strip()
    if stripped in _BACKCHANNEL_TOKENS:
        return True
    if _PURE_PUNCT_RE.match(text):
        return True
    return False


_QUESTION_TAIL_MAX_LEN = 14


def is_question_tail(text: str) -> bool:
    """判断文本是否为短问句尾巴 (含 '对吗' / '是吧' 等 marker 且 ≤14 字).

    Args:
        text: segment 的 ASR 文本.
    """
    stripped = (text or "").strip()
    if len(stripped) > _QUESTION_TAIL_MAX_LEN:
        return False
    return bool(_QUESTION_TAIL_RE.search(stripped))


def drop_tiny_segments(
    segments: list[dict], min_sec: float
) -> tuple[list[dict], dict]:
    """合并 dur < min_sec 的微短段到时间最近邻段.

    Args:
        segments: ordered list of {start, end, speaker, text}.
        min_sec: 短段阈值, dur 小于此值且 text 非空才会被合并.

    Returns:
        (new_segments, stats) — stats 含 dropped_total / merged_into_prev / merged_into_next.
    """
    out: list[dict] = []
    dropped = 0
    merged_into_prev = 0
    merged_into_next = 0
    skip: set[int] = set()
    for i, seg in enumerate(segments):
        if i in skip:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        text = (seg.get("text") or "").strip()
        if dur >= min_sec or not text:
            out.append(seg)
            continue
        # 微短且 text 非空: 选 gap 更近一侧合并
        prev_seg = out[-1] if out else None
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        target = None
        if next_seg is not None and prev_seg is not None:
            gap_prev = float(seg["start"]) - float(prev_seg["end"])
            gap_next = float(next_seg["start"]) - float(seg["end"])
            target = "prev" if gap_prev <= gap_next else "next"
        elif next_seg is not None:
            target = "next"
        elif prev_seg is not None:
            target = "prev"
        if target == "prev":
            prev_seg["text"] = (prev_seg.get("text") or "") + (seg.get("text") or "")
            prev_seg["end"] = max(float(prev_seg["end"]), float(seg["end"]))
            merged_into_prev += 1
            dropped += 1
        elif target == "next":
            new_next = dict(next_seg)
            new_next["text"] = (seg.get("text") or "") + (next_seg.get("text") or "")
            new_next["start"] = min(float(seg["start"]), float(next_seg["start"]))
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
