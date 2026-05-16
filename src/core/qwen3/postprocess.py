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
