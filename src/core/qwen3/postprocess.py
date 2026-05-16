"""
Qwen3-Diarize 后处理 (PR2 short-segment guard).

按 TDD 红绿循环逐步实现, 每次只为当前红测试加最少代码.
"""
from __future__ import annotations


def is_backchannel(text: str) -> bool:
    """判断文本是否为 backchannel (空/单字短语气词).

    Args:
        text: segment 的 ASR 文本.
    """
    if not text:
        return True
    return False
