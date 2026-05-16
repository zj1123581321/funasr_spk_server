"""
src/core/qwen3/postprocess.py 单元测试 (PR2 short-segment guard).

严格 TDD: 每个测试先写, 看红, 最小实现, 看绿, commit. 不复用 PoC 单测
(tests/unit/test_qwen3_short_segment_guard.py 仅作行为参考).
"""
from __future__ import annotations


class TestIsBackchannel:
    """is_backchannel: 判断文本是否为 backchannel (空/单字/短语气词)."""

    def test_empty_text_is_backchannel(self) -> None:
        """空字符串视为 backchannel — 后续 ABA smoothing 会拿它当"虚位"段处理."""
        from src.core.qwen3.postprocess import is_backchannel

        assert is_backchannel("") is True
