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

    def test_pure_single_char_backchannel(self) -> None:
        """单字短语气词 '对' 视为 backchannel."""
        from src.core.qwen3.postprocess import is_backchannel

        assert is_backchannel("对") is True

    def test_multi_char_backchannel(self) -> None:
        """多字 backchannel '嗯嗯' / '好的' / '是的' 都视为 backchannel."""
        from src.core.qwen3.postprocess import is_backchannel

        assert is_backchannel("嗯嗯") is True
        assert is_backchannel("好的") is True
        assert is_backchannel("是的") is True

    def test_long_text_is_not_backchannel(self) -> None:
        """完整句子 (即使含 backchannel token) 不算 backchannel."""
        from src.core.qwen3.postprocess import is_backchannel

        assert is_backchannel("这是一段完整的句子描述") is False
        assert is_backchannel("我们今天讨论一下 AI") is False

    def test_punctuation_only_is_backchannel(self) -> None:
        """只有标点 (如 '。' / '？') 视为 backchannel — ASR 噪声段."""
        from src.core.qwen3.postprocess import is_backchannel

        assert is_backchannel("。") is True
        assert is_backchannel("？") is True
        assert is_backchannel("，") is True
