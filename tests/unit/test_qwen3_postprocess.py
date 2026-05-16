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


class TestIsQuestionTail:
    """is_question_tail: 识别短问句尾巴 (常出现在 turn 切换处)."""

    def test_short_question_tail(self) -> None:
        """短问句尾 (≤14 字) 含 '对吗' 等 marker 视为问句尾巴."""
        from src.core.qwen3.postprocess import is_question_tail

        assert is_question_tail("对吗") is True
        assert is_question_tail("你说的对吗") is True

    def test_long_text_with_marker_not_question_tail(self) -> None:
        """超过 14 字即使尾部含 marker, 也不算"短问句尾巴" — 太长不是 turn 切换信号."""
        from src.core.qwen3.postprocess import is_question_tail

        long_text = "这是一段非常长非常长非常长非常长非常长非常长的话你说对吗"
        assert len(long_text) > 14
        assert is_question_tail(long_text) is False

    def test_no_marker_text_is_not_question_tail(self) -> None:
        """无问句 marker 的陈述句不是问句尾巴."""
        from src.core.qwen3.postprocess import is_question_tail

        assert is_question_tail("今天天气好") is False
        assert is_question_tail("") is False
        assert is_question_tail("这是一段完整的陈述句") is False


def _seg(start: float, end: float, speaker: str, text: str) -> dict:
    """测试辅助: 构造一个 segment dict."""
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class TestDropTinySegments:
    """drop_tiny_segments: 合并 < min_sec 微短段到时间最近邻段."""

    def test_keeps_normal_segments(self) -> None:
        """全部段 ≥ min_sec, 应原样返回."""
        from src.core.qwen3.postprocess import drop_tiny_segments

        segments = [
            _seg(0.0, 3.0, "0", "你好今天天气真好"),
            _seg(3.0, 6.5, "1", "嗯是的我觉得也很好"),
        ]
        out, stats = drop_tiny_segments(segments, min_sec=1.5)
        assert out == segments
        assert stats["dropped_total"] == 0

    def test_drops_zero_duration_ghost_with_text_into_nearer_prev(self) -> None:
        """0s 幽灵段含 text, gap_prev=0 <= gap_next, 应合并到 prev."""
        from src.core.qwen3.postprocess import drop_tiny_segments

        segments = [
            _seg(0.0, 3.0, "0", "前段文本"),
            _seg(3.0, 3.0, "1", "幽灵"),  # dur=0
            _seg(3.5, 6.0, "0", "后段文本"),
        ]
        out, stats = drop_tiny_segments(segments, min_sec=1.5)
        assert len(out) == 2
        assert out[0]["text"] == "前段文本幽灵"
        assert out[0]["end"] == 3.0
        assert out[1] == segments[2]
        assert stats["dropped_total"] == 1
        assert stats["merged_into_prev"] == 1

    def test_chooses_next_when_gap_next_smaller(self) -> None:
        """tiny 段中间, gap_next < gap_prev, 应合并到 next."""
        from src.core.qwen3.postprocess import drop_tiny_segments

        segments = [
            _seg(0.0, 2.0, "0", "前"),
            _seg(5.0, 5.5, "1", "中"),  # dur=0.5, gap_prev=3.0, gap_next=0.1
            _seg(5.6, 8.0, "0", "后"),  # dur=2.4, 不算 tiny
        ]
        out, stats = drop_tiny_segments(segments, min_sec=1.5)
        assert len(out) == 2
        assert out[0] == segments[0]
        assert out[1]["start"] == 5.0  # min(5.0, 5.6)
        assert out[1]["text"] == "中后"
        assert out[1]["end"] == 8.0
        assert stats["merged_into_next"] == 1
        assert stats["dropped_total"] == 1

    def test_first_segment_tiny_merges_to_next(self) -> None:
        """首段 tiny, 没有 prev, 只能合并到 next."""
        from src.core.qwen3.postprocess import drop_tiny_segments

        segments = [
            _seg(0.0, 0.5, "0", "首"),  # tiny, 无 prev
            _seg(0.6, 3.0, "1", "正文"),
        ]
        out, stats = drop_tiny_segments(segments, min_sec=1.5)
        assert len(out) == 1
        assert out[0]["start"] == 0.0
        assert out[0]["text"] == "首正文"
        assert stats["merged_into_next"] == 1

    def test_last_segment_tiny_merges_to_prev(self) -> None:
        """末段 tiny, 没有 next, 只能合并到 prev."""
        from src.core.qwen3.postprocess import drop_tiny_segments

        segments = [
            _seg(0.0, 3.0, "0", "前段"),
            _seg(3.0, 3.3, "1", "尾"),  # tiny, 无 next
        ]
        out, stats = drop_tiny_segments(segments, min_sec=1.5)
        assert len(out) == 1
        assert out[0]["text"] == "前段尾"
        assert out[0]["end"] == 3.3
        assert stats["merged_into_prev"] == 1


class TestABASmoothing:
    """aba_smoothing: A-B-A 抖动短中间段强制改回 A speaker (不动 text)."""

    def test_backchannel_aba_replaced(self) -> None:
        """A-B-A 三段中 B 是 backchannel, 改 B.speaker 为 A.speaker."""
        from src.core.qwen3.postprocess import aba_smoothing

        segments = [
            _seg(0.0, 2.0, "0", "正常说话"),
            _seg(2.0, 2.5, "1", "对"),  # backchannel, dur=0.5
            _seg(2.5, 5.0, "0", "继续说"),
        ]
        out, stats = aba_smoothing(segments, max_mid_sec=1.5)
        assert out[1]["speaker"] == "0"
        assert out[1]["text"] == "对"  # text 不动
        assert stats["changed"] == 1

    def test_long_mid_segment_not_changed(self) -> None:
        """A-B-A 三段, B 太长 (> max_mid_sec) 不改 — 长段是真正说话."""
        from src.core.qwen3.postprocess import aba_smoothing

        segments = [
            _seg(0.0, 2.0, "0", "前"),
            _seg(2.0, 5.0, "1", "对"),  # dur=3.0 > max_mid_sec=1.5
            _seg(5.0, 7.0, "0", "后"),
        ]
        out, stats = aba_smoothing(segments, max_mid_sec=1.5)
        assert out[1]["speaker"] == "1"
        assert stats["changed"] == 0

    def test_different_outer_speakers_not_changed(self) -> None:
        """A-B-C 三段外层 speaker 不同, 不算 ABA, 不改."""
        from src.core.qwen3.postprocess import aba_smoothing

        segments = [
            _seg(0.0, 2.0, "0", "前"),
            _seg(2.0, 2.5, "1", "对"),  # backchannel, 但 A != C
            _seg(2.5, 5.0, "2", "后"),  # 不同于 A
        ]
        out, stats = aba_smoothing(segments, max_mid_sec=1.5)
        assert out[1]["speaker"] == "1"
        assert stats["changed"] == 0

    def test_high_density_short_carbage_replaced(self) -> None:
        """A-B-A 中 B 高 char/sec (≥5 cps) 且极短 (≤1.0s) 视为切碎噪声 → 改."""
        from src.core.qwen3.postprocess import aba_smoothing

        segments = [
            _seg(0.0, 2.0, "0", "前"),
            _seg(2.0, 2.5, "1", "高密度话语片段"),  # dur=0.5, len=8, cps=16
            _seg(2.5, 5.0, "0", "后"),
        ]
        out, stats = aba_smoothing(segments, max_mid_sec=1.5)
        assert out[1]["speaker"] == "0"
        assert stats["changed"] == 1
