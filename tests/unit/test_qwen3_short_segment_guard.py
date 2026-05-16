"""Unit tests for short-segment guard postprocess (PR4 long-audio quality).

Verifies that:
- < short_drop_sec 微短段并入相邻段, text 保留
- A-B-A 抖动平滑只在符合规则时触发 (backchannel / question_tail / 高密度短碎片)
- merge_consecutive_same_speaker 合并连续同 speaker 段
- 普通段不被误处理
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.manual.server.postprocess_qwen3_short_segment_guard import (
    aba_smoothing,
    drop_tiny_segments,
    is_backchannel,
    is_question_tail,
    merge_consecutive_same_speaker,
)


def _seg(start: float, end: float, speaker: str, text: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class TestIsBackchannel:
    def test_empty_text_is_backchannel(self):
        assert is_backchannel("")

    def test_pure_backchannel(self):
        for t in ["对", "嗯", "对啊", "嗯嗯", "是的", "好的"]:
            assert is_backchannel(t), t

    def test_long_text_not_backchannel(self):
        assert not is_backchannel("我们今天来聊一下大模型的事情")

    def test_punctuation_only_backchannel(self):
        assert is_backchannel("，")


class TestIsQuestionTail:
    def test_short_question_tail(self):
        assert is_question_tail("对吗")
        assert is_question_tail("是不是")
        assert is_question_tail("你觉得呢有没有")

    def test_long_text_not_tail(self):
        assert not is_question_tail("我觉得你说的可能不太对你觉得呢有没有可能不一样" * 2)

    def test_no_marker(self):
        assert not is_question_tail("我先说")


class TestDropTinySegments:
    def test_keeps_normal_segments(self):
        segs = [
            _seg(0.0, 3.0, "A", "正常段一"),
            _seg(3.0, 6.0, "B", "正常段二"),
        ]
        out, stats = drop_tiny_segments(segs, min_sec=0.3)
        assert len(out) == 2
        assert stats["dropped_total"] == 0

    def test_drops_zero_duration_ghost(self):
        segs = [
            _seg(0.0, 3.0, "A", "前段"),
            _seg(3.0, 3.0, "B", "幽"),
            _seg(3.1, 6.0, "A", "后段"),
        ]
        out, stats = drop_tiny_segments(segs, min_sec=0.3)
        assert stats["dropped_total"] == 1
        # 幽灵段应被合并; 总段数 -1
        assert len(out) == 2
        # 文本要保留 (并入 prev 或 next)
        all_text = "".join(s["text"] for s in out)
        assert "幽" in all_text
        assert "前段" in all_text
        assert "后段" in all_text

    def test_drops_subsec_segment_into_nearer_neighbor(self):
        # 中间段 0.2s, gap to prev=0, gap to next=0 → 优先 prev (gap_prev <= gap_next)
        segs = [
            _seg(0.0, 3.0, "A", "前段"),
            _seg(3.0, 3.2, "B", "小"),
            _seg(3.2, 6.0, "A", "后段"),
        ]
        out, stats = drop_tiny_segments(segs, min_sec=0.5)
        assert stats["dropped_total"] == 1
        assert stats["merged_into_prev"] == 1
        # prev 段被扩展, 文本含 "前段小"
        assert out[0]["text"] == "前段小"
        assert out[0]["end"] >= 3.2


class TestABASmoothing:
    def test_backchannel_aba_replaced(self):
        segs = [
            _seg(0.0, 3.0, "A", "我觉得这事儿挺有意思"),
            _seg(3.0, 4.0, "B", "对啊"),  # backchannel
            _seg(4.0, 7.0, "A", "然后我们再考虑下一个问题"),
        ]
        out, stats = aba_smoothing(segs, max_mid_sec=1.5)
        assert stats["aba_changed"] == 1
        assert stats["aba_backchannel_changes"] == 1
        assert out[1]["speaker"] == "A"

    def test_long_mid_segment_not_changed(self):
        segs = [
            _seg(0.0, 3.0, "A", "前面"),
            _seg(3.0, 6.0, "B", "中间段挺长的不应该被平滑掉因为这个段不是短回应"),
            _seg(6.0, 9.0, "A", "后面"),
        ]
        out, stats = aba_smoothing(segs, max_mid_sec=1.5)
        assert stats["aba_changed"] == 0
        assert out[1]["speaker"] == "B"

    def test_different_outer_speakers_not_changed(self):
        # 不是 A-B-A 模式
        segs = [
            _seg(0.0, 3.0, "A", "前面"),
            _seg(3.0, 4.0, "B", "对"),
            _seg(4.0, 7.0, "C", "后面"),
        ]
        out, stats = aba_smoothing(segs, max_mid_sec=1.5)
        assert stats["aba_changed"] == 0
        assert out[1]["speaker"] == "B"

    def test_high_density_short_carbage_replaced(self):
        # dur 0.5s, 5 字 → cps=10 触发条件
        segs = [
            _seg(0.0, 3.0, "A", "前面"),
            _seg(3.0, 3.5, "B", "这一个产品"),
            _seg(3.5, 6.0, "A", "后面"),
        ]
        out, stats = aba_smoothing(segs, max_mid_sec=1.5)
        assert stats["aba_changed"] == 1
        assert out[1]["speaker"] == "A"


class TestMergeConsecutiveSameSpeaker:
    def test_merges_adjacent_same_speaker(self):
        segs = [
            _seg(0.0, 3.0, "A", "段一"),
            _seg(3.0, 6.0, "A", "段二"),
            _seg(6.0, 9.0, "B", "段三"),
        ]
        out, merged = merge_consecutive_same_speaker(segs)
        assert merged == 1
        assert len(out) == 2
        assert out[0]["text"] == "段一段二"
        assert out[0]["end"] == 6.0

    def test_keeps_speaker_changes(self):
        segs = [
            _seg(0.0, 3.0, "A", "段一"),
            _seg(3.0, 6.0, "B", "段二"),
        ]
        out, merged = merge_consecutive_same_speaker(segs)
        assert merged == 0
        assert len(out) == 2

    def test_respects_gap(self):
        # 大 gap > merge_gap_sec → 不合并
        segs = [
            _seg(0.0, 3.0, "A", "段一"),
            _seg(5.0, 8.0, "A", "段二"),
        ]
        out, merged = merge_consecutive_same_speaker(segs, merge_gap_sec=0.05)
        assert merged == 0
        assert len(out) == 2


class TestIntegration:
    def test_v12_default_pipeline_on_ghost_and_aba(self):
        # 综合: 含幽灵段 + ABA 抖动 + 同 speaker 相邻
        segs = [
            _seg(0.0, 3.0, "A", "段一"),
            _seg(3.0, 3.05, "B", "幽"),       # 0.05s 幽灵 -> drop into prev
            _seg(3.05, 6.0, "A", "段二"),     # 同 A, 与上一段合并
            _seg(6.0, 7.0, "B", "对啊"),      # 1s backchannel A-B-A
            _seg(7.0, 10.0, "A", "段三"),
        ]
        out1, _ = drop_tiny_segments(segs, min_sec=0.3)
        out2, _ = aba_smoothing(out1, max_mid_sec=1.5)
        out3, _ = merge_consecutive_same_speaker(out2)
        # 最终应该只剩 1 个 A 大段 (全部相同 speaker A)
        assert len(out3) == 1
        assert out3[0]["speaker"] == "A"
