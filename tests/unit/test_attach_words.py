"""
词→段归属纯函数 attach_words_to_segments (src/core/qwen3/merge.py)

设计:
- Segment dataclass 加 words: Optional[list] = None.
- attach_words_to_segments(segments, words): 词按时间落在哪个段的 [start,end] 窗就挂哪个段;
  边界用最大重叠时长优先; 词不落任何段 → 丢弃; 段无词 → words 保持 None (向后兼容).
- 纯函数, 不改入参 (dataclasses.replace 出新 Segment).
"""
from __future__ import annotations

from src.core.qwen3.merge import Segment, attach_words_to_segments


def _seg(start, end, speaker=0, text="x"):
    return Segment(start=start, end=end, speaker=speaker, text=text)


def test_word_attaches_to_containing_segment():
    segs = [_seg(0.0, 5.0), _seg(5.0, 10.0)]
    words = [
        {"text": "甲", "start": 1.0, "end": 1.5, "score": -1.0},
        {"text": "乙", "start": 6.0, "end": 6.5, "score": -1.1},
    ]
    out = attach_words_to_segments(segs, words)
    assert [w["text"] for w in out[0].words] == ["甲"]
    assert [w["text"] for w in out[1].words] == ["乙"]


def test_empty_words_keeps_segments_unchanged():
    segs = [_seg(0.0, 5.0)]
    out = attach_words_to_segments(segs, [])
    assert len(out) == 1
    assert out[0].words is None


def test_empty_segments_returns_empty():
    assert attach_words_to_segments([], [{"text": "x", "start": 0, "end": 1}]) == []


def test_word_outside_all_segments_dropped():
    segs = [_seg(0.0, 5.0)]
    words = [{"text": "远", "start": 100.0, "end": 100.5, "score": -1.0}]
    out = attach_words_to_segments(segs, words)
    # 词不落任何段 → 丢弃, 段 words 保持 None
    assert out[0].words is None


def test_word_crossing_boundary_goes_to_max_overlap():
    segs = [_seg(0.0, 5.0), _seg(5.0, 10.0)]
    # 词 [4.0, 7.0]: 与 seg0 重叠 1.0s, 与 seg1 重叠 2.0s → 归 seg1
    words = [{"text": "跨", "start": 4.0, "end": 7.0, "score": -1.0}]
    out = attach_words_to_segments(segs, words)
    assert out[0].words is None
    assert [w["text"] for w in out[1].words] == ["跨"]


def test_does_not_mutate_input_segments():
    segs = [_seg(0.0, 5.0)]
    words = [{"text": "甲", "start": 1.0, "end": 1.5, "score": -1.0}]
    attach_words_to_segments(segs, words)
    # 原 Segment 不被改 (纯函数)
    assert segs[0].words is None


def test_multiple_words_same_segment_preserve_order():
    segs = [_seg(0.0, 10.0)]
    words = [
        {"text": "一", "start": 1.0, "end": 1.5, "score": -1.0},
        {"text": "二", "start": 2.0, "end": 2.5, "score": -1.0},
        {"text": "三", "start": 3.0, "end": 3.5, "score": -1.0},
    ]
    out = attach_words_to_segments(segs, words)
    assert [w["text"] for w in out[0].words] == ["一", "二", "三"]
