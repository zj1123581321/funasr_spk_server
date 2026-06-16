"""
critical 回归 (codex #6): apply_short_segment_guard_to_segments 必须透传 Segment.words

helper 把 Segment 转 dict 再转回, 老实现只搬 start/end/speaker/text, 会静默丢词.
本测试钉死: guard 后 (pass-through 的段) words 不丢.
"""
from __future__ import annotations

from src.core.qwen3.merge import Segment
from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments


class _Cfg:
    short_segment_guard_enabled = True
    short_segment_drop_sec = 1.5
    short_segment_aba_max_mid_sec = 1.5
    short_segment_merge_same = True


def test_guard_preserves_words_on_passthrough_segments():
    segs = [
        Segment(
            start=0.0, end=5.0, speaker=0, text="第一段",
            words=[{"text": "第", "start": 0.0, "end": 0.3, "score": -1.0}],
        ),
        Segment(
            start=5.0, end=10.0, speaker=1, text="第二段",
            words=[{"text": "二", "start": 5.0, "end": 5.3, "score": -1.0}],
        ),
    ]
    out, stats = apply_short_segment_guard_to_segments(segs, _Cfg())
    # 两段都够长 + 不同 speaker → pass-through, 不 drop 不 merge
    assert len(out) == 2
    assert out[0].words is not None and out[0].words[0]["text"] == "第"
    assert out[1].words is not None and out[1].words[0]["text"] == "二"


def test_guard_disabled_keeps_words():
    class _Off(_Cfg):
        short_segment_guard_enabled = False

    segs = [
        Segment(
            start=0.0, end=5.0, speaker=0, text="x",
            words=[{"text": "x", "start": 0.0, "end": 0.3, "score": -1.0}],
        )
    ]
    out, stats = apply_short_segment_guard_to_segments(segs, _Off())
    assert out[0].words[0]["text"] == "x"
