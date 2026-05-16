"""
PR2 — qwen3_transcriber 集成 short-segment guard 的 helper.

测试 apply_short_segment_guard_to_segments(merged_segments, qwen3_config):
- enabled=False 直接返回原 List[Segment]
- enabled=True 转 dict -> apply guard -> 转回 List[Segment]
- 转换前后 Segment 字段语义保持 (start/end/speaker/text)
"""
from __future__ import annotations

from src.core.config import Qwen3Config
from src.core.qwen3.merge import Segment


class TestApplyShortSegmentGuardToSegments:
    def test_disabled_returns_original_segments_list(self) -> None:
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config(short_segment_guard_enabled=False)
        segs = [
            Segment(start=0.0, end=0.1, speaker=0, text="ghost"),  # tiny 但应保留
            Segment(start=0.1, end=2.0, speaker=1, text="正常"),
        ]
        out, stats = apply_short_segment_guard_to_segments(segs, cfg)
        assert out == segs
        assert stats.get("enabled") is False

    def test_enabled_drops_ghost_and_returns_segment_objects(self) -> None:
        """enabled=True 时, 0s ghost 段被 drop, 返回类型仍是 List[Segment]."""
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config(
            short_segment_guard_enabled=True,
            short_segment_drop_sec=0.5,  # ghost dur=0 < 0.5
            short_segment_aba_max_mid_sec=1.5,
            short_segment_merge_same=True,
        )
        segs = [
            Segment(start=0.0, end=3.0, speaker=0, text="前段"),
            Segment(start=3.0, end=3.0, speaker=1, text="幽灵"),  # 0s ghost
            Segment(start=3.5, end=6.0, speaker=0, text="后段"),
        ]
        out, stats = apply_short_segment_guard_to_segments(segs, cfg)
        # 转换类型保持
        assert all(isinstance(s, Segment) for s in out)
        # ghost 段已被合并
        assert len(out) <= 2
        assert stats.get("enabled") is True
        assert stats.get("drop", {}).get("dropped_total", 0) >= 1

    def test_enabled_preserves_segment_field_semantics(self) -> None:
        """guard 后 Segment 的 start/end 为 float, speaker 为 int, text 为 str."""
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config()
        segs = [
            Segment(start=0.0, end=3.0, speaker=0, text="A"),
            Segment(start=3.0, end=6.0, speaker=1, text="B"),
        ]
        out, _stats = apply_short_segment_guard_to_segments(segs, cfg)
        for s in out:
            assert isinstance(s.start, float)
            assert isinstance(s.end, float)
            assert isinstance(s.speaker, int)
            assert isinstance(s.text, str)
