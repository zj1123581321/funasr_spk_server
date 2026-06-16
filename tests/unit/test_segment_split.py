"""diarize 开关 (落地步骤 4, D5+T1) — split_long_segments 分层切段纯函数

diarize=false 时段 = ASR ~40s chunk, SRT 不可用 → 对超长段做两层 fallback 切分:
1. 有 segment.words (word_align, CUDA 主力路径默认开) → 词隙中点精确切
2. 无 words → 静音表切 (silence midpoints) + char-ratio 文本归属 (现役机制)
无候选切点时硬切在目标等分点 (保证 max_dur 上界), 内置最小段时长阈值.
"""
from __future__ import annotations

import pytest

from src.core.qwen3.merge import Segment
from src.core.qwen3.segment_split import split_long_segments


def _speech_regions_with_gap_at(*gap_centers, width=0.4, total=100.0):
    """构造 speech_regions, 在指定时间点留 width 宽静音 (silence mid = gap_center)"""
    cuts = sorted(gap_centers)
    regions = []
    cursor = 0.0
    for c in cuts:
        regions.append({"start": cursor, "end": c - width / 2})
        cursor = c + width / 2
    regions.append({"start": cursor, "end": total})
    return regions


class TestShortSegmentsUntouched:
    def test_short_segment_passthrough(self):
        segs = [Segment(start=0.0, end=10.0, speaker=0, text="短段")]
        out, stats = split_long_segments(
            segs, speech_regions=[], audio_duration=10.0, max_dur=12.0, min_dur=1.5,
        )
        assert out == segs
        assert stats["split_segments"] == 0

    def test_empty_input(self):
        out, stats = split_long_segments(
            [], speech_regions=[], audio_duration=0.0, max_dur=12.0, min_dur=1.5,
        )
        assert out == []


class TestWordLevelSplit:
    def _seg_with_words(self):
        # 40s 段, 词均匀分布, 词隙在任意位置可切
        words = [
            {"text": f"w{i}", "start": i * 2.0, "end": i * 2.0 + 1.0, "score": None}
            for i in range(20)  # 0..39s
        ]
        return Segment(start=0.0, end=40.0, speaker=0, text="字" * 40, words=words)

    def test_long_segment_with_words_splits_at_word_gaps(self):
        seg = self._seg_with_words()
        out, stats = split_long_segments(
            [seg], speech_regions=[], audio_duration=40.0, max_dur=12.0, min_dur=1.5,
        )
        assert len(out) >= 3, "40s/max12 应切成 >=3 片"
        assert stats["split_segments"] == 1
        assert stats["word_split"] == 1
        assert stats["silence_split"] == 0
        # 文本无损
        assert "".join(p.text for p in out) == seg.text
        # 切点落在词隙 (词中点不被切开): 每片边界不在任何词的 [start,end] 内部
        for p in out[:-1]:
            for w in seg.words:
                assert not (w["start"] < p.end < w["end"]), f"切点 {p.end} 切进词 {w}"
        # 片时长有上界 (词隙近似等分, 允许 1 个词隙的余量)
        assert all(p.end - p.start <= 12.0 + 3.0 for p in out)
        # speaker 透传
        assert all(p.speaker == 0 for p in out)

    def test_words_redistributed_to_pieces(self):
        seg = self._seg_with_words()
        out, _ = split_long_segments(
            [seg], speech_regions=[], audio_duration=40.0, max_dur=12.0, min_dur=1.5,
        )
        all_words = [w for p in out if p.words for w in p.words]
        assert len(all_words) == 20, "词必须无损落回各片"
        # 词归属正确: 词中点落在片时间窗内
        for p in out:
            for w in p.words or []:
                mid = (w["start"] + w["end"]) / 2
                assert p.start <= mid <= p.end


class TestSilenceFallbackSplit:
    def test_long_segment_without_words_splits_at_silence_mid(self):
        seg = Segment(start=0.0, end=40.0, speaker=0, text="字" * 40)
        regions = _speech_regions_with_gap_at(13.0, 26.0, total=40.0)
        out, stats = split_long_segments(
            [seg], speech_regions=regions, audio_duration=40.0, max_dur=15.0, min_dur=1.5,
        )
        assert len(out) == 3
        assert stats["silence_split"] == 1
        assert stats["word_split"] == 0
        # 切点 = 静音中点
        assert out[0].end == pytest.approx(13.0, abs=0.01)
        assert out[1].end == pytest.approx(26.0, abs=0.01)
        assert "".join(p.text for p in out) == seg.text

    def test_no_candidates_hard_cut_keeps_max_bound(self):
        """无任何静音候选 → 硬切在目标等分点, 保证 max_dur 上界"""
        seg = Segment(start=0.0, end=40.0, speaker=0, text="字" * 40)
        out, stats = split_long_segments(
            [seg], speech_regions=[{"start": 0.0, "end": 40.0}],  # 全程 speech 无静音
            audio_duration=40.0, max_dur=12.0, min_dur=1.5,
        )
        assert len(out) == 4
        assert stats["hard_cuts"] == 3
        assert all(p.end - p.start <= 12.0 + 0.01 for p in out)
        assert "".join(p.text for p in out) == seg.text

    def test_min_dur_respected(self):
        """切出的片不短于 min_dur"""
        seg = Segment(start=0.0, end=14.0, speaker=0, text="字" * 14)
        # 静音中点贴边 (0.5s), 不可用 → 选目标点附近
        regions = _speech_regions_with_gap_at(0.5, 7.0, total=14.0)
        out, _ = split_long_segments(
            [seg], speech_regions=regions, audio_duration=14.0, max_dur=10.0, min_dur=2.0,
        )
        assert all(p.end - p.start >= 2.0 - 0.01 for p in out)
        assert "".join(p.text for p in out) == seg.text


class TestMixedBatch:
    def test_mixed_segments_each_handled(self):
        words = [
            {"text": f"w{i}", "start": 50 + i * 2.0, "end": 50 + i * 2.0 + 1.0, "score": None}
            for i in range(20)
        ]
        segs = [
            Segment(start=0.0, end=8.0, speaker=0, text="短"),
            Segment(start=10.0, end=45.0, speaker=0, text="字" * 35),          # 静音切
            Segment(start=50.0, end=90.0, speaker=0, text="词" * 40, words=words),  # 词切
        ]
        regions = _speech_regions_with_gap_at(22.0, 33.0, total=100.0)
        out, stats = split_long_segments(
            segs, speech_regions=regions, audio_duration=100.0, max_dur=14.0, min_dur=1.5,
        )
        assert stats["split_segments"] == 2
        assert stats["word_split"] == 1
        assert stats["silence_split"] == 1
        assert out[0].text == "短"
        full_text = "".join(p.text for p in out)
        assert full_text == "短" + "字" * 35 + "词" * 40
