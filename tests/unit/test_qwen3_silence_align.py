"""Unit tests for silence-aware segment boundary alignment (spike 405abf6).

覆盖:
  - silence_intervals_from_speech 边界 case (空 / 全覆盖 / 乱序 / 重叠)
  - snap_segments_to_silence 核心逻辑 (吸附 / 跳过 / 0 时长保护 / 第一段 + 最后段不动)
  - diarize overlap 场景 (独立 snap start/end 不破坏 overlap)
  - apply_silence_align_to_segments helper (enabled / 空 / ffmpeg 失败 fallback)
  - 端到端: 用合成 baseline 段 + speech_regions, snap 后 align_ratio ≥ baseline
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.qwen3.merge import (
    Segment,
    SilenceInterval,
    silence_intervals_from_speech,
    snap_segments_to_silence,
)
from src.core.qwen3_transcriber import apply_silence_align_to_segments


# -------------------- helpers --------------------


def _seg(start: float, end: float, speaker: int = 0, text: str = "x") -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text=text)


def _speech(start: float, end: float) -> dict:
    return {"start": start, "end": end}


def _cfg(
    enabled: bool = True,
    tolerance: float = 2.0,
    min_dur: float = 0.1,
    noise_db: str = "-25dB",
    min_silence_sec: float = 0.2,
) -> SimpleNamespace:
    """伪 Qwen3Config (鸭子类型, 只需 silence_* 字段)."""
    return SimpleNamespace(
        silence_align_enabled=enabled,
        silence_align_tolerance_sec=tolerance,
        silence_align_min_segment_dur_sec=min_dur,
        silence_vad_noise_db=noise_db,
        silence_vad_min_silence_sec=min_silence_sec,
    )


def _align_ratio(
    segments: list[Segment],
    speech_regions: list[dict],
    audio_duration: float,
    tolerance: float = 0.3,
) -> float:
    """复刻 spike align_lib/metric.py 的核心 align_ratio 计算 (inline 简化版).

    一个段切点算 aligned, 若:
      - 落在某个 silence (region 之外) 区段内, 或
      - 距最近 speech_region 边界 <= tolerance 秒
    """
    sorted_regions = sorted(
        [(float(r["start"]), float(r["end"])) for r in speech_regions]
    )
    # 收集去掉音频起止的唯一切点
    eps = 0.01
    raw = []
    for s in segments:
        raw.append(float(s.start))
        raw.append(float(s.end))
    raw = [ts for ts in raw if eps < ts < audio_duration - eps]
    raw.sort()
    uniq: list[float] = []
    for ts in raw:
        if not uniq or ts - uniq[-1] > eps:
            uniq.append(ts)

    if not uniq:
        return 1.0

    aligned = 0
    for ts in uniq:
        in_silence = True
        nearest_dist = float("inf")
        for r_start, r_end in sorted_regions:
            if r_start < ts < r_end:
                in_silence = False
            for boundary in (r_start, r_end):
                d = abs(ts - boundary)
                if d < nearest_dist:
                    nearest_dist = d
        if in_silence or nearest_dist <= tolerance:
            aligned += 1
    return aligned / len(uniq)


# -------------------- silence_intervals_from_speech --------------------


class TestSilenceIntervalsFromSpeech:
    def test_empty_speech_yields_one_full_silence(self):
        out = silence_intervals_from_speech([], audio_duration=10.0)
        assert len(out) == 1
        assert out[0].start == 0.0
        assert out[0].end == 10.0

    def test_full_coverage_speech_yields_no_silence(self):
        out = silence_intervals_from_speech([_speech(0.0, 10.0)], audio_duration=10.0)
        assert out == []

    def test_normal_case_inverts_to_silences_between(self):
        # speech: [1,3], [5,8] → silence: [0,1], [3,5], [8,10]
        out = silence_intervals_from_speech(
            [_speech(1.0, 3.0), _speech(5.0, 8.0)],
            audio_duration=10.0,
        )
        assert len(out) == 3
        assert (out[0].start, out[0].end) == (0.0, 1.0)
        assert (out[1].start, out[1].end) == (3.0, 5.0)
        assert (out[2].start, out[2].end) == (8.0, 10.0)

    def test_unsorted_speech_is_sorted_internally(self):
        # 乱序输入也要给出 [0,1], [3,5], [8,10]
        out = silence_intervals_from_speech(
            [_speech(5.0, 8.0), _speech(1.0, 3.0)],
            audio_duration=10.0,
        )
        assert [(s.start, s.end) for s in out] == [(0.0, 1.0), (3.0, 5.0), (8.0, 10.0)]

    def test_overlapping_speech_collapses(self):
        # [1,5] 和 [3,6] 重叠 → speech 整体覆盖 [1,6], silence [0,1], [6,10]
        out = silence_intervals_from_speech(
            [_speech(1.0, 5.0), _speech(3.0, 6.0)],
            audio_duration=10.0,
        )
        assert [(s.start, s.end) for s in out] == [(0.0, 1.0), (6.0, 10.0)]

    def test_cursor_at_audio_end_no_trailing_silence(self):
        # speech 直到末尾 → 无尾部 silence
        out = silence_intervals_from_speech([_speech(0.0, 10.0)], audio_duration=10.0)
        assert out == []


# -------------------- SilenceInterval --------------------


class TestSilenceInterval:
    def test_mid(self):
        s = SilenceInterval(start=1.0, end=3.0)
        assert s.mid == 2.0

    def test_contains_boundary_inclusive(self):
        s = SilenceInterval(start=1.0, end=3.0)
        assert s.contains(1.0)
        assert s.contains(3.0)
        assert s.contains(2.0)
        assert not s.contains(0.99)
        assert not s.contains(3.01)


# -------------------- snap_segments_to_silence --------------------


class TestSnapSegmentsToSilence:
    def test_empty_segments_returns_empty(self):
        out, stats = snap_segments_to_silence([], [_speech(0, 10)], audio_duration=10.0)
        assert out == []
        assert stats["snapped_starts"] == 0
        assert stats["snapped_ends"] == 0
        assert stats["total_starts"] == 0
        assert stats["total_ends"] == 0

    def test_first_start_and_last_end_never_snapped(self):
        # 即使 silence 离 segments[0].start 很近, 也不动
        # silence 区间 [0, 1], [9, 10] (speech: [1,9])
        segs = [_seg(0.1, 4.0), _seg(4.0, 9.95)]
        out, stats = snap_segments_to_silence(
            segs, [_speech(1.0, 9.0)], audio_duration=10.0, tolerance=2.0
        )
        # n=2 → total_starts=1 (跳过 [0]), total_ends=1 (跳过 [-1])
        assert stats["total_starts"] == 1
        assert stats["total_ends"] == 1
        # 第一段 start (0.1) 没动
        assert out[0].start == 0.1
        # 最后段 end (9.95) 没动
        assert out[-1].end == 9.95

    def test_snap_to_nearest_silence_midpoint(self):
        # speech [0, 4], [5, 10] → silence [4, 5] mid=4.5
        # 切点必须落在 speech 内部 (不在 silence 边界上) 才会触发 snap
        # end[0]=3.8 在 speech [0,4] 内, 距 mid=4.5 是 0.7s, snap 到 4.5
        # start[1]=5.2 在 speech [5,10] 内, 距 mid=4.5 是 0.7s, snap 到 4.5
        segs = [_seg(0.0, 3.8), _seg(5.2, 10.0)]
        out, stats = snap_segments_to_silence(
            segs,
            [_speech(0, 4), _speech(5, 10)],
            audio_duration=10.0,
            tolerance=2.0,
        )
        assert out[0].end == 4.5
        assert out[1].start == 4.5
        assert stats["snapped_starts"] == 1
        assert stats["snapped_ends"] == 1

    def test_ts_in_silence_not_changed(self):
        # 切点已经落在 silence [4, 5] 内 (ts=4.6) → 不动
        segs = [_seg(0.0, 4.6), _seg(4.6, 10.0)]
        out, stats = snap_segments_to_silence(
            segs,
            [_speech(0, 4), _speech(5, 10)],
            audio_duration=10.0,
            tolerance=2.0,
        )
        assert out[0].end == 4.6  # 没动
        assert out[1].start == 4.6  # 没动
        assert stats["snapped_starts"] == 0
        assert stats["snapped_ends"] == 0

    def test_beyond_tolerance_not_snapped(self):
        # silence mid=4.5, 切点 7.0 距 mid 2.5s > tolerance=1.0
        segs = [_seg(0.0, 7.0), _seg(7.0, 10.0)]
        out, stats = snap_segments_to_silence(
            segs,
            [_speech(0, 4), _speech(5, 10)],
            audio_duration=10.0,
            tolerance=1.0,
        )
        # end[0]=7.0 距任何 silence mid 都 > 1s, 不动
        assert out[0].end == 7.0
        assert stats["snapped_ends"] == 0

    def test_zero_dur_protection_reverts_snap(self):
        # 构造: segments[1] start=4.0 end=4.1 (段时长 0.1)
        # speech [0,3], [3.6, 4.05], [4.1, 10] → silence [3, 3.6] mid=3.3, [4.05, 4.1] mid=4.075
        # start[1]=4.0 距最近 mid=4.075 仅 0.075s, tolerance 1.0 内 → snap 候选
        # 但 new_dur = 4.1 - 4.075 = 0.025 < min_dur=0.1 → 触发回退, start 保持 4.0
        segs = [_seg(0.0, 3.5), _seg(4.0, 4.1)]
        out, stats = snap_segments_to_silence(
            segs,
            [_speech(0, 3), _speech(3.6, 4.05), _speech(4.1, 10)],
            audio_duration=10.0,
            tolerance=1.0,
            min_segment_dur=0.1,
        )
        assert out[1].start == 4.0
        assert stats["skipped_zero_dur"] >= 1

    def test_diarize_overlap_preserved(self):
        # baseline 含 overlap: seg0 [0~4.5], seg1 [4.0~10] (重叠 0.5s)
        # silence [4.0~4.4] mid=4.2
        # end[0]=4.5 距 mid=0.3, snap 到 4.2 (4.2 < 4.0 段 start? 不: seg0 start=0, end snap 4.2 OK)
        # start[1]=4.0 已在 silence [4.0,4.4] 内 → 不动
        # 验证: 算法不强行同步 end[0] == start[1], overlap 不被破坏
        segs = [_seg(0.0, 4.5, speaker=0), _seg(4.0, 10.0, speaker=1)]
        out, stats = snap_segments_to_silence(
            segs,
            [_speech(0, 4), _speech(4.4, 10)],
            audio_duration=10.0,
            tolerance=1.0,
        )
        # start[1]=4.0 在 silence [4,4.4] 内 → 不动
        assert out[1].start == 4.0
        # end[0] snap 后 ≠ start[1] 也 OK (独立 snap)
        # 段时长都还 > 0
        for s in out:
            assert s.end > s.start

    def test_stats_total_counts(self):
        # 3 段 → total_starts=2 (跳过 [0]), total_ends=2 (跳过 [-1])
        segs = [_seg(0.0, 3.0), _seg(3.0, 6.0), _seg(6.0, 10.0)]
        _, stats = snap_segments_to_silence(
            segs, [_speech(0, 10)], audio_duration=10.0
        )
        assert stats["total_starts"] == 2
        assert stats["total_ends"] == 2

    def test_original_segments_not_mutated(self):
        # 切点 3.8 在 speech 内, 会被 snap 到 silence mid 4.5
        segs = [_seg(0.0, 3.8), _seg(5.2, 10.0)]
        original_end = segs[0].end
        out, _ = snap_segments_to_silence(
            segs,
            [_speech(0, 4), _speech(5, 10)],
            audio_duration=10.0,
            tolerance=2.0,
        )
        # 入参未被修改 (replace 拷贝)
        assert segs[0].end == original_end
        # 输出已变
        assert out[0].end != original_end


# -------------------- apply_silence_align_to_segments helper --------------------


class TestApplySilenceAlignHelper:
    def test_disabled_returns_input_unchanged(self):
        segs = [_seg(0.0, 4.0), _seg(4.0, 10.0)]
        out, stats = apply_silence_align_to_segments(
            segs, audio_path="/nonexistent.wav", audio_duration=10.0,
            qwen3_config=_cfg(enabled=False),
        )
        assert out is segs
        assert stats == {"enabled": False}

    def test_empty_segments_short_circuits(self):
        out, stats = apply_silence_align_to_segments(
            [], audio_path="/nonexistent.wav", audio_duration=10.0,
            qwen3_config=_cfg(enabled=True),
        )
        assert out == []
        assert stats["enabled"] is True
        assert stats.get("skipped") == "empty_segments"

    def test_ffmpeg_failure_falls_back_to_original(self):
        # 故意指 nonexistent 文件触发 ffmpeg 失败
        segs = [_seg(0.0, 4.0), _seg(4.0, 10.0)]
        out, stats = apply_silence_align_to_segments(
            segs, audio_path="/path/does/not/exist.wav", audio_duration=10.0,
            qwen3_config=_cfg(enabled=True),
        )
        # 失败时原样返回 + stats 含 error
        assert out is segs
        assert stats["enabled"] is True
        assert "error" in stats

    def test_mocked_ffmpeg_drives_snap(self):
        # mock ffmpeg_speech_regions 返回固定 speech, 验证 snap 真的跑了
        # 切点 3.8 / 5.2 都在 speech 内部, snap 到 silence mid=4.5
        segs = [_seg(0.0, 3.8), _seg(5.2, 10.0)]
        with patch(
            "src.core.qwen3_transcriber.ffmpeg_speech_regions",
            return_value=[_speech(0, 4), _speech(5, 10)],
        ):
            out, stats = apply_silence_align_to_segments(
                segs, audio_path="/any.wav", audio_duration=10.0,
                qwen3_config=_cfg(enabled=True, tolerance=2.0),
            )
        assert stats["enabled"] is True
        assert stats["snapped_starts"] + stats["snapped_ends"] >= 1
        assert stats["speech_regions_count"] == 2
        # end[0] 被吸到 silence mid=4.5
        assert out[0].end == 4.5


# -------------------- 端到端: snap 后 align_ratio 不降 --------------------


class TestEndToEndAlignRatio:
    """跑 baseline + snap, 验证 snap 后 align_ratio ≥ baseline."""

    def test_snap_improves_align_ratio_on_synthetic_case(self):
        # 合成 baseline: 段切点在 speech 区间正中 (典型 "切在话中间" 场景)
        # speech: [0, 4], [5, 9], [10, 15]
        # silence mid: 4.5, 9.5
        # baseline 段切点: end[0]=3.0 (落在 speech 内 + 距边界 1s), end[1]=7.0 (落在 speech 内 + 距 5/9 都 2s)
        segs = [
            _seg(0.0, 3.0, speaker=0),
            _seg(3.0, 7.0, speaker=1),
            _seg(7.0, 14.5, speaker=0),
        ]
        speech = [_speech(0, 4), _speech(5, 9), _speech(10, 15)]

        baseline_ratio = _align_ratio(segs, speech, audio_duration=15.0, tolerance=0.3)
        snapped, stats = snap_segments_to_silence(
            segs, speech, audio_duration=15.0, tolerance=2.0
        )
        snapped_ratio = _align_ratio(snapped, speech, audio_duration=15.0, tolerance=0.3)

        # 至少不降, 且应该有真正的吸附发生
        assert snapped_ratio >= baseline_ratio
        assert stats["snapped_starts"] + stats["snapped_ends"] > 0
