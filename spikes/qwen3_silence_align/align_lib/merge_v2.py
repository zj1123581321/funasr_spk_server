"""Silence-aware 段切点对齐 — merge_v2.

策略: 不重写 production merge 算法, 只在它输出后做 snap-to-silence:
  1. 跑现有 merge_asr_chunks_and_diarize 拿到 segments
  2. 对每个相邻 segment 的边界 (segment[i].end == segment[i+1].start), 找最近 silence
  3. 在容忍窗口 tolerance_sec 内, 把边界吸附到 silence 的中点
     - 优先级 1: 如果 ts 已经在某个 silence 区间内, 不动
     - 优先级 2: 否则找最近 silence 中点, 距离 <= tolerance 才动

设计考量:
  - 改造最小, 保留 production 逻辑的稳定性 (回退只需关 flag)
  - 文字不重切, 仅时间戳吸附 — trade-off: 段时长可能变化 ±tolerance, 但听感"切在静音"更重要
  - 第一段 start 和最后一段 end 不吸附 (音频起止天然对齐)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def contains(self, ts: float) -> bool:
        return self.start <= ts <= self.end


def silence_intervals_from_speech(
    speech_regions: List[dict],
    audio_duration: float,
) -> List[SilenceInterval]:
    """speech_regions → silence_intervals (相邻 speech 之间的间隙 + 音频起止外的间隙).

    speech_regions: ffmpeg_speech_regions 输出, list[{'start', 'end'}], 按 start 排序.
    """
    sorted_regions = sorted(
        [{"start": float(r["start"]), "end": float(r["end"])} for r in speech_regions],
        key=lambda r: r["start"],
    )
    silences: List[SilenceInterval] = []
    cursor = 0.0
    for r in sorted_regions:
        if r["start"] > cursor:
            silences.append(SilenceInterval(start=cursor, end=r["start"]))
        cursor = max(cursor, r["end"])
    if cursor < audio_duration:
        silences.append(SilenceInterval(start=cursor, end=audio_duration))
    return silences


def find_snap_target(
    ts: float,
    silences: List[SilenceInterval],
    tolerance: float,
) -> float | None:
    """找 ts 应该吸附到的目标时间戳, None 表示不动.

    规则:
      1. ts 已在某个 silence 内 → 不动 (返回 ts)
      2. ts 不在 silence 内 → 找最近 silence 中点, 距离 <= tolerance 才返回
    """
    for s in silences:
        if s.contains(ts):
            return ts
    best_target = None
    best_dist = tolerance
    for s in silences:
        d = abs(ts - s.mid)
        if d < best_dist:
            best_dist = d
            best_target = s.mid
    return best_target


def snap_segments_to_silence(
    segments: List,
    speech_regions: List[dict],
    audio_duration: float,
    tolerance: float = 2.0,
    min_segment_dur: float = 0.1,
) -> tuple[List, dict]:
    """对 segments 做 snap-to-silence (独立吸附 start 和 end).

    设计:
      - 每个 segment 的 start 和 end 独立 snap 到最近 silence (容忍 tolerance 内)
      - 不强求 end[i] == start[i+1]: baseline 的 diarize overlap (插话场景) 必须保留,
        否则会触发 0 时长段
      - snap 后段时长 < min_segment_dur 时回退该 snap, 保护段时长

    Args:
      segments: production merge 输出的 Segment list.
      speech_regions: ffmpeg_speech_regions 输出.
      audio_duration: 音频总时长.
      tolerance: 吸附容忍窗口 (秒). 距离 > tolerance 不动.
      min_segment_dur: snap 后段时长 < 此值则回退, 保护段不退化.

    Returns:
      (new_segments, stats).
    """
    if not segments:
        return [], {
            "snapped_starts": 0, "snapped_ends": 0,
            "total_starts": 0, "total_ends": 0,
            "skipped_zero_dur": 0,
        }

    silences = silence_intervals_from_speech(speech_regions, audio_duration)
    new_segs = [replace(s) for s in segments]
    snapped_starts = 0
    snapped_ends = 0
    skipped_zero_dur = 0
    n = len(new_segs)
    # 第一个 segment 的 start 和最后一个 segment 的 end 不 snap (音频起止)
    total_starts = max(0, n - 1)  # 除了 segments[0].start
    total_ends = max(0, n - 1)    # 除了 segments[-1].end

    for i, s in enumerate(new_segs):
        # snap start (跳过第一段)
        if i > 0:
            target = find_snap_target(s.start, silences, tolerance)
            if target is not None and target != s.start:
                new_dur = s.end - target
                if new_dur >= min_segment_dur:
                    s.start = round(target, 3)
                    snapped_starts += 1
                else:
                    skipped_zero_dur += 1
        # snap end (跳过最后一段)
        if i < n - 1:
            target = find_snap_target(s.end, silences, tolerance)
            if target is not None and target != s.end:
                new_dur = target - s.start
                if new_dur >= min_segment_dur:
                    s.end = round(target, 3)
                    snapped_ends += 1
                else:
                    skipped_zero_dur += 1

    return new_segs, {
        "snapped_starts": snapped_starts,
        "snapped_ends": snapped_ends,
        "total_starts": total_starts,
        "total_ends": total_ends,
        "skipped_zero_dur": skipped_zero_dur,
        "tolerance_sec": tolerance,
        "min_segment_dur_sec": min_segment_dur,
        "silence_intervals_count": len(silences),
    }


def merge_v2(
    chunks: Iterable,
    turns: List[dict],
    speech_regions: List[dict],
    audio_duration: float,
    tolerance: float = 2.0,
    min_segment_dur: float = 0.1,
) -> tuple[List, dict]:
    """跑 production merge + snap-to-silence.

    便利包装, 一行调完整流程.
    """
    from src.core.qwen3.merge import merge_asr_chunks_and_diarize
    segments = merge_asr_chunks_and_diarize(chunks, turns)
    return snap_segments_to_silence(
        segments, speech_regions, audio_duration, tolerance, min_segment_dur
    )
