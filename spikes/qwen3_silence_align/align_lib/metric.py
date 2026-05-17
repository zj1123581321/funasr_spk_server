"""自动 metric: 段切点对齐质量。

定义:
  - 段切点 = 每个 segment 的 start 和 end 时间戳
  - speech_regions = ffmpeg silencedetect 得到的"说话"区间 list[{start, end}]
  - silence_regions = audio_duration - speech_regions
  - 一个切点"对齐到 silence" = 切点落在 silence 区域内, 或距离最近的 silence 边界 <= tolerance(默认 0.3s)
    - "在 silence 内"语义: 段切换发生在静音处, 听感自然
    - "距离 silence 边界近"语义: 段切换发生在话语开始/结束的瞬间, 也可接受

输出:
  - align_ratio: 对齐切点数 / 总切点数 (0~1)
  - per_segment_detail: 每个切点的 (timestamp, type, aligned, dist_to_silence)

注意:
  - 段的 start 切点 = "上一段结束 / 本段开始" 边界
  - 段的 end 切点 = "本段结束 / 下一段开始" 边界
  - 相邻 segment 的 end[i] == start[i+1] 时算同一个切点, 去重统计
  - 音频起点 0.0 和终点 audio_duration 不算切点(音频边界天然对齐)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SpeechRegion:
    start: float
    end: float


@dataclass
class BoundaryDetail:
    """单个段切点的对齐细节."""
    timestamp: float
    in_silence: bool
    dist_to_nearest_silence_boundary: float
    nearest_speech_region_idx: Optional[int]


@dataclass
class AlignmentMetric:
    total_boundaries: int
    aligned_boundaries: int
    align_ratio: float
    in_silence_count: int
    near_boundary_count: int
    details: List[BoundaryDetail]


def _normalize_regions(speech_regions: List[dict]) -> List[SpeechRegion]:
    """把 ffmpeg_speech_regions 输出的 dict list 转 SpeechRegion list, 按 start 排序."""
    out = [SpeechRegion(start=float(r["start"]), end=float(r["end"])) for r in speech_regions]
    out.sort(key=lambda r: r.start)
    return out


def _classify_boundary(
    ts: float,
    regions: List[SpeechRegion],
    tolerance: float,
) -> BoundaryDetail:
    """判断时间戳 ts 是否对齐到 silence.

    判定规则:
      1. 若 ts 落在任何 speech_region 的 (start, end) 严格区间内 → 不在 silence (in_silence=False)
         同时检查最近边界距离, 若 <= tolerance → 仍算"对齐"(near_boundary)
      2. 若 ts 落在 region 之间(silence)或外部 → in_silence=True (天然对齐)
    """
    # 找最近的 region 边界距离
    nearest_dist = float("inf")
    nearest_idx = None
    in_silence = True
    for idx, r in enumerate(regions):
        if r.start < ts < r.end:
            in_silence = False
        # ts 到该 region 左右边界的最近距离
        for boundary_ts in (r.start, r.end):
            d = abs(ts - boundary_ts)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = idx
    return BoundaryDetail(
        timestamp=ts,
        in_silence=in_silence,
        dist_to_nearest_silence_boundary=nearest_dist if nearest_dist != float("inf") else 0.0,
        nearest_speech_region_idx=nearest_idx,
    )


def _collect_unique_boundaries(
    segments: List[dict],
    audio_duration: float,
    epsilon: float = 0.01,
) -> List[float]:
    """收集 segment 的所有切点(去重 + 去掉音频起止).

    segment dict 至少含 'start' 和 'end' 字段(单位秒).
    """
    raw: List[float] = []
    for seg in segments:
        raw.append(float(seg["start"]))
        raw.append(float(seg["end"]))
    raw = [ts for ts in raw if epsilon < ts < audio_duration - epsilon]
    # 去重: 相邻 segment 共享同一切点, 排序后用 epsilon 合并
    raw.sort()
    uniq: List[float] = []
    for ts in raw:
        if not uniq or ts - uniq[-1] > epsilon:
            uniq.append(ts)
    return uniq


def evaluate_alignment(
    segments: List[dict],
    speech_regions: List[dict],
    audio_duration: float,
    tolerance: float = 0.3,
) -> AlignmentMetric:
    """计算段切点对齐 silence 的比例.

    Args:
      segments: list[dict], 每项至少含 'start' 和 'end' 字段(秒).
      speech_regions: ffmpeg_speech_regions 输出, list[{'start', 'end'}].
      audio_duration: 音频总时长(秒), 用于过滤音频边界切点.
      tolerance: 切点距 silence 边界 <= tolerance 也算对齐(秒).

    Returns:
      AlignmentMetric.
    """
    regions = _normalize_regions(speech_regions)
    boundaries = _collect_unique_boundaries(segments, audio_duration)
    details: List[BoundaryDetail] = []
    in_silence_count = 0
    near_boundary_count = 0
    for ts in boundaries:
        d = _classify_boundary(ts, regions, tolerance)
        details.append(d)
        if d.in_silence:
            in_silence_count += 1
        elif d.dist_to_nearest_silence_boundary <= tolerance:
            near_boundary_count += 1

    aligned = in_silence_count + near_boundary_count
    total = len(boundaries)
    ratio = aligned / total if total > 0 else 1.0
    return AlignmentMetric(
        total_boundaries=total,
        aligned_boundaries=aligned,
        align_ratio=ratio,
        in_silence_count=in_silence_count,
        near_boundary_count=near_boundary_count,
        details=details,
    )


def format_metric_report(m: AlignmentMetric, name: str = "metric") -> str:
    """人读的简单格式化输出."""
    lines = [
        f"=== {name} ===",
        f"  total_boundaries:   {m.total_boundaries}",
        f"  aligned_boundaries: {m.aligned_boundaries}",
        f"  align_ratio:        {m.align_ratio:.4f}  ({m.align_ratio * 100:.2f}%)",
        f"    in_silence:       {m.in_silence_count}",
        f"    near_boundary:    {m.near_boundary_count}",
    ]
    return "\n".join(lines)
