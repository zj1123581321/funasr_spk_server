"""
时间轴融合 — 把 ASR 全文按 speaker turn 时长比例线性切到各 turn

输入:
  - asr_text: ASR 输出的完整文本 (无 timestamp,因为 production aligner 模型不存在)
  - turns: [{start, end, speaker}, ...] 由 sherpa diarization 给出

策略 (PoC 第一版,粗放线性切分):
  1. 总有效说话时间 D_total = sum(turn.end - turn.start)
  2. 每秒文本字符数 c_per_sec = len(asr_text) / D_total
  3. 按 turn 顺序累计,turn_i 的文本起止字符位置 = round(累积秒数 × c_per_sec)

精度局限:
  - 假设 ASR 文本时间线性 — 实际语速不均时会偏
  - 不能处理静音段 / 插话 / overlap
  - 后续优化: 接 forced aligner 拿词级 timestamp 再融合,精度可以到秒以内

输出:
  - segments: [{start, end, speaker, text}, ...]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    speaker: int
    text: str


def filter_spurious_speakers(turns: List[dict], min_speaker_total: float = 2.0) -> List[dict]:
    """合并"假说话人" — 某 speaker 总时长 < 阈值则归到时间最近的另一 speaker.

    这是 sherpa-onnx 在 cluster_threshold=0.9 + int8 segmentation 下会出的工件:
    偶尔切出 0.3s 短 turn 被聚成独立 speaker. 总时长很小 = 噪声片段.
    """
    if not turns:
        return turns
    totals: dict = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0) + (t["end"] - t["start"])
    spurious = {sp for sp, dur in totals.items() if dur < min_speaker_total}
    if not spurious:
        return turns

    valid_speakers = sorted(totals.keys() - spurious, key=lambda s: -totals[s])
    if not valid_speakers:
        return turns  # 全是噪声,什么都不改

    out = []
    for t in turns:
        if t["speaker"] in spurious:
            # 找时间最近的 valid speaker 邻居
            mid = (t["start"] + t["end"]) / 2
            best = valid_speakers[0]
            best_dist = float("inf")
            for ot in turns:
                if ot["speaker"] not in spurious and ot is not t:
                    omid = (ot["start"] + ot["end"]) / 2
                    d = abs(omid - mid)
                    if d < best_dist:
                        best_dist = d
                        best = ot["speaker"]
            out.append({**t, "speaker": best})
        else:
            out.append(t)
    return out


def merge_asr_and_diarize(asr_text: str, turns: List[dict]) -> List[Segment]:
    """按 turn 时长比例线性切分 ASR 文本."""
    if not turns or not asr_text:
        return []

    # 按 start 时间排序 (sherpa 已经排好,这里防御)
    turns_sorted = sorted(turns, key=lambda t: t["start"])

    durations = [(t["end"] - t["start"]) for t in turns_sorted]
    total_dur = sum(durations)
    if total_dur <= 0:
        return []

    total_chars = len(asr_text)
    c_per_sec = total_chars / total_dur

    segments: List[Segment] = []
    cumulative = 0.0
    prev_char_pos = 0

    for i, t in enumerate(turns_sorted):
        cumulative += durations[i]
        # 最后一个 turn 直接吃完剩余文本(避免浮点截断丢字)
        if i == len(turns_sorted) - 1:
            char_end = total_chars
        else:
            char_end = int(round(cumulative * c_per_sec))
            char_end = min(max(char_end, prev_char_pos), total_chars)

        chunk = asr_text[prev_char_pos:char_end]
        segments.append(
            Segment(
                start=float(t["start"]),
                end=float(t["end"]),
                speaker=int(t["speaker"]),
                text=chunk,
            )
        )
        prev_char_pos = char_end

    return segments


def segments_to_srt(segments: List[Segment]) -> str:
    """把 segments 转 SRT 字幕格式."""
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        if not seg.text.strip():
            continue
        lines.append(str(i))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        lines.append(f"Speaker{seg.speaker + 1}: {seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def segments_to_rttm(segments: List[Segment], file_id: str = "audio") -> str:
    """把 segments 转 RTTM (diarization 通用评估格式)."""
    lines: List[str] = []
    for seg in segments:
        dur = seg.end - seg.start
        lines.append(
            f"SPEAKER {file_id} 1 {seg.start:.3f} {dur:.3f} <NA> <NA> Speaker{seg.speaker + 1} <NA> <NA>"
        )
    return "\n".join(lines)
