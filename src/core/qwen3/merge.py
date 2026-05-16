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

本文件移植自 spikes/qwen3_diarize/src/merge.py, 纯函数, 无侧依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Segment:
    """融合后的转录片段(内部表示, 不直接暴露到 API 层)"""
    start: float
    end: float
    speaker: int
    text: str


def filter_spurious_speakers(
    turns: List[dict],
    min_speaker_total: float = 2.0,
    min_speaker_share: float = 0.01,
    audio_duration: float = 0.0,
) -> List[dict]:
    """合并"假说话人" — 某 speaker 总时长太小则归到时间最近的另一 speaker.

    阈值组合(满足任一即视为 spurious):
      - 绝对时长 < min_speaker_total (默认 2s)
      - 占音频时长比例 < min_speaker_share (默认 1%; 仅当 audio_duration > 0 才启用)

    场景:
      - 2 人 5min(300s) 音频: 1% = 3s,刚好滤掉<3s 的噪声碎片
      - 4 人 8min(480s) 音频: 1% = 4.8s,滤掉 NeMo+0.9 产生的 5 个 <3% spurious cluster
    """
    if not turns:
        return turns
    totals: dict = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0) + (t["end"] - t["start"])

    abs_threshold = min_speaker_total
    pct_threshold = min_speaker_share * audio_duration if audio_duration > 0 else 0.0
    threshold = max(abs_threshold, pct_threshold)
    spurious = {sp for sp, dur in totals.items() if dur < threshold}
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
    """按 turn 时长比例线性切分 ASR 文本.

    边界条件:
      - turns 为空 / asr_text 为空 → 返回 []
      - 总时长 <= 0 → 返回 []
      - 最后一个 turn 直接吃完剩余文本, 避免浮点截断丢字
    """
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


def _choose_text_boundary(text: str, target: int, search: int = 18) -> int:
    """Choose a split point near target, preferring sentence punctuation."""
    if target <= 0:
        return 0
    if target >= len(text):
        return len(text)

    punct = "。？！；!?;"
    lo = max(1, target - search)
    hi = min(len(text) - 1, target + search)
    candidates = []
    for idx in range(lo, hi + 1):
        if text[idx - 1] in punct:
            candidates.append(idx)
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - target))
    return target


def _split_text_by_weights(text: str, weights: List[float]) -> List[str]:
    """Split text into len(weights) chunks while keeping punctuation intact."""
    if not weights:
        return []
    if len(weights) == 1:
        return [text]

    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        weights = [1.0] * len(weights)
        total = float(len(weights))

    chunks: List[str] = []
    cursor = 0
    acc = 0.0
    for i, weight in enumerate(weights[:-1]):
        acc += max(0.0, weight)
        target = int(round(len(text) * acc / total))
        target = max(cursor, target)
        expected_piece_chars = max(0, target - cursor)
        # A fixed wide punctuation search lets very short diarization turns
        # steal a whole sentence.  Scale the search window with the piece size.
        boundary_search = min(18, max(2, int(round(expected_piece_chars * 0.5))))
        boundary = _choose_text_boundary(text, target, search=boundary_search)
        chunks.append(text[cursor:boundary])
        cursor = boundary
    chunks.append(text[cursor:])
    return chunks


def merge_asr_chunks_and_diarize(chunks: Iterable, turns: List[dict]) -> List[Segment]:
    """Merge ASR chunk text with diarization turns using chunk time windows.

    This is a stricter version of ``merge_asr_and_diarize`` for Qwen3 long
    audio.  Qwen3-ASR exposes rough 40s internal chunk boundaries even without
    a forced aligner.  Splitting text inside each 40s chunk bounds drift much
    better than linearly cutting a whole 12min macro window.
    """
    chunks_sorted = sorted(
        [c for c in chunks if getattr(c, "text", "")],
        key=lambda c: (float(getattr(c, "start", 0.0)), float(getattr(c, "end", 0.0))),
    )
    if not chunks_sorted or not turns:
        return []

    turns_sorted = sorted(turns, key=lambda t: t["start"])
    segments: List[Segment] = []

    for chunk in chunks_sorted:
        chunk_start = float(chunk.start)
        chunk_end = float(chunk.end)
        text = str(chunk.text)
        if chunk_end <= chunk_start or not text:
            continue

        overlaps = []
        for turn in turns_sorted:
            start = max(chunk_start, float(turn["start"]))
            end = min(chunk_end, float(turn["end"]))
            if end - start > 0.03:
                overlaps.append((start, end, int(turn["speaker"])))

        if not overlaps:
            mid = (chunk_start + chunk_end) / 2.0
            nearest = min(
                turns_sorted,
                key=lambda t: min(abs(float(t["start"]) - mid), abs(float(t["end"]) - mid)),
            )
            overlaps = [(chunk_start, chunk_end, int(nearest["speaker"]))]

        pieces = _split_text_by_weights(text, [end - start for start, end, _sp in overlaps])
        for (start, end, speaker), piece in zip(overlaps, pieces):
            if not piece:
                continue
            segments.append(
                Segment(
                    start=round(start, 2),
                    end=round(end, 2),
                    speaker=speaker,
                    text=piece,
                )
            )

    return segments


def segments_to_srt(segments: List[Segment]) -> str:
    """把 segments 转 SRT 字幕格式 (跳过空文本片段, 索引按非空 segment 重新编号)."""
    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    idx = 0
    for seg in segments:
        if not seg.text.strip():
            continue
        idx += 1
        lines.append(str(idx))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        # 与 FunASR SRT 字节级对齐: 冒号后无空格(funasr_transcriber.py:516)
        lines.append(f"Speaker{seg.speaker + 1}:{seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)
