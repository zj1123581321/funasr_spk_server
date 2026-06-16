"""Long-audio helpers for Qwen3 ASR PoC.

The vendor Qwen3 engine already decodes audio in short internal chunks.  This
module plans a higher-level "macro segment" layer so long recordings can reset
ASR session state before quality drifts, while still preferring VAD/silence
boundaries for natural cuts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class MacroSegment:
    idx: int
    start: float
    end: float
    asr_start: float
    asr_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def asr_duration(self) -> float:
        return max(0.0, self.asr_end - self.asr_start)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_speech_regions(
    regions: Optional[Iterable[dict | SpeechRegion]],
    audio_duration: float,
    merge_gap_sec: float = 0.5,
    min_region_sec: float = 0.1,
) -> List[SpeechRegion]:
    """Clamp, sort, and merge speech regions.

    Accepts either ``SpeechRegion`` objects or dicts with ``start``/``end``.
    """
    if not regions:
        return []

    out: List[SpeechRegion] = []
    for region in regions:
        if isinstance(region, SpeechRegion):
            start, end = region.start, region.end
        else:
            start, end = float(region["start"]), float(region["end"])
        start = max(0.0, min(float(start), audio_duration))
        end = max(0.0, min(float(end), audio_duration))
        if end - start >= min_region_sec:
            out.append(SpeechRegion(start=start, end=end))

    out.sort(key=lambda r: r.start)
    merged: List[SpeechRegion] = []
    for region in out:
        if not merged or region.start - merged[-1].end > merge_gap_sec:
            merged.append(region)
        else:
            prev = merged[-1]
            merged[-1] = SpeechRegion(start=prev.start, end=max(prev.end, region.end))
    return merged


def _silence_boundaries(
    speech_regions: List[SpeechRegion],
    audio_duration: float,
    min_silence_sec: float,
) -> List[tuple[float, float, float]]:
    """Return candidate cut points as ``(boundary, gap_start, gap_end)``."""
    if audio_duration <= 0:
        return []
    if not speech_regions:
        return []

    gaps: List[tuple[float, float]] = []
    cursor = 0.0
    for region in speech_regions:
        if region.start - cursor >= min_silence_sec:
            gaps.append((cursor, region.start))
        cursor = max(cursor, region.end)
    if audio_duration - cursor >= min_silence_sec:
        gaps.append((cursor, audio_duration))

    return [((start + end) / 2.0, start, end) for start, end in gaps]


def _choose_boundary(
    segment_start: float,
    audio_duration: float,
    speech_regions: List[SpeechRegion],
    target_segment_sec: float,
    soft_max_segment_sec: float,
    hard_max_segment_sec: float,
    min_segment_sec: float,
    boundary_search_sec: float,
    min_silence_sec: float,
) -> float:
    remaining = audio_duration - segment_start
    if remaining <= soft_max_segment_sec:
        return audio_duration

    target = min(audio_duration, segment_start + target_segment_sec)
    min_end = min(audio_duration, segment_start + min_segment_sec)
    soft_end = min(audio_duration, segment_start + soft_max_segment_sec)
    hard_end = min(audio_duration, segment_start + hard_max_segment_sec)

    boundaries = _silence_boundaries(speech_regions, audio_duration, min_silence_sec)
    search_start = max(min_end, target - boundary_search_sec)
    search_end = min(soft_end, target + boundary_search_sec)

    candidates = [
        (boundary, gap_start, gap_end)
        for boundary, gap_start, gap_end in boundaries
        if search_start <= boundary <= search_end
    ]
    if candidates:
        boundary, _gap_start, _gap_end = min(
            candidates,
            key=lambda item: (abs(item[0] - target), -(item[2] - item[1])),
        )
        return boundary

    fallback = [
        (boundary, gap_start, gap_end)
        for boundary, gap_start, gap_end in boundaries
        if min_end <= boundary <= hard_end
    ]
    if fallback:
        boundary, _gap_start, _gap_end = min(
            fallback,
            key=lambda item: (abs(item[0] - target), -(item[2] - item[1])),
        )
        return boundary

    return min(target, soft_end, hard_end)


def plan_macro_segments(
    audio_duration: float,
    speech_regions: Optional[Iterable[dict | SpeechRegion]] = None,
    target_segment_sec: float = 12 * 60,
    soft_max_segment_sec: float = 15 * 60,
    hard_max_segment_sec: float = 20 * 60,
    min_segment_sec: float = 3 * 60,
    overlap_sec: float = 0.0,
    boundary_search_sec: float = 60.0,
    min_silence_sec: float = 0.8,
) -> List[MacroSegment]:
    """Plan quality-first macro segments for long Qwen3 ASR runs.

    Defaults intentionally prefer 12-minute segments with a 15-minute soft max.
    20 minutes is treated as an emergency hard cap, not the normal target.
    """
    if audio_duration <= 0:
        return []
    if min_segment_sec <= 0:
        raise ValueError("min_segment_sec must be > 0")
    if not (min_segment_sec <= target_segment_sec <= soft_max_segment_sec <= hard_max_segment_sec):
        raise ValueError(
            "Expected min_segment_sec <= target_segment_sec <= "
            "soft_max_segment_sec <= hard_max_segment_sec"
        )
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")

    regions = normalize_speech_regions(speech_regions, audio_duration)
    segments: List[MacroSegment] = []
    start = 0.0

    while start < audio_duration - 1e-6:
        end = _choose_boundary(
            segment_start=start,
            audio_duration=audio_duration,
            speech_regions=regions,
            target_segment_sec=target_segment_sec,
            soft_max_segment_sec=soft_max_segment_sec,
            hard_max_segment_sec=hard_max_segment_sec,
            min_segment_sec=min_segment_sec,
            boundary_search_sec=boundary_search_sec,
            min_silence_sec=min_silence_sec,
        )
        if end <= start:
            end = min(audio_duration, start + hard_max_segment_sec)

        # Avoid a tiny tail segment by merging it into the previous segment.
        if audio_duration - end < min_segment_sec and end < audio_duration:
            end = audio_duration

        idx = len(segments)
        asr_start = max(0.0, start - overlap_sec) if idx > 0 else start
        segments.append(
            MacroSegment(
                idx=idx,
                start=round(start, 3),
                end=round(end, 3),
                asr_start=round(asr_start, 3),
                asr_end=round(end, 3),
            )
        )
        start = end

    return segments


def clip_turns_to_window(
    turns: Iterable[dict],
    window_start: float,
    window_end: float,
    relative: bool = True,
    min_turn_sec: float = 0.05,
) -> List[dict]:
    """Clip diarization turns to an ASR window."""
    clipped: List[dict] = []
    for turn in turns:
        start = max(float(turn["start"]), window_start)
        end = min(float(turn["end"]), window_end)
        if end - start < min_turn_sec:
            continue
        if relative:
            out_start = start - window_start
            out_end = end - window_start
        else:
            out_start = start
            out_end = end
        clipped.append(
            {
                **turn,
                "start": round(out_start, 3),
                "end": round(out_end, 3),
                "speaker": int(turn["speaker"]),
            }
        )
    return sorted(clipped, key=lambda t: t["start"])


def detect_repetition_warnings(
    text: str,
    min_phrase_chars: int = 2,
    max_phrase_chars: int = 12,
    min_repeats: int = 6,
) -> List[dict]:
    """Detect obvious repeated-token/text loops in ASR output."""
    warnings: List[dict] = []
    compact = "".join(text.split())
    if not compact:
        return warnings

    for size in range(min_phrase_chars, max_phrase_chars + 1):
        for start in range(0, max(0, len(compact) - size * min_repeats + 1)):
            phrase = compact[start : start + size]
            repeats = 1
            cursor = start + size
            while compact[cursor : cursor + size] == phrase:
                repeats += 1
                cursor += size
            if repeats >= min_repeats:
                warnings.append(
                    {
                        "type": "repeated_phrase",
                        "phrase": phrase,
                        "repeats": repeats,
                        "char_start": start,
                    }
                )
                return warnings
    return warnings


def detect_segment_similarity_warnings(
    segment_texts: Iterable[str],
    similarity_threshold: float = 0.95,
    min_run: int = 4,
) -> List[dict]:
    """Detect consecutive near-identical final segments."""
    texts = [t.strip() for t in segment_texts if t and t.strip()]
    if len(texts) < min_run:
        return []

    run_start = 0
    for idx in range(1, len(texts)):
        sim = SequenceMatcher(None, texts[idx - 1], texts[idx]).ratio()
        if sim >= similarity_threshold:
            if idx - run_start + 1 >= min_run:
                return [
                    {
                        "type": "similar_segment_run",
                        "segment_start": run_start,
                        "segment_end": idx,
                        "similarity": round(sim, 4),
                    }
                ]
        else:
            run_start = idx
    return []
