from __future__ import annotations

from src.core.qwen3_segment import (
    SpeechRegion,
    clip_turns_to_window,
    detect_repetition_warnings,
    detect_segment_similarity_warnings,
    normalize_speech_regions,
    plan_macro_segments,
)
from src.core.qwen3.merge import merge_asr_chunks_and_diarize
from src.core.qwen3_postprocess import apply_tech_podcast_glossary


def test_plan_defaults_use_quality_first_12min_segments():
    segments = plan_macro_segments(audio_duration=83 * 60)

    assert segments[0].start == 0.0
    assert segments[-1].end == 83 * 60
    assert all(seg.duration <= 15 * 60 for seg in segments)
    assert [seg.duration for seg in segments[:3]] == [12 * 60, 12 * 60, 12 * 60]


def test_plan_prefers_silence_boundary_near_target():
    # Speech regions leave a 704-716s silence gap around the 12min target.
    regions = [
        {"start": 0.0, "end": 704.0},
        {"start": 716.0, "end": 1800.0},
    ]

    segments = plan_macro_segments(
        audio_duration=1800.0,
        speech_regions=regions,
        target_segment_sec=720.0,
        soft_max_segment_sec=900.0,
        hard_max_segment_sec=1200.0,
        min_segment_sec=180.0,
        min_silence_sec=1.0,
    )

    assert segments[0].end == 710.0
    assert segments[1].start == 710.0


def test_plan_uses_target_when_no_silence_boundary_exists():
    regions = [{"start": 0.0, "end": 2400.0}]

    segments = plan_macro_segments(
        audio_duration=2400.0,
        speech_regions=regions,
        target_segment_sec=720.0,
        soft_max_segment_sec=900.0,
        hard_max_segment_sec=1200.0,
        min_segment_sec=180.0,
    )

    assert segments[0].end == 720.0
    assert all(seg.duration <= 900.0 for seg in segments)


def test_plan_adds_asr_overlap_without_moving_output_boundary():
    segments = plan_macro_segments(audio_duration=1800.0, overlap_sec=10.0)

    assert segments[0].asr_start == 0.0
    assert segments[1].start == 720.0
    assert segments[1].asr_start == 710.0
    assert segments[1].asr_end == segments[1].end


def test_normalize_speech_regions_clamps_sorts_and_merges():
    regions = normalize_speech_regions(
        [
            {"start": 4.0, "end": 6.0},
            {"start": -1.0, "end": 2.0},
            SpeechRegion(start=2.2, end=3.0),
        ],
        audio_duration=5.0,
        merge_gap_sec=0.3,
    )

    assert regions == [
        SpeechRegion(start=0.0, end=3.0),
        SpeechRegion(start=4.0, end=5.0),
    ]


def test_clip_turns_to_window_returns_relative_turns():
    turns = [
        {"start": 10.0, "end": 20.0, "speaker": 0},
        {"start": 25.0, "end": 40.0, "speaker": 1},
    ]

    clipped = clip_turns_to_window(turns, window_start=18.0, window_end=30.0)

    assert clipped == [
        {"start": 0.0, "end": 2.0, "speaker": 0},
        {"start": 7.0, "end": 12.0, "speaker": 1},
    ]


def test_detect_repetition_warnings_finds_token_loop():
    warnings = detect_repetition_warnings("我这个AI我这个AI我这个AI我这个AI我这个AI我这个AI")

    assert warnings
    assert warnings[0]["type"] == "repeated_phrase"


def test_detect_segment_similarity_warnings_finds_repeated_segments():
    warnings = detect_segment_similarity_warnings(
        ["我们关注这个能力"] * 4,
        similarity_threshold=0.95,
        min_run=4,
    )

    assert warnings == [
        {
            "type": "similar_segment_run",
            "segment_start": 0,
            "segment_end": 3,
            "similarity": 1.0,
        }
    ]


def test_merge_asr_chunks_and_diarize_limits_text_drift_to_chunk():
    class Chunk:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    chunks = [
        Chunk("主持人提问？嘉宾回答。", 0.0, 10.0),
        Chunk("嘉宾继续解释。", 10.0, 20.0),
    ]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": 1},
        {"start": 4.0, "end": 20.0, "speaker": 0},
    ]

    merged = merge_asr_chunks_and_diarize(chunks, turns)

    assert [m.speaker for m in merged] == [1, 0, 0]
    assert merged[0].text == "主持人提问？"
    assert "".join(m.text for m in merged) == "主持人提问？嘉宾回答。嘉宾继续解释。"


def test_merge_asr_chunks_keeps_short_turn_from_stealing_sentence():
    class Chunk:
        def __init__(self, text, start, end):
            self.text = text
            self.start = start
            self.end = end

    chunks = [Chunk("这是一整句话。后面内容继续。", 0.0, 10.0)]
    turns = [
        {"start": 0.0, "end": 0.5, "speaker": 1},
        {"start": 0.5, "end": 10.0, "speaker": 0},
    ]

    merged = merge_asr_chunks_and_diarize(chunks, turns)

    assert [m.speaker for m in merged] == [1, 0]
    assert len(merged[0].text) <= 3
    assert "一整句话" in merged[1].text
    assert "".join(m.text for m in merged) == "这是一整句话。后面内容继续。"


def test_apply_tech_podcast_glossary_corrects_common_terms():
    text = "OpenCLow 和 Cloud Code 里开了 C L I，张鱼也提到了且微信。"

    assert apply_tech_podcast_glossary(text) == (
        "OpenClaw 和 Claude Code 里开了 API，章鱼也提到了企业微信。"
    )
