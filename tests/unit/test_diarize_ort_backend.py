"""OrtCudaDiarizeBackend 串联 + diarize dispatch 单测.

覆盖:
1. _extract_turns_from_activity: (T_frame, S) binary → turn list, on/off filter
2. run_diarization_ort_cuda: 端到端串联 pyannote sliding window + TitaNet embedding + cluster
3. run_diarization_dispatched: backend 路由 (sherpa 默认 / ort_cuda 强制切)

ORT session 全 mock, 不真实加载. Integration test (commit 9) 在远端真跑.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ==================== _extract_turns_from_activity ====================


def test_extract_turns_from_all_silence_returns_empty():
    from src.core.qwen3.diarize_ort import _extract_turns_from_activity

    activity = np.zeros((1000, 3), dtype=np.int8)
    turns = _extract_turns_from_activity(activity)
    assert turns == []


def test_extract_turns_single_continuous_speaker_returns_one_turn():
    """spk0 在 frame [100:500] 活动 → 1 turn 跨该区间."""
    from src.core.qwen3.diarize_ort import (
        PYANNOTE_FRAME_RATE_HZ,
        _extract_turns_from_activity,
    )

    activity = np.zeros((1000, 3), dtype=np.int8)
    activity[100:500, 0] = 1
    turns = _extract_turns_from_activity(activity, min_duration_on=0.1)
    assert len(turns) == 1
    assert turns[0]["local_speaker"] == 0
    np.testing.assert_allclose(turns[0]["start"], 100 / PYANNOTE_FRAME_RATE_HZ, atol=1e-3)
    np.testing.assert_allclose(turns[0]["end"], 500 / PYANNOTE_FRAME_RATE_HZ, atol=1e-3)


def test_extract_turns_filters_short_segments_below_min_duration_on():
    """min_duration_on=0.5s → 0.1s 段被丢弃."""
    from src.core.qwen3.diarize_ort import (
        PYANNOTE_FRAME_RATE_HZ,
        _extract_turns_from_activity,
    )

    activity = np.zeros((1000, 3), dtype=np.int8)
    # 短段 (0.1s) — 应丢弃
    short_end = int(0.1 * PYANNOTE_FRAME_RATE_HZ)
    activity[0:short_end, 0] = 1
    # 长段 (1.0s) — 应保留
    long_start = 100
    long_end = long_start + int(1.0 * PYANNOTE_FRAME_RATE_HZ)
    activity[long_start:long_end, 1] = 1
    turns = _extract_turns_from_activity(activity, min_duration_on=0.5)
    assert len(turns) == 1
    assert turns[0]["local_speaker"] == 1


def test_extract_turns_two_speakers_both_emitted_sorted_by_start():
    from src.core.qwen3.diarize_ort import _extract_turns_from_activity

    activity = np.zeros((1000, 3), dtype=np.int8)
    activity[100:200, 0] = 1  # spk0 first
    activity[300:400, 2] = 1  # spk2 later
    turns = _extract_turns_from_activity(activity, min_duration_on=0.5)
    assert len(turns) == 2
    assert turns[0]["local_speaker"] == 0
    assert turns[1]["local_speaker"] == 2
    assert turns[0]["start"] < turns[1]["start"]


# ==================== run_diarization_ort_cuda ====================


def _make_fake_pyannote_session(class_idx: int):
    sess = MagicMock()
    fake_in = MagicMock()
    fake_in.name = "x"
    sess.get_inputs.return_value = [fake_in]

    def fake_run(_, feed):
        out = np.zeros((1, 589, 7), dtype=np.float32)
        out[..., class_idx] = 1.0
        return [out]

    sess.run.side_effect = fake_run
    return sess


def _make_fake_titanet_session(embedding_vec: np.ndarray):
    sess = MagicMock()
    in_audio = MagicMock()
    in_audio.name = "audio_signal"
    in_len = MagicMock()
    in_len.name = "length"
    sess.get_inputs.return_value = [in_audio, in_len]

    def fake_run(_, feed):
        return [embedding_vec.reshape(1, -1).astype(np.float32)]

    sess.run.side_effect = fake_run
    return sess


def test_run_diarization_ort_cuda_returns_empty_for_silent_audio(tmp_path):
    """全 silence 音频 → pyannote 全 class 0 → 无 turn → 空 list."""
    import soundfile as sf

    from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

    audio_path = tmp_path / "silence.wav"
    sf.write(str(audio_path), np.zeros(16000 * 5, dtype=np.float32), 16000)

    with patch(
        "src.core.qwen3.diarize_ort._get_pyannote_session",
        return_value=_make_fake_pyannote_session(class_idx=0),
    ), patch(
        "src.core.qwen3.diarize_ort._get_titanet_session",
        return_value=_make_fake_titanet_session(np.ones(192, dtype=np.float32)),
    ):
        turns = run_diarization_ort_cuda(
            str(audio_path),
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
    assert turns == []


def test_run_diarization_ort_cuda_returns_turn_list_for_single_speaker(tmp_path):
    """pyannote mock 全 class 1 (spk0) → 该段 audio 1 个 turn, speaker 0."""
    import soundfile as sf

    from src.core.qwen3.diarize_ort import run_diarization_ort_cuda

    audio_path = tmp_path / "voice.wav"
    rng = np.random.RandomState(0)
    sf.write(str(audio_path), (rng.randn(16000 * 5) * 0.1).astype(np.float32), 16000)

    with patch(
        "src.core.qwen3.diarize_ort._get_pyannote_session",
        return_value=_make_fake_pyannote_session(class_idx=1),
    ), patch(
        "src.core.qwen3.diarize_ort._get_titanet_session",
        return_value=_make_fake_titanet_session(np.ones(192, dtype=np.float32)),
    ):
        turns = run_diarization_ort_cuda(
            str(audio_path),
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
    assert len(turns) >= 1
    for t in turns:
        assert t["speaker"] == 0
        assert "start" in t and "end" in t
        assert t["end"] > t["start"]


# ==================== run_diarization_dispatched ====================


def test_dispatch_default_routes_to_sherpa_on_mac():
    """无 env override 时 Mac 上应该路由到 sherpa."""
    import sys

    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(sys, "platform", "darwin"), patch.object(
        diarize_mod, "run_diarization", return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}]
    ) as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda",
        return_value=[],
    ) as ort_mock:
        result = diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
    sherpa_mock.assert_called_once()
    ort_mock.assert_not_called()
    assert result[0]["speaker"] == 0


def test_dispatch_env_override_forces_ort_cuda(monkeypatch):
    """FUNASR_QWEN3_DIARIZE_BACKEND=ort_cuda 强制走 ORT backend."""
    monkeypatch.setenv("FUNASR_QWEN3_DIARIZE_BACKEND", "ort_cuda")
    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(diarize_mod, "run_diarization") as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda",
        return_value=[{"start": 0.0, "end": 1.0, "speaker": 0}],
    ) as ort_mock:
        result = diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
    sherpa_mock.assert_not_called()
    ort_mock.assert_called_once()
    assert result[0]["speaker"] == 0


def test_dispatch_explicit_backend_param_overrides_runtime():
    """显式 backend='sherpa' 参数 > runtime 推荐."""
    from src.core.qwen3 import diarize as diarize_mod

    with patch.object(diarize_mod, "run_diarization", return_value=[]) as sherpa_mock, patch(
        "src.core.qwen3.diarize_ort.run_diarization_ort_cuda"
    ) as ort_mock:
        diarize_mod.run_diarization_dispatched(
            "/fake.wav",
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
            backend="sherpa",
        )
    sherpa_mock.assert_called_once()
    ort_mock.assert_not_called()
