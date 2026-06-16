from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.core.qwen3.asr import run_asr_window


class FakeEngine:
    def __init__(self):
        self.config = SimpleNamespace(chunk_size=40.0, memory_num=1)
        self.calls = []

    def asr(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text="窗口文本",
            alignment=None,
            chunks=[
                SimpleNamespace(
                    text="窗口文本",
                    start_time=0.0,
                    end_time=3.0,
                    index=0,
                )
            ],
        )


def test_run_asr_window_loads_bounded_audio_and_uses_engine_config():
    engine = FakeEngine()
    audio = np.zeros(16000 * 3, dtype=np.float32)

    with patch("src.core.qwen3.asr._load_audio_file", return_value=audio) as load_audio:
        result = run_asr_window(
            "/tmp/audio.m4a",
            engine=engine,
            start_second=120.0,
            duration=3.0,
            language="Chinese",
            temperature=0.4,
        )

    load_audio.assert_called_once_with("/tmp/audio.m4a", start_second=120.0, duration=3.0)
    assert result.text == "窗口文本"
    assert result.duration == 3.0
    assert len(result.chunks) == 1
    assert result.chunks[0].end == 3.0
    assert engine.calls[0]["chunk_size_sec"] == 40.0
    assert engine.calls[0]["memory_chunks"] == 1


def test_run_asr_window_rejects_zero_duration():
    with pytest.raises(ValueError, match="duration must be > 0"):
        run_asr_window(
            "/tmp/audio.m4a",
            engine=FakeEngine(),
            start_second=0.0,
            duration=0.0,
        )
