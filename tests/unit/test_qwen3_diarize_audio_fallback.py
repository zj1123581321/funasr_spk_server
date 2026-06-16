"""
load_audio_mono_16k 加 librosa fallback (m4a/aac/etc).

soundfile 不支持的格式应 fallback 到 librosa, 否则 cluster_merge / word_align
在 m4a 输入上会因为无法加载 audio 而 crash.

注: 2026-06-16 音频加载从 diarize.py 抽到 audio_io.py (DRY, 共享给 sidecar 瘦入口);
diarize._load_audio_mono_16k 现为 audio_io.load_audio_mono_16k 的别名.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestLoadAudioMono16kFallback:
    def test_falls_back_to_librosa_when_soundfile_raises(self, monkeypatch) -> None:
        """sf.read 抛 RuntimeError 时, fallback 到 librosa.load."""
        import src.core.qwen3.audio_io as audio_io

        # 1) Stub soundfile.read 抛错
        def _stub_sf_read(*args, **kwargs):
            raise RuntimeError("Format not recognised")

        monkeypatch.setattr(audio_io.sf, "read", _stub_sf_read)

        # 2) Stub librosa.load 返回固定 audio
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        class _LibrosaStub:
            @staticmethod
            def load(path, sr=None, mono=True):
                return fake_audio.copy(), sr or 16000

        # librosa 是惰性 import, 我们 patch sys.modules
        import sys
        monkeypatch.setitem(sys.modules, "librosa", _LibrosaStub)

        audio, sample_rate = audio_io.load_audio_mono_16k("/fake/audio.m4a")
        assert sample_rate == 16000
        assert audio.dtype == np.float32
        assert np.allclose(audio, fake_audio)

    def test_diarize_alias_still_resolves(self) -> None:
        """diarize._load_audio_mono_16k 仍是 audio_io.load_audio_mono_16k (向后兼容)."""
        from src.core.qwen3.diarize import _load_audio_mono_16k
        from src.core.qwen3.audio_io import load_audio_mono_16k

        assert _load_audio_mono_16k is load_audio_mono_16k
