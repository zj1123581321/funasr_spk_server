"""
PR3 — _load_audio_mono_16k 加 librosa fallback (m4a/aac/etc).

soundfile 不支持的格式应 fallback 到 librosa, 否则 PR3 cluster_merge
在 m4a 输入上会因为无法加载 audio 而 crash.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestLoadAudioMono16kFallback:
    def test_falls_back_to_librosa_when_soundfile_raises(self, monkeypatch) -> None:
        """sf.read 抛 RuntimeError 时, fallback 到 librosa.load."""
        import src.core.qwen3.diarize as diarize_mod

        # 1) Stub soundfile.read 抛错
        def _stub_sf_read(*args, **kwargs):
            raise RuntimeError("Format not recognised")

        monkeypatch.setattr(diarize_mod.sf, "read", _stub_sf_read)

        # 2) Stub librosa.load 返回固定 audio
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        class _LibrosaStub:
            @staticmethod
            def load(path, sr=None, mono=True):
                return fake_audio.copy(), sr or 16000

        # librosa 是惰性 import, 我们 patch sys.modules
        import sys
        monkeypatch.setitem(sys.modules, "librosa", _LibrosaStub)

        audio, sample_rate = diarize_mod._load_audio_mono_16k("/fake/audio.m4a")
        assert sample_rate == 16000
        assert audio.dtype == np.float32
        assert np.allclose(audio, fake_audio)
