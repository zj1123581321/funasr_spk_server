"""Lane 2 (#18) — sidecar 进程入口 run_sidecar (瘦入口 codex #9).

解析 argv → 组 CUDA WordAligner factory + lean audio_loader → 调 serve.
mock serve + WordAligner, 不起真进程 / 不加载真 ONNX.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.qwen3 import word_align_sidecar as sc


def test_run_sidecar_parses_args_and_serves():
    """argv → serve(sock_path, idle_ttl, factory, audio_loader)."""
    captured = {}

    def _fake_serve(sock_path, *, aligner_factory, audio_loader, idle_ttl_sec):
        captured["sock_path"] = sock_path
        captured["idle_ttl"] = idle_ttl_sec
        captured["factory"] = aligner_factory
        captured["audio_loader"] = audio_loader

    with patch.object(sc, "serve", _fake_serve):
        sc.run_sidecar([
            "--socket", "/tmp/wa.sock",
            "--model-path", "/models/mms.onnx",
            "--language", "chi",
            "--cuda-batch-size", "1",
            "--idle-ttl", "90",
        ])

    assert captured["sock_path"] == "/tmp/wa.sock"
    assert captured["idle_ttl"] == 90.0
    assert callable(captured["factory"])
    assert callable(captured["audio_loader"])


def test_run_sidecar_factory_builds_cuda_aligner():
    """factory() → WordAligner(provider='cuda', model_path, cuda_batch_size)."""
    captured = {}

    def _fake_serve(sock_path, *, aligner_factory, audio_loader, idle_ttl_sec):
        captured["factory"] = aligner_factory

    with patch.object(sc, "serve", _fake_serve):
        sc.run_sidecar([
            "--socket", "/tmp/wa.sock",
            "--model-path", "/models/mms.onnx",
            "--language", "eng",
            "--cuda-batch-size", "1",
            "--idle-ttl", "60",
        ])

    fake_aligner = MagicMock()
    with patch("src.core.qwen3.word_align.WordAligner", return_value=fake_aligner) as WA:
        aligner = captured["factory"]()
    assert aligner is fake_aligner
    _, kwargs = WA.call_args
    assert kwargs["provider"] == "cuda"
    assert kwargs["model_path"] == "/models/mms.onnx"
    assert kwargs["language"] == "eng"
    assert kwargs["cuda_batch_size"] == 1


def test_run_sidecar_audio_loader_is_lean():
    """audio_loader 用 audio_io.load_audio_mono_16k (不拖 diarize/sherpa), 返回 1D 波形."""
    captured = {}

    def _fake_serve(sock_path, *, aligner_factory, audio_loader, idle_ttl_sec):
        captured["audio_loader"] = audio_loader

    with patch.object(sc, "serve", _fake_serve):
        sc.run_sidecar([
            "--socket", "/tmp/wa.sock", "--model-path", "/m.onnx",
            "--language", "chi", "--cuda-batch-size", "1", "--idle-ttl", "90",
        ])

    import numpy as np
    with patch("src.core.qwen3.audio_io.load_audio_mono_16k",
               return_value=(np.zeros(16000, dtype=np.float32), 16000)) as loader:
        out = captured["audio_loader"]("/x.wav")
    loader.assert_called_once_with("/x.wav")
    assert out.shape == (16000,)  # 返回 1D 波形 (剥掉 sample_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
