"""TitaNet mel preprocessing 单元测试.

NeMo TitaNet 默认 mel 配置 (硬编码到 ONNX 之前的 preprocessor):
- sample_rate=16000, n_window_size=400 (25ms), n_window_stride=160 (10ms)
- n_fft=512, n_mels=80, window=hann, center=True
- preemphasis=0.97, mag_power=2.0
- log + per_feature normalize (mean/std per mel band 沿 time axis)

本文件不依赖 nemo-toolkit (mac 上装不动), 只测:
1. shape / 范围
2. preemphasis 应用
3. per_feature 归一化结果

NeMo parity 留给 commit 9 的 sherpa end-to-end embedding cosine 验证 — mel 中间结果
跟 ORT inference 串起来跑 192-d embedding 跟 sherpa CPU baseline 比 cosine ≥ 0.99
就证明 mel 实现 correct.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_log_mel_shape_for_5s_audio_is_80_by_about_500():
    """5s @ 16k → ~500 mel frames, n_mels=80."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    audio = np.random.RandomState(0).randn(5 * 16000).astype(np.float32) * 0.1
    mel = compute_titanet_log_mel(audio)
    assert mel.ndim == 2
    assert mel.shape[0] == 80
    # T_mel = 1 + (audio_len + 2*pad - n_fft) / hop = 1 + (80000+512-512)/160 = 501
    assert mel.shape[1] in {500, 501, 502}


def test_log_mel_short_audio_yields_at_least_one_frame():
    """100ms audio → 至少 1 frame, 不 raise."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    audio = np.random.RandomState(1).randn(1600).astype(np.float32) * 0.1
    mel = compute_titanet_log_mel(audio)
    assert mel.shape[0] == 80
    assert mel.shape[1] >= 1


def test_log_mel_silence_input_does_not_produce_nan_or_inf():
    """全 0 audio → log(0 + zero_guard) finite, normalize 不除 0."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    audio = np.zeros(2 * 16000, dtype=np.float32)
    mel = compute_titanet_log_mel(audio)
    assert np.isfinite(mel).all()


def test_log_mel_per_feature_normalize_zero_mean_unit_std():
    """每个 mel band 沿 time axis: mean ≈ 0, std ≈ 1 (per_feature 归一化)."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    rng = np.random.RandomState(42)
    audio = rng.randn(3 * 16000).astype(np.float32) * 0.1
    mel = compute_titanet_log_mel(audio)
    # 每 row (band) 的 mean 应接近 0
    band_means = mel.mean(axis=1)
    band_stds = mel.std(axis=1)
    np.testing.assert_allclose(band_means, 0.0, atol=1e-3)
    np.testing.assert_allclose(band_stds, 1.0, atol=1e-3)


def test_log_mel_preemphasis_changes_spectrum_vs_zero_preemph():
    """preemphasis 0.97 跟 0 应该产生不同 spectrum (高频 emphasize)."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    rng = np.random.RandomState(2)
    audio = rng.randn(1 * 16000).astype(np.float32) * 0.1
    mel_with = compute_titanet_log_mel(audio, preemph=0.97)
    mel_without = compute_titanet_log_mel(audio, preemph=0.0)
    # 顶 mel band (高频) 应该有差异 (preemphasis 强化高频)
    assert not np.allclose(mel_with[-10:], mel_without[-10:], atol=1e-3)


def test_log_mel_dtype_is_float32():
    """ONNX 期望 float32 输入, mel 输出必须 float32."""
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    audio = np.random.RandomState(3).randn(8000).astype(np.float32) * 0.1
    mel = compute_titanet_log_mel(audio)
    assert mel.dtype == np.float32


def test_log_mel_default_kwargs_match_nemo_titanet_config():
    """signature 默认值要跟 NeMo TitaNet preprocessor config 对齐."""
    import inspect
    from src.core.qwen3.diarize_ort import compute_titanet_log_mel

    sig = inspect.signature(compute_titanet_log_mel)
    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    assert defaults["sample_rate"] == 16000
    assert defaults["n_window_size"] == 400
    assert defaults["n_window_stride"] == 160
    assert defaults["n_fft"] == 512
    assert defaults["n_mels"] == 80
    assert defaults["preemph"] == pytest.approx(0.97)
