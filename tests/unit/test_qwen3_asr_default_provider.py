"""
Phase 2 Step 2.2 — build_engine_config macOS 默认走 COREML_ANE_FE

设计:
- macOS (sys.platform == 'darwin'): 默认 onnx_provider="COREML_ANE_FE"
- 其他平台: 默认 onnx_provider="CPU"
- 用户显式传 onnx_provider 时必须保留 (不覆盖)
"""
from __future__ import annotations

import sys

from src.core.qwen3.asr import build_engine_config


class TestBuildEngineConfigDefaultProvider:
    def test_default_macos_is_coreml_ane_fe(self, monkeypatch):
        """macOS 默认平台感知: onnx_provider == 'COREML_ANE_FE'"""
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "COREML_ANE_FE"

    def test_default_linux_is_cpu(self, monkeypatch):
        """非 macOS 平台默认仍 CPU (避免在 Linux/Windows 上误触 CoreML 路径)"""
        monkeypatch.setattr(sys, "platform", "linux")
        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "CPU"

    def test_default_windows_is_cpu(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "CPU"


class TestBuildEngineConfigExplicitOverride:
    def test_explicit_cpu_on_macos_not_overridden(self, monkeypatch):
        """用户在 macOS 上显式传 CPU, 不被自动改回 COREML_ANE_FE"""
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(model_dir="/tmp/fake", onnx_provider="CPU")
        assert cfg.onnx_provider == "CPU"

    def test_explicit_coreml_ane_fe_on_linux_not_overridden(self, monkeypatch):
        """用户在 linux 上显式传 COREML_ANE_FE, 也保留 (encoder 自身会 fallback)"""
        monkeypatch.setattr(sys, "platform", "linux")
        cfg = build_engine_config(model_dir="/tmp/fake", onnx_provider="COREML_ANE_FE")
        assert cfg.onnx_provider == "COREML_ANE_FE"


class TestBuildEngineConfigCoremlAneFull:
    """Phase 3: COREML_ANE_FULL 通过 config knob 触发, backend_fn 自动指向 mlpackage"""

    def test_config_value_coreml_ane_full(self, monkeypatch):
        """config.qwen3.asr_encoder_provider='coreml_ane_full' → onnx_provider == COREML_ANE_FULL"""
        monkeypatch.setattr(sys, "platform", "darwin")
        import src.core.config as _config_module
        monkeypatch.setattr(_config_module.config.qwen3, "asr_encoder_provider", "coreml_ane_full")
        cfg = build_engine_config(model_dir="/tmp/fake")
        assert cfg.onnx_provider == "COREML_ANE_FULL"

    def test_coreml_ane_full_uses_mlpackage_backend_fn(self, monkeypatch):
        """COREML_ANE_FULL 时 encoder_backend_fn 默认指向 .mlpackage"""
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(
            model_dir="/tmp/fake",
            onnx_provider="COREML_ANE_FULL",
        )
        assert cfg.encoder_backend_fn.endswith(".mlpackage"), (
            f"COREML_ANE_FULL 应当默认用 mlpackage backend 文件, 实际 {cfg.encoder_backend_fn!r}"
        )

    def test_coreml_ane_full_explicit_backend_fn_preserved(self, monkeypatch):
        """用户显式传 encoder_backend_fn, 不被覆盖"""
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(
            model_dir="/tmp/fake",
            onnx_provider="COREML_ANE_FULL",
            encoder_backend_fn="custom_backend.mlpackage",
        )
        assert cfg.encoder_backend_fn == "custom_backend.mlpackage"

    def test_coreml_ane_fe_uses_onnx_backend_fn_not_mlpackage(self, monkeypatch):
        """COREML_ANE_FE (Phase 2) 保持用 .onnx backend, 不受 Phase 3 影响"""
        monkeypatch.setattr(sys, "platform", "darwin")
        cfg = build_engine_config(
            model_dir="/tmp/fake",
            onnx_provider="COREML_ANE_FE",
        )
        assert cfg.encoder_backend_fn.endswith(".onnx")
