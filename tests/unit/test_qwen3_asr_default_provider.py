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
