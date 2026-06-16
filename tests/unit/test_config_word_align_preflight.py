"""Lane 1 (#17) — Qwen3Config word_align preflight 阈值字段 + env override.

设计 (评审定案 P1): 阈值进 config (env 可覆盖) + 保守默认 + 3060 标定.
- word_align_preflight_enabled: bool = True   (CUDA word_align 加载前是否探显存)
- word_align_preflight_free_mib: int = 4608    (~4.5GB, 加载 CUDA session 要求的最小空闲)
"""
from __future__ import annotations

import pytest

from src.core.config import Qwen3Config, Config


class TestPreflightDefaults:
    def test_default_preflight_enabled_is_true(self) -> None:
        assert Qwen3Config().word_align_preflight_enabled is True

    def test_default_preflight_free_mib(self) -> None:
        # 保守起点 4.5GB (session ~3.4GB + 余量), 3060 标定后可改
        assert Qwen3Config().word_align_preflight_free_mib == 4608


class TestPreflightEnvOverride:
    def test_env_override_preflight_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_PREFLIGHT_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_preflight_enabled is False

    def test_env_override_preflight_free_mib(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_PREFLIGHT_FREE_MIB", "6000")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_preflight_free_mib == 6000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
