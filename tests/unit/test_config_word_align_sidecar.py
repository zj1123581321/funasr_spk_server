"""Lane 2 (#18) — Qwen3Config word_align sidecar 字段 + env override.

设计 (评审定案 A1/A3): sidecar 进程参数走 config (env 可覆盖).
- word_align_sidecar_enabled: bool = True    (仅 cuda runtime 生效, runtime gate 另控)
- word_align_sidecar_idle_ttl_sec: float = 90 (空闲多久自杀释放 VRAM)
- word_align_sidecar_align_timeout_sec: float = 180 (单次对齐超时, 超时杀 sidecar)
"""
from __future__ import annotations

import pytest

from src.core.config import Qwen3Config, Config


class TestSidecarDefaults:
    def test_default_sidecar_enabled_is_true(self) -> None:
        assert Qwen3Config().word_align_sidecar_enabled is True

    def test_default_idle_ttl(self) -> None:
        assert Qwen3Config().word_align_sidecar_idle_ttl_sec == 90.0

    def test_default_align_timeout(self) -> None:
        assert Qwen3Config().word_align_sidecar_align_timeout_sec == 180.0


class TestSidecarEnvOverride:
    def test_env_override_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_SIDECAR_ENABLED", "false")
        f = tmp_path / "config.json"; f.write_text("{}")
        assert Config.load_from_file(str(f)).qwen3.word_align_sidecar_enabled is False

    def test_env_override_idle_ttl(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_SIDECAR_IDLE_TTL_SEC", "45")
        f = tmp_path / "config.json"; f.write_text("{}")
        assert Config.load_from_file(str(f)).qwen3.word_align_sidecar_idle_ttl_sec == 45.0

    def test_env_override_align_timeout(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_SIDECAR_ALIGN_TIMEOUT_SEC", "300")
        f = tmp_path / "config.json"; f.write_text("{}")
        assert Config.load_from_file(str(f)).qwen3.word_align_sidecar_align_timeout_sec == 300.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
