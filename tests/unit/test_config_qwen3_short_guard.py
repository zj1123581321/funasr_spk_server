"""
PR2 — Qwen3Config 加 short_segment_guard_* 字段 + env override

设计:
- Qwen3Config.short_segment_guard_enabled: bool = True (默认开启)
- Qwen3Config.short_segment_drop_sec: float = 1.5
- Qwen3Config.short_segment_aba_max_mid_sec: float = 1.5
- Qwen3Config.short_segment_merge_same: bool = True

env override:
- FUNASR_QWEN3_SHORT_GUARD_ENABLED (bool)
- FUNASR_QWEN3_SHORT_DROP_SEC (float)
- FUNASR_QWEN3_SHORT_ABA_MAX_MID_SEC (float)
- FUNASR_QWEN3_SHORT_MERGE_SAME (bool)
"""
from __future__ import annotations

import pytest

from src.core.config import Qwen3Config, Config


class TestQwen3ConfigShortGuardDefaults:
    """short_segment_guard_* 字段默认值."""

    def test_default_short_segment_guard_enabled_is_true(self) -> None:
        cfg = Qwen3Config()
        assert cfg.short_segment_guard_enabled is True

    def test_default_short_segment_drop_sec_is_1_5(self) -> None:
        cfg = Qwen3Config()
        assert cfg.short_segment_drop_sec == 1.5

    def test_default_short_segment_aba_max_mid_sec_is_1_5(self) -> None:
        cfg = Qwen3Config()
        assert cfg.short_segment_aba_max_mid_sec == 1.5

    def test_default_short_segment_merge_same_is_true(self) -> None:
        cfg = Qwen3Config()
        assert cfg.short_segment_merge_same is True


class TestQwen3ConfigShortGuardEnvOverride:
    """4 个 env vars 覆盖 short_segment_guard_* 字段."""

    def test_env_override_short_guard_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SHORT_GUARD_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.short_segment_guard_enabled is False

    def test_env_override_short_drop_sec(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SHORT_DROP_SEC", "0.3")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.short_segment_drop_sec == 0.3
        assert isinstance(cfg.qwen3.short_segment_drop_sec, float)

    def test_env_override_short_aba_max_mid_sec(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SHORT_ABA_MAX_MID_SEC", "2.0")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.short_segment_aba_max_mid_sec == 2.0

    def test_env_override_short_merge_same(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SHORT_MERGE_SAME", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.short_segment_merge_same is False
