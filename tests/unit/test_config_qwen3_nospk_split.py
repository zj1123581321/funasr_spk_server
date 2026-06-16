"""diarize 开关 (落地步骤 4, D5 config 件套) — nospk_split_* 字段 + env override

- nospk_split_enabled: bool = True (diarize=false 时对超长 chunk 段做分层切分)
- nospk_split_max_segment_sec: float = 12.0 (超过此时长触发切分)
- nospk_split_min_segment_sec: float = 1.5 (切出片的最小时长, 吸收
  short_segment_guard 的通用清理职责)

env override:
- FUNASR_QWEN3_NOSPK_SPLIT_ENABLED (bool)
- FUNASR_QWEN3_NOSPK_SPLIT_MAX_SEGMENT_SEC (float)
- FUNASR_QWEN3_NOSPK_SPLIT_MIN_SEGMENT_SEC (float)
"""
from __future__ import annotations

from src.core.config import Config, Qwen3Config


class TestNospkSplitDefaults:
    def test_default_enabled_true(self) -> None:
        cfg = Qwen3Config()
        assert cfg.nospk_split_enabled is True

    def test_default_max_segment_sec(self) -> None:
        cfg = Qwen3Config()
        assert cfg.nospk_split_max_segment_sec == 12.0

    def test_default_min_segment_sec(self) -> None:
        cfg = Qwen3Config()
        assert cfg.nospk_split_min_segment_sec == 1.5


class TestNospkSplitEnvOverride:
    def test_env_override_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_NOSPK_SPLIT_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.nospk_split_enabled is False

    def test_env_override_max_segment_sec(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_NOSPK_SPLIT_MAX_SEGMENT_SEC", "20.5")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.nospk_split_max_segment_sec == 20.5

    def test_env_override_min_segment_sec(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_NOSPK_SPLIT_MIN_SEGMENT_SEC", "2.5")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.nospk_split_min_segment_sec == 2.5
