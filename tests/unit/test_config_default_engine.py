"""
PR1 — config 默认引擎字段

验证 TranscriptionConfig.default_engine 字段存在 + env override 工作
"""
import os
import pytest

from src.core.config import TranscriptionConfig, Config


class TestTranscriptionConfigDefaultEngine:
    def test_default_engine_field_exists_and_defaults_to_funasr(self):
        cfg = TranscriptionConfig()
        assert cfg.default_engine == "funasr"

    def test_default_engine_can_be_overridden_via_init(self):
        cfg = TranscriptionConfig(default_engine="qwen3")
        assert cfg.default_engine == "qwen3"


class TestEnvOverride:
    def test_env_override_default_engine(self, monkeypatch, tmp_path):
        """FUNASR_DEFAULT_ENGINE 应能覆盖 default_engine"""
        monkeypatch.setenv("FUNASR_DEFAULT_ENGINE", "qwen3")
        # 用一个最小的 config.json 触发完整 load 流程
        config_file = tmp_path / "config.json"
        config_file.write_text('{"transcription": {"default_engine": "funasr"}}')
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.default_engine == "qwen3"

    def test_no_env_uses_config_file_value(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FUNASR_DEFAULT_ENGINE", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text('{"transcription": {"default_engine": "funasr"}}')
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.default_engine == "funasr"
