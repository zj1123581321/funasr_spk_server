"""
PR3 — config 加 qwen3_pool_size 字段

设计:
- TranscriptionConfig.qwen3_pool_size: int = 3 (PoC v5 sweet spot)
- 独立于 max_concurrent_tasks (FunASR=2, Qwen3=3), 切引擎不需要改 env
- env FUNASR_QWEN3_POOL_SIZE 覆盖

覆盖:
1. 默认值 3
2. init 可显式覆盖
3. env FUNASR_QWEN3_POOL_SIZE=N 覆盖, int 解析
4. 同时存在 max_concurrent_tasks 和 qwen3_pool_size, 两者独立
"""
import os
import pytest

from src.core.config import TranscriptionConfig, Config


class TestQwen3PoolSizeField:
    def test_default_qwen3_pool_size_is_3(self):
        cfg = TranscriptionConfig()
        assert cfg.qwen3_pool_size == 3

    def test_qwen3_pool_size_init_override(self):
        cfg = TranscriptionConfig(qwen3_pool_size=4)
        assert cfg.qwen3_pool_size == 4

    def test_qwen3_pool_size_independent_of_max_concurrent_tasks(self):
        """FunASR 池(max_concurrent_tasks) 跟 Qwen3 池(qwen3_pool_size) 是独立字段"""
        cfg = TranscriptionConfig(max_concurrent_tasks=2, qwen3_pool_size=3)
        assert cfg.max_concurrent_tasks == 2
        assert cfg.qwen3_pool_size == 3


class TestQwen3PoolSizeEnvOverride:
    def test_env_override_qwen3_pool_size(self, monkeypatch, tmp_path):
        """FUNASR_QWEN3_POOL_SIZE 应能覆盖默认值, int 解析"""
        monkeypatch.setenv("FUNASR_QWEN3_POOL_SIZE", "4")
        config_file = tmp_path / "config.json"
        config_file.write_text('{"transcription": {"qwen3_pool_size": 3}}')
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.qwen3_pool_size == 4
        assert isinstance(cfg.transcription.qwen3_pool_size, int)

    def test_no_env_uses_config_file_value(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FUNASR_QWEN3_POOL_SIZE", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text('{"transcription": {"qwen3_pool_size": 2}}')
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.qwen3_pool_size == 2

    def test_env_overrides_independent_funasr_and_qwen3(self, monkeypatch, tmp_path):
        """两个池 env 独立, 互不影响"""
        monkeypatch.setenv("FUNASR_MAX_CONCURRENT_TASKS", "2")
        monkeypatch.setenv("FUNASR_QWEN3_POOL_SIZE", "3")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.transcription.max_concurrent_tasks == 2
        assert cfg.transcription.qwen3_pool_size == 3
