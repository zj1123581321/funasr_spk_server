"""
Phase 1 — Qwen3 num_threads 默认值 8 → 4 (PoC 验证 N=2 wall -11.5%)

设计:
- Qwen3Config.num_threads: int = 4 (新默认)
- env FUNASR_QWEN3_NUM_THREADS 覆盖
- 计算依据: pool_size=2 × num_threads=4 = 8 ≤ 10 cores (M1 Max), 留 2 core 给系统
- 实测依据: spikes/qwen3_mac_hw_accel/num_threads_tuning.md

覆盖:
1. Qwen3Config() 默认 num_threads == 4
2. init 显式覆盖
3. env FUNASR_QWEN3_NUM_THREADS=N 覆盖, int 解析
"""
import pytest

from src.core.config import Qwen3Config, Config


class TestQwen3NumThreadsDefault:
    def test_qwen3_num_threads_default_is_4(self):
        """PoC 验证: t=4 比 t=8 wall -11.5%, 是 M1 Max + pool=2 的最优配置"""
        cfg = Qwen3Config()
        assert cfg.num_threads == 4

    def test_qwen3_num_threads_init_override(self):
        cfg = Qwen3Config(num_threads=8)
        assert cfg.num_threads == 8


class TestQwen3NumThreadsEnvOverride:
    def test_qwen3_num_threads_env_override_still_works(self, monkeypatch, tmp_path):
        """env FUNASR_QWEN3_NUM_THREADS 必须仍能覆盖 (不影响 escape hatch)"""
        monkeypatch.setenv("FUNASR_QWEN3_NUM_THREADS", "8")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.num_threads == 8
        assert isinstance(cfg.qwen3.num_threads, int)

    def test_qwen3_num_threads_no_env_uses_default_4(self, monkeypatch, tmp_path):
        """没设 env 时, 加载完 config 应是新默认 4"""
        monkeypatch.delenv("FUNASR_QWEN3_NUM_THREADS", raising=False)
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.num_threads == 4
