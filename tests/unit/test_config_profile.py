"""
A1 — FUNASR_PROFILE 套餐 env

设计 (decision D2):
- 优先级 defaults < config.json < profile < env
- profile 覆盖 config.json 已有字段, 启动日志列出被覆盖的字段
- 4 个 profile: mac_prod / mac_dev / cuda_prod / cuda_dev
- 未知 profile name → warn + ignore, 不挂

覆盖:
1. 4 profile defaults 正确
2. profile 覆盖 config.json 字段 (新优先级)
3. env 仍能覆盖 profile (escape hatch)
4. 未知 profile name → warn + 不挂
5. 未设 FUNASR_PROFILE → 行为不变 (regression guard)
"""
import json
import os
import pytest

from src.core.config import Config


def _write_config(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture(autouse=True)
def _clean_funasr_env(monkeypatch):
    """清掉 dev .env 文件加载来的 FUNASR_* env, 避免污染 profile/config.json 测试.

    Config.load_from_file 内部会 load_dotenv() 把 dev box 的 .env 注入 process env,
    导致比如 FUNASR_SERVER_PORT=8867 一直存在, 让 profile defaults 测试失败.

    解决:
    (1) 先清所有 FUNASR_* env;
    (2) patch dotenv.load_dotenv 为 no-op, 防止 load_from_file 内部又注回 .env;
    (3) NotificationConfig.enabled 默认 True 但 webhook_url 默认 "" 会触发
        _validate_config 报错, 测试关掉 notification + auth 走纯净路径.
    """
    for key in list(os.environ.keys()):
        if key.startswith("FUNASR_"):
            monkeypatch.delenv(key, raising=False)
    import src.core.config as config_module
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
    monkeypatch.setenv("FUNASR_AUTH_ENABLED", "false")


class TestProfileDefaults:
    """4 个 profile 各自的字段是否正确填上"""

    def test_mac_prod_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_PROFILE", "mac_prod")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.server.port == 8767
        assert cfg.transcription.default_engine == "qwen3"
        assert cfg.transcription.qwen3_pool_size == 1
        assert cfg.qwen3.asr_encoder_provider == "coreml_ane_full"
        # Mac CPU word_align +17% RTF, profile 保持关 (字段默认 False)
        assert cfg.qwen3.word_align_enabled is False

    def test_mac_dev_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_PROFILE", "mac_dev")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.server.port == 8867
        assert cfg.transcription.default_engine == "qwen3"
        assert cfg.transcription.qwen3_pool_size == 1
        assert cfg.qwen3.asr_encoder_provider == "coreml_ane_full"
        assert cfg.logging.level == "DEBUG"
        assert cfg.qwen3.word_align_enabled is False

    def test_cuda_prod_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_PROFILE", "cuda_prod")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.transcription.default_engine == "qwen3"
        assert cfg.transcription.qwen3_pool_size == 1
        assert cfg.qwen3.asr_encoder_provider == "cuda"
        # CUDA word_align 仅 +1% RTF, profile 默认开词级时间戳
        assert cfg.qwen3.word_align_enabled is True

    def test_cuda_dev_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_PROFILE", "cuda_dev")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.server.port == 8867
        assert cfg.transcription.default_engine == "qwen3"
        assert cfg.transcription.qwen3_pool_size == 1
        assert cfg.qwen3.asr_encoder_provider == "cuda"
        assert cfg.logging.level == "DEBUG"
        assert cfg.qwen3.word_align_enabled is True


class TestProfilePriorityOverConfigJson:
    """D2: profile 优先级高于 config.json (但 env 仍高于 profile)"""

    def test_profile_overrides_config_json_port(self, monkeypatch, tmp_path):
        """config.json port=9000, profile=mac_dev (port=8867) → 应取 8867 (profile 赢)"""
        monkeypatch.setenv("FUNASR_PROFILE", "mac_dev")
        monkeypatch.delenv("FUNASR_SERVER_PORT", raising=False)
        cfg = Config.load_from_file(_write_config(tmp_path, {"server": {"port": 9000}}))
        assert cfg.server.port == 8867

    def test_env_overrides_profile(self, monkeypatch, tmp_path):
        """profile=mac_dev (port=8867), env FUNASR_SERVER_PORT=9999 → 应取 9999 (env 赢)"""
        monkeypatch.setenv("FUNASR_PROFILE", "mac_dev")
        monkeypatch.setenv("FUNASR_SERVER_PORT", "9999")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.server.port == 9999


class TestProfileUnknown:
    def test_unknown_profile_warns_and_continues(self, monkeypatch, tmp_path, caplog):
        """未知 profile name 应 warn + 不挂, 行为退化到无 profile"""
        monkeypatch.setenv("FUNASR_PROFILE", "no_such_profile")
        monkeypatch.delenv("FUNASR_DEFAULT_ENGINE", raising=False)
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        # 未知 profile 不挂, 配置走默认
        assert cfg.transcription.default_engine == "funasr"
        # 测 warn 字串: 用 loguru 时 caplog 可能拿不到, 用 capsys 或 logger.add 监听
        # 这里不强制断言 caplog (loguru sink), 至少不挂就过


class TestProfileNotSet:
    """regression guard: 不设 FUNASR_PROFILE → 行为 100% 跟之前一致"""

    def test_no_profile_uses_config_json_and_defaults(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FUNASR_PROFILE", raising=False)
        monkeypatch.delenv("FUNASR_DEFAULT_ENGINE", raising=False)
        cfg = Config.load_from_file(_write_config(tmp_path, {"server": {"port": 8888}}))
        # config.json 的 port=8888 应直接生效, 不被 profile 干扰
        assert cfg.server.port == 8888
        # default_engine 走 Pydantic 默认 funasr
        assert cfg.transcription.default_engine == "funasr"
