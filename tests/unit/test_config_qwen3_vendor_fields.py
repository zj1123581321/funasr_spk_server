"""
A2 下 — vendor 字段 backend_mlpackage_units + encoder_timing_enabled 提升到 Qwen3Config

设计 (decision D4):
- vendor encoder.py:226 直接读 os.environ.get("FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS") → 改成函数参数
- vendor encoder.py:385 同样读 FUNASR_QWEN3_ENCODER_TIMING → 改成函数参数
- 2 个字段都进 Qwen3Config + .env override + Literal 约束 (backend_mlpackage_units)
- build_engine_config 从 cfg 读字段, 传 ASREngineConfig, 再透传到 QwenAudioEncoder

覆盖:
1. Qwen3Config.backend_mlpackage_units default = "CPU_AND_NE"
2. Qwen3Config.encoder_timing_enabled default = False
3. env FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS 覆盖
4. env FUNASR_QWEN3_ENCODER_TIMING 覆盖
5. Literal 约束: backend_mlpackage_units 无效值 raise
6. ASREngineConfig 含 backend_mlpackage_units + encoder_timing_enabled 字段 + 默认值
7. build_engine_config 把字段传到 ASREngineConfig
8. vendor encoder.py 不再 os.environ.get (grep 验证)
"""
import os
import re
import pytest
from pathlib import Path
from pydantic import ValidationError

import src.core.config as config_module
from src.core.config import Qwen3Config, Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("FUNASR_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("FUNASR_NOTIFICATION_ENABLED", "false")
    monkeypatch.setenv("FUNASR_AUTH_ENABLED", "false")


def _write_config(tmp_path, data):
    import json
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return str(p)


class TestQwen3ConfigVendorFieldsDefault:
    def test_backend_mlpackage_units_default(self):
        """encoder.py:226 老默认 CPU_AND_NE, 字段化后保持"""
        cfg = Qwen3Config()
        assert cfg.backend_mlpackage_units == "CPU_AND_NE"

    def test_encoder_timing_enabled_default_false(self):
        """生产默认不打印 timing (老 env 不设时 False)"""
        cfg = Qwen3Config()
        assert cfg.encoder_timing_enabled is False


class TestEnvOverride:
    def test_backend_mlpackage_units_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS", "CPU_AND_GPU")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.qwen3.backend_mlpackage_units == "CPU_AND_GPU"

    def test_encoder_timing_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_QWEN3_ENCODER_TIMING", "true")
        cfg = Config.load_from_file(_write_config(tmp_path, {}))
        assert cfg.qwen3.encoder_timing_enabled is True


class TestLiteralConstraint:
    def test_invalid_backend_units_raises(self):
        """非 (CPU_AND_NE / CPU_AND_GPU / ALL) 应 Pydantic 报错"""
        with pytest.raises(ValidationError):
            Qwen3Config(backend_mlpackage_units="MAGIC")

    def test_all_valid_values(self):
        for v in ("CPU_AND_NE", "CPU_AND_GPU", "ALL"):
            cfg = Qwen3Config(backend_mlpackage_units=v)
            assert cfg.backend_mlpackage_units == v


class TestASREngineConfigFields:
    def test_asr_engine_config_has_new_fields_with_defaults(self):
        from src.core.vendor.qwen_asr_gguf.inference.schema import ASREngineConfig
        cfg = ASREngineConfig(model_dir="/tmp/fake")
        assert cfg.backend_mlpackage_units == "CPU_AND_NE"
        assert cfg.encoder_timing_enabled is False


class TestBuildEngineConfigPassesFields:
    def test_build_engine_config_reads_qwen3_config_fields(self, monkeypatch):
        """build_engine_config 把 cfg.backend_mlpackage_units / encoder_timing_enabled
        传到 ASREngineConfig"""
        from src.core.qwen3 import asr as asr_module

        # mock config.qwen3 的相关字段
        class _FakeQwen3:
            asr_encoder_provider = "coreml_ane_full"
            backend_mlpackage_units = "CPU_AND_GPU"
            encoder_timing_enabled = True

        class _FakeRoot:
            qwen3 = _FakeQwen3()

        # 注入 fake config 模块
        import src.core.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "config", _FakeRoot())

        eng_cfg = asr_module.build_engine_config(model_dir="/tmp/fake")
        assert eng_cfg.backend_mlpackage_units == "CPU_AND_GPU"
        assert eng_cfg.encoder_timing_enabled is True


class TestVendorEncoderNoMoreOsGetenv:
    def test_encoder_py_no_funasr_env_reads(self):
        """vendor encoder.py 不再有 os.environ.get('FUNASR_QWEN3_*') 或 os.getenv 痕迹"""
        encoder_path = Path(__file__).parent.parent.parent / "src/core/vendor/qwen_asr_gguf/inference/encoder.py"
        text = encoder_path.read_text()
        # 容忍 import os 本身, 但不允许读 FUNASR_QWEN3_* env
        matches = re.findall(r'(os\.environ\.get|os\.getenv)\(\s*["\']FUNASR_QWEN3_', text)
        assert matches == [], f"vendor encoder.py 仍读 FUNASR_QWEN3_* env: {matches}"
