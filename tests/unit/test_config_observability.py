"""可观测性仪表盘 (P1) — ObservabilityConfig 字段 + env override.

设计定案: docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md
- metrics_enabled: bool = True       (/health + /metrics 端点总开关)
- metrics_token: Optional[str] = None (设了则 /metrics 校验 token; A5: 0.0.0.0 默认下未设则拒)
"""
from __future__ import annotations

import pytest

from src.core.config import ObservabilityConfig, Config


class TestObservabilityDefaults:
    def test_default_metrics_enabled_is_true(self) -> None:
        assert ObservabilityConfig().metrics_enabled is True

    def test_default_metrics_token_is_none(self) -> None:
        assert ObservabilityConfig().metrics_token is None

    def test_config_has_observability_section(self) -> None:
        assert isinstance(Config().observability, ObservabilityConfig)


class TestObservabilityEnvOverride:
    def test_env_override_metrics_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_METRICS_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.observability.metrics_enabled is False

    def test_env_override_metrics_token(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_METRICS_TOKEN", "s3cr3t")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.observability.metrics_token == "s3cr3t"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
