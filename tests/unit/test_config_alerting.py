"""ObservabilityConfig 告警阈值字段单测 (默认值 + env 覆盖)。

写法对齐 test_config_observability.py: 默认值用 ObservabilityConfig() 直构 (不读 env),
env 覆盖用 Config.load_from_file + 空 config.json (不用 importlib.reload, 避免污染
全局 config 单例)。
"""
from __future__ import annotations

import pytest

from src.core.config import ObservabilityConfig, Config


class TestAlertDefaults:
    def test_alert_disabled_by_default(self):
        assert ObservabilityConfig().alert_enabled is False

    def test_queue_saturation_ratio_default(self):
        assert ObservabilityConfig().alert_queue_saturation_ratio == 0.8

    def test_error_surge_threshold_default(self):
        assert ObservabilityConfig().alert_error_surge_threshold == 5

    def test_cooldown_seconds_default(self):
        assert ObservabilityConfig().alert_cooldown_seconds == 900


class TestAlertEnvOverride:
    def _load(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        return Config.load_from_file(str(config_file))

    def test_enabled_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_ALERT_ENABLED", "true")
        assert self._load(tmp_path).observability.alert_enabled is True

    def test_ratio_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_ALERT_QUEUE_SATURATION_RATIO", "0.95")
        assert self._load(tmp_path).observability.alert_queue_saturation_ratio == 0.95

    def test_surge_threshold_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_ALERT_ERROR_SURGE_THRESHOLD", "12")
        assert self._load(tmp_path).observability.alert_error_surge_threshold == 12

    def test_cooldown_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUNASR_ALERT_COOLDOWN_SECONDS", "300")
        assert self._load(tmp_path).observability.alert_cooldown_seconds == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
