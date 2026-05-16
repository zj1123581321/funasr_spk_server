"""
PR3 — Qwen3Config 加 cluster_merge_* 字段 + env override.

设计:
- cluster_merge_enabled: bool = True
- cluster_merge_min_main_share: float = 0.03
- cluster_merge_relabel_threshold: float = 0.55
- cluster_merge_main_threshold: float = 0.78
- cluster_merge_dominant_share: float = 0.6
- cluster_merge_dominant_threshold: float = 0.6

env override (6 vars).
"""
from __future__ import annotations

import pytest

from src.core.config import Qwen3Config, Config


class TestQwen3ConfigClusterMergeDefaults:
    def test_default_cluster_merge_enabled_is_true(self) -> None:
        assert Qwen3Config().cluster_merge_enabled is True

    def test_default_cluster_merge_min_main_share(self) -> None:
        assert Qwen3Config().cluster_merge_min_main_share == 0.03

    def test_default_cluster_merge_relabel_threshold(self) -> None:
        assert Qwen3Config().cluster_merge_relabel_threshold == 0.55

    def test_default_cluster_merge_main_threshold(self) -> None:
        assert Qwen3Config().cluster_merge_main_threshold == 0.78

    def test_default_cluster_merge_dominant_share(self) -> None:
        assert Qwen3Config().cluster_merge_dominant_share == 0.6

    def test_default_cluster_merge_dominant_threshold(self) -> None:
        assert Qwen3Config().cluster_merge_dominant_threshold == 0.6


class TestQwen3ConfigClusterMergeEnvOverride:
    def test_env_override_cluster_merge_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_enabled is False

    def test_env_override_cluster_merge_min_main_share(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_MIN_MAIN_SHARE", "0.05")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_min_main_share == 0.05

    def test_env_override_cluster_merge_relabel_threshold(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_RELABEL_THRESHOLD", "0.6")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_relabel_threshold == 0.6

    def test_env_override_cluster_merge_main_threshold(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_MAIN_THRESHOLD", "0.8")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_main_threshold == 0.8

    def test_env_override_cluster_merge_dominant_share(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_SHARE", "0.7")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_dominant_share == 0.7

    def test_env_override_cluster_merge_dominant_threshold(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_CLUSTER_MERGE_DOMINANT_THRESHOLD", "0.65")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.cluster_merge_dominant_threshold == 0.65
