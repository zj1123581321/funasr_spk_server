"""
silence-aware align — Qwen3Config 新增 silence_* 字段 + env override (spike 405abf6)

设计:
- Qwen3Config.silence_align_enabled: bool = True (默认开)
- Qwen3Config.silence_align_tolerance_sec: float = 2.0 (用户拍板平衡值)
- Qwen3Config.silence_align_min_segment_dur_sec: float = 0.1 (snap 后段时长保护)
- Qwen3Config.silence_vad_noise_db: str = "-25dB" (podcast sweet spot)
- Qwen3Config.silence_vad_min_silence_sec: float = 0.20

env override:
- FUNASR_QWEN3_SILENCE_ALIGN_ENABLED (bool)
- FUNASR_QWEN3_SILENCE_ALIGN_TOLERANCE_SEC (float)
- FUNASR_QWEN3_SILENCE_ALIGN_MIN_SEGMENT_DUR_SEC (float)
- FUNASR_QWEN3_SILENCE_VAD_NOISE_DB (str)
- FUNASR_QWEN3_SILENCE_VAD_MIN_SILENCE_SEC (float)
"""
from __future__ import annotations

from src.core.config import Config, Qwen3Config


class TestQwen3ConfigSilenceAlignDefaults:
    def test_default_silence_align_enabled_is_true(self) -> None:
        cfg = Qwen3Config()
        assert cfg.silence_align_enabled is True

    def test_default_tolerance_is_2_0(self) -> None:
        cfg = Qwen3Config()
        assert cfg.silence_align_tolerance_sec == 2.0

    def test_default_min_segment_dur_is_0_1(self) -> None:
        cfg = Qwen3Config()
        assert cfg.silence_align_min_segment_dur_sec == 0.1

    def test_default_vad_noise_db_is_25db(self) -> None:
        cfg = Qwen3Config()
        assert cfg.silence_vad_noise_db == "-25dB"

    def test_default_vad_min_silence_sec_is_0_20(self) -> None:
        cfg = Qwen3Config()
        assert cfg.silence_vad_min_silence_sec == 0.20


class TestQwen3ConfigSilenceAlignEnvOverride:
    def test_env_override_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SILENCE_ALIGN_ENABLED", "false")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.silence_align_enabled is False

    def test_env_override_tolerance(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SILENCE_ALIGN_TOLERANCE_SEC", "3.5")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.silence_align_tolerance_sec == 3.5
        assert isinstance(cfg.qwen3.silence_align_tolerance_sec, float)

    def test_env_override_min_segment_dur(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SILENCE_ALIGN_MIN_SEGMENT_DUR_SEC", "0.25")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.silence_align_min_segment_dur_sec == 0.25

    def test_env_override_vad_noise_db(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SILENCE_VAD_NOISE_DB", "-30dB")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.silence_vad_noise_db == "-30dB"

    def test_env_override_vad_min_silence_sec(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_SILENCE_VAD_MIN_SILENCE_SEC", "0.5")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.silence_vad_min_silence_sec == 0.5
