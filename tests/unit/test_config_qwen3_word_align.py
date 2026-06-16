"""
词级时间戳 — Qwen3Config 新增 word_align 5 件套 + env override

设计:
- word_align_enabled: bool = False (新功能默认关, opt-in; 改 cache key + 加 RTF)
- word_align_language: str = "chi" (ISO 码兜底语言, per-request language 优先)
- word_align_model_path: str = "./models/qwen3_diarize/ctc_forced_aligner/model.onnx"
- word_align_provider: str = "auto" (auto → runtime.recommend_word_align_provider())
- word_align_batch_size: int = 16

env override:
- FUNASR_QWEN3_WORD_ALIGN_ENABLED (bool)
- FUNASR_QWEN3_WORD_ALIGN_LANGUAGE (str)
- FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH (str)
- FUNASR_QWEN3_WORD_ALIGN_PROVIDER (str)
- FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE (int)
"""
from __future__ import annotations

from src.core.config import Config, Qwen3Config


class TestWordAlignDefaults:
    def test_default_enabled_is_false(self) -> None:
        assert Qwen3Config().word_align_enabled is False

    def test_default_language_is_chi(self) -> None:
        assert Qwen3Config().word_align_language == "chi"

    def test_default_model_path(self) -> None:
        assert (
            Qwen3Config().word_align_model_path
            == "./models/qwen3_diarize/ctc_forced_aligner/model.onnx"
        )

    def test_default_provider_is_auto(self) -> None:
        # auto 不在 Pydantic 解析 (ORT EP 由 wrapper 按 runtime 选), 保持字面 "auto"
        assert Qwen3Config().word_align_provider == "auto"

    def test_default_batch_size_is_16(self) -> None:
        assert Qwen3Config().word_align_batch_size == 16


class TestWordAlignEnvOverride:
    def test_env_override_enabled(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_ENABLED", "true")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_enabled is True

    def test_env_override_language(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_LANGUAGE", "eng")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_language == "eng"

    def test_env_override_model_path(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_MODEL_PATH", "/tmp/mms.onnx")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_model_path == "/tmp/mms.onnx"

    def test_env_override_provider(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_PROVIDER", "CUDAExecutionProvider")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_provider == "CUDAExecutionProvider"

    def test_env_override_batch_size(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("FUNASR_QWEN3_WORD_ALIGN_BATCH_SIZE", "8")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        cfg = Config.load_from_file(str(config_file))
        assert cfg.qwen3.word_align_batch_size == 8
        assert isinstance(cfg.qwen3.word_align_batch_size, int)
