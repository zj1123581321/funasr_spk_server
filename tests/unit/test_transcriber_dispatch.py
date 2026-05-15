"""
PR1 — transcriber_dispatch 路由测试

resolve_transcriber 是 PR1 阶段的薄 dispatch 函数（不是 EngineRouter 类）。
意图：让 task.engine 名能选到正确的 transcriber 单例。
PR2 阶段会演进为完整的 ASREngine 抽象 + factory。
"""
from unittest.mock import patch, MagicMock

import pytest

from src.core.transcriber_dispatch import resolve_transcriber


class TestResolveTranscriber:
    def test_funasr_returns_funasr_transcriber(self):
        """engine='funasr' → 返回 FunASR 单例（即现有 get_transcriber()）"""
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_transcriber = MagicMock(name="funasr_instance")
            mock_get.return_value = mock_transcriber
            result = resolve_transcriber("funasr")
            assert result is mock_transcriber
            mock_get.assert_called_once()

    def test_empty_string_falls_back_to_default(self):
        """空字符串视为未指定 → 走 config.default_engine（默认 funasr）"""
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_transcriber = MagicMock()
            mock_get.return_value = mock_transcriber
            result = resolve_transcriber("")
            assert result is mock_transcriber

    def test_none_falls_back_to_default(self):
        """None 视为未指定 → 走 config.default_engine"""
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_transcriber = MagicMock()
            mock_get.return_value = mock_transcriber
            result = resolve_transcriber(None)
            assert result is mock_transcriber

    def test_unknown_engine_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_transcriber("nonexistent_engine_xyz")
        assert "nonexistent_engine_xyz" in str(exc_info.value)

    def test_qwen3_returns_qwen3_transcriber(self):
        """PR1 阶段 Qwen3 是占位，但 dispatch 应能解析到它"""
        with patch("src.core.qwen3_transcriber.get_qwen3_transcriber") as mock_get:
            mock_transcriber = MagicMock(name="qwen3_instance")
            mock_get.return_value = mock_transcriber
            result = resolve_transcriber("qwen3")
            assert result is mock_transcriber

    def test_default_engine_respects_config(self, monkeypatch):
        """当 engine 为 None 时应从 config 读 default_engine"""
        from src.core.config import config
        # 临时把 default_engine 切到 qwen3
        original = config.transcription.default_engine
        try:
            config.transcription.default_engine = "qwen3"
            with patch("src.core.qwen3_transcriber.get_qwen3_transcriber") as mock_get:
                mock_transcriber = MagicMock()
                mock_get.return_value = mock_transcriber
                result = resolve_transcriber(None)
                assert result is mock_transcriber
        finally:
            config.transcription.default_engine = original


class TestQwen3TranscriberPlaceholder:
    """Qwen3 transcriber 在 PR1 是占位，spike 后才落地"""

    def test_qwen3_module_importable(self):
        """模块必须能 import（即便实现是占位）"""
        from src.core import qwen3_transcriber
        assert hasattr(qwen3_transcriber, "get_qwen3_transcriber")

    def test_qwen3_transcribe_raises_not_implemented(self):
        """transcribe 调用应明确抛 NotImplementedError，标注 PR1 阶段未启用"""
        from src.core.qwen3_transcriber import get_qwen3_transcriber
        t = get_qwen3_transcriber()
        with pytest.raises(NotImplementedError) as exc:
            # 同步调用最浅层断言 — 不必走 async
            t.transcribe_sync_stub()
        assert "PR1" in str(exc.value) or "spike" in str(exc.value).lower()
