"""
transcriber_dispatch 路由测试 (PR2: 全局唯一引擎模式)

策略变化(相对 PR1):
- PR1: dispatch 按 per-request engine 选 transcriber, 任意引擎都可被请求
- PR2: 服务器启动时由 config.transcription.default_engine 决定**全局唯一引擎**,
  upload_request.engine 字段只能与之相同或留空, 跨引擎请求 reject

接口约束:
    resolve_transcriber(requested_engine)
        - None / "" → 走 config 单例
        - 等于 config → 走 config 单例
        - 不等 → ValueError "Server configured with X, cannot accept Y"
        - config 引擎名未知 → ValueError "未知 ASR 引擎"
"""
from unittest.mock import patch, MagicMock

import pytest

from src.core.transcriber_dispatch import resolve_transcriber


# ==================== fixture: 临时切 default_engine ====================

@pytest.fixture
def server_engine_funasr():
    from src.core.config import config
    original = config.transcription.default_engine
    config.transcription.default_engine = "funasr"
    yield "funasr"
    config.transcription.default_engine = original


@pytest.fixture
def server_engine_qwen3():
    from src.core.config import config
    original = config.transcription.default_engine
    config.transcription.default_engine = "qwen3"
    yield "qwen3"
    config.transcription.default_engine = original


# ==================== 默认引擎走通 ====================

class TestResolveTranscriberFollowsConfig:
    """server engine 由 config 决定, requested engine 留空 / 相同 → 通过"""

    def test_none_falls_back_to_funasr_when_config_funasr(self, server_engine_funasr):
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_get.return_value = MagicMock(name="funasr_instance")
            result = resolve_transcriber(None)
            assert result is mock_get.return_value

    def test_empty_string_falls_back_to_config(self, server_engine_funasr):
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_get.return_value = MagicMock()
            result = resolve_transcriber("")
            assert result is mock_get.return_value

    def test_funasr_matches_config_funasr(self, server_engine_funasr):
        with patch("src.core.funasr_transcriber.get_transcriber") as mock_get:
            mock_get.return_value = MagicMock()
            result = resolve_transcriber("funasr")
            assert result is mock_get.return_value

    def test_qwen3_matches_config_qwen3(self, server_engine_qwen3):
        """PR3: qwen3 走 pool wrapper, 不再调 Qwen3DiarizeTranscriber 单例"""
        with patch(
            "src.core.qwen3_pool_transcriber.get_qwen3_pool_transcriber"
        ) as mock_get:
            mock_get.return_value = MagicMock(name="qwen3_pool_instance")
            result = resolve_transcriber("qwen3")
            assert result is mock_get.return_value

    def test_none_falls_back_to_qwen3_when_config_qwen3(self, server_engine_qwen3):
        with patch(
            "src.core.qwen3_pool_transcriber.get_qwen3_pool_transcriber"
        ) as mock_get:
            mock_get.return_value = MagicMock()
            result = resolve_transcriber(None)
            assert result is mock_get.return_value

    def test_qwen3_dispatch_returns_pool_wrapper_not_diarize_singleton(
        self, server_engine_qwen3
    ):
        """PR3 显式契约: 主进程 qwen3 dispatch 必须返回 pool wrapper, 不再持有 libllama context"""
        # 旧路径不应被调
        with patch(
            "src.core.qwen3_transcriber.get_qwen3_transcriber"
        ) as old_get, patch(
            "src.core.qwen3_pool_transcriber.get_qwen3_pool_transcriber"
        ) as new_get:
            new_get.return_value = MagicMock(name="qwen3_pool_instance")
            resolve_transcriber("qwen3")
            assert new_get.called, "应走 pool wrapper 工厂"
            assert not old_get.called, "主进程 dispatch 不应再调 Qwen3DiarizeTranscriber 单例"


# ==================== Mismatch reject ====================

class TestResolveTranscriberRejectsMismatch:
    """跨引擎请求必须 reject, 错误信息要明确告知 server 配的是哪个 / client 传了哪个"""

    def test_request_qwen3_when_server_funasr_rejects(self, server_engine_funasr):
        with pytest.raises(ValueError) as exc:
            resolve_transcriber("qwen3")
        msg = str(exc.value)
        # 错误信息应同时含 server engine + requested engine, 方便客户端定位
        assert "funasr" in msg
        assert "qwen3" in msg

    def test_request_funasr_when_server_qwen3_rejects(self, server_engine_qwen3):
        with pytest.raises(ValueError) as exc:
            resolve_transcriber("funasr")
        msg = str(exc.value)
        assert "funasr" in msg
        assert "qwen3" in msg


# ==================== 未知 engine ====================

class TestResolveTranscriberUnknownEngine:
    def test_unknown_requested_rejects(self, server_engine_funasr):
        with pytest.raises(ValueError) as exc:
            resolve_transcriber("magic_engine_xyz")
        msg = str(exc.value)
        assert "magic_engine_xyz" in msg

    def test_unknown_config_engine_raises(self):
        """如果 config 配了不支持的引擎名, 即便不传 requested 也应抛"""
        from src.core.config import config
        original = config.transcription.default_engine
        try:
            config.transcription.default_engine = "definitely_not_a_real_engine"
            with pytest.raises(ValueError) as exc:
                resolve_transcriber(None)
            assert "definitely_not_a_real_engine" in str(exc.value)
        finally:
            config.transcription.default_engine = original


# ==================== qwen3_transcriber 模块健壮性 ====================

class TestQwen3TranscriberModule:
    """qwen3_transcriber 模块基本健壮性 — 真实接口实现在 test_qwen3_transcriber.py 验"""

    def test_qwen3_module_importable(self):
        from src.core import qwen3_transcriber
        assert hasattr(qwen3_transcriber, "get_qwen3_transcriber")
        assert hasattr(qwen3_transcriber, "Qwen3DiarizeTranscriber")
