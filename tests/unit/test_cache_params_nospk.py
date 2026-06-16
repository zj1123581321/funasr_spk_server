"""diarize 开关 (D9 + D4 收拢) — 缓存 key +nospk 折维 + cache_params_for 统一入口

缓存 tag 形态 (顺序固定 +wa 在前 +nospk 在后, 缺省不写; 仅 qwen3 折维):
  qwen3 / qwen3+wa:<lang> / qwen3+nospk / qwen3+wa:<lang>+nospk
funasr 免折维 (D4): 存一行 diarized, serve 层按需投影, 一行通吃两种请求.
带任何折维 tag 时禁 cross-engine 回退 (T-D #7).
language 规范化: strip() or fallback, None/""/" eng " 三态 (T-D #8).
维度 >3 时升级结构化 variant (D9 触发条件, 目前 2 维 4 形态).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database import (
    cache_lookup_params,
    cache_params,
    cache_params_for,
    compute_cache_engine,
)
from src.models.schemas import TranscribeOptions, TranscriptionTask


# ==================== compute_cache_engine 4 形态矩阵 ====================


class TestComputeCacheEngineMatrix:
    def test_plain_qwen3(self):
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=False, language=None,
            word_align_language="chi", diarize=True,
        )
        assert tag == "qwen3"

    def test_wa_only(self):
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=True, language=None,
            word_align_language="chi", diarize=True,
        )
        assert tag == "qwen3+wa:chi"

    def test_nospk_only(self):
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=False, language=None,
            word_align_language="chi", diarize=False,
        )
        assert tag == "qwen3+nospk"

    def test_wa_plus_nospk_order_fixed(self):
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=True, language="eng",
            word_align_language="chi", diarize=False,
        )
        assert tag == "qwen3+wa:eng+nospk"

    def test_diarize_default_true_keeps_old_tag(self):
        """不传 diarize → 老 key 不变 (向后兼容)"""
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=False, language=None,
            word_align_language="chi",
        )
        assert tag == "qwen3"

    def test_funasr_never_folds(self):
        """funasr 免折维 (D4): diarize=False 也不折, 一行通吃"""
        tag = compute_cache_engine(
            "funasr", word_align_enabled=False, language=None,
            word_align_language="chi", diarize=False,
        )
        assert tag == "funasr"


# ==================== language 规范化 (T-D #8) ====================


class TestLanguageNormalization:
    @pytest.mark.parametrize("lang,expected", [
        (None, "qwen3+wa:chi"),       # None → config 兜底
        ("", "qwen3+wa:chi"),         # 空串 → 兜底
        ("  ", "qwen3+wa:chi"),       # 纯空白 → 兜底
        (" eng ", "qwen3+wa:eng"),    # 两侧空白 strip
        ("eng", "qwen3+wa:eng"),
    ])
    def test_language_strip_or_fallback(self, lang, expected):
        tag = compute_cache_engine(
            "qwen3", word_align_enabled=True, language=lang,
            word_align_language="chi", diarize=True,
        )
        assert tag == expected


# ==================== strict 回退 (T-D #7) ====================


class TestLookupStrictness:
    def test_nospk_tag_forbids_cross_engine(self):
        tag, allow_cross = cache_lookup_params(
            "qwen3", word_align_enabled=False, language=None,
            word_align_language="chi", diarize=False,
        )
        assert tag == "qwen3+nospk"
        assert allow_cross is False

    def test_plain_tag_keeps_config_default(self):
        tag, allow_cross = cache_lookup_params(
            "qwen3", word_align_enabled=False, language=None,
            word_align_language="chi", diarize=True,
        )
        assert tag == "qwen3"
        assert allow_cross is None  # 走 config.transcription.cache_cross_engine


# ==================== cache_params / cache_params_for 收拢入口 ====================


@pytest.fixture
def wa_off():
    from src.core.config import config
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = False
    yield
    config.qwen3.word_align_enabled = original


@pytest.fixture
def wa_on():
    from src.core.config import config
    original = config.qwen3.word_align_enabled
    config.qwen3.word_align_enabled = True
    yield
    config.qwen3.word_align_enabled = original


class TestCacheParamsFor:
    def _task(self, engine="qwen3", language=None, diarize=True, word_align=False):
        return TranscriptionTask(
            task_id="t", file_name="a.wav", file_path="", file_size=1,
            file_hash="h", engine=engine,
            options=TranscribeOptions(language=language, diarize=diarize, word_align=word_align),
        )

    def test_reads_options_word_align(self):
        """决策 1A: cache_params_for 折维读 options.word_align (effective 值), 非全局 config."""
        tag, allow_cross = cache_params_for(
            self._task(language="eng", diarize=False, word_align=True)
        )
        assert tag == "qwen3+wa:eng+nospk"
        assert allow_cross is False

    def test_plain_task(self, wa_off):
        tag, allow_cross = cache_params_for(self._task())
        assert tag == "qwen3"
        assert allow_cross is None

    def test_cache_params_without_task_object(self, wa_off):
        """分片 finalize 阶段无 task 对象, 用 (engine, options) 低层入口"""
        tag, allow_cross = cache_params(
            "qwen3", TranscribeOptions(language=None, diarize=False)
        )
        assert tag == "qwen3+nospk"
        assert allow_cross is False


# ==================== 4 处重复消灭: task_manager / websocket_handler 走收拢入口 ====================


class TestCallSitesUseCacheParams:
    @pytest.mark.asyncio
    async def test_submit_task_lookup_uses_nospk_tag(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager
        from src.core.config import config

        tm = TaskManager()
        task = TranscriptionTask(
            task_id="t-ns", file_name="a.wav", file_path="", file_size=1,
            file_hash="h-ns", engine="qwen3",
            options=TranscribeOptions(diarize=False),
        )
        tm.tasks["t-ns"] = task
        f = tmp_path / "x.wav"
        f.write_bytes(b"\x00" * 10)
        with patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            await tm.submit_task("t-ns", str(f))
            kwargs = mock_db.get_cached_result.call_args.kwargs
            assert kwargs.get("engine") == "qwen3+nospk"
            assert kwargs.get("allow_cross_engine") is False

    @pytest.mark.asyncio
    async def test_process_task_saves_with_nospk_tag(self, wa_off, tmp_path):
        from src.core.task_manager import TaskManager
        from src.models.schemas import TranscriptionResult, TranscriptionSegment

        mgr = TaskManager()
        task = TranscriptionTask(
            task_id="t-sv", file_name="x.wav", file_path=str(tmp_path / "x.wav"),
            file_size=100, file_hash="h", engine="qwen3",
            options=TranscribeOptions(diarize=False),
        )
        mgr.tasks["t-sv"] = task
        (tmp_path / "x.wav").write_bytes(b"\x00" * 100)

        fake_result = TranscriptionResult(
            task_id="t-sv", file_name="x.wav", file_hash="h", duration=1.0,
            segments=[TranscriptionSegment(start_time=0, end_time=1, text="hi", speaker="Speaker1")],
            speakers=["Speaker1"], processing_time=0.1,
        )
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe = AsyncMock(return_value=(fake_result, {}))

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_transcriber), \
             patch("src.core.task_manager.db_manager") as mock_db:
            mock_db.save_result = AsyncMock()
            with patch.object(mgr, "_notify_task_progress", new=AsyncMock()), \
                 patch.object(mgr, "_notify_task_complete", new=AsyncMock()):
                await mgr._process_task("t-sv")
        assert mock_db.save_result.call_args.kwargs.get("engine") == "qwen3+nospk"

    @pytest.mark.asyncio
    async def test_chunked_finalize_lookup_uses_nospk_tag(self, wa_off, tmp_path):
        """分片 finalize 的缓存查询走收拢入口, tag 折 +nospk"""
        from src.api.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        temp_file = tmp_path / "chunks.bin"
        temp_file.write_bytes(b"\x00" * 8)
        session = {
            "task_id": "t-fin2", "file_name": "x.wav", "file_size": 8,
            "file_hash": "match", "chunk_size": 8, "total_chunks": 1,
            "received_chunks": 1, "temp_file_path": str(temp_file),
            "chunks_received": {0}, "output_format": "json",
            "force_refresh": False, "connection_id": "conn-1",
            "engine": "qwen3", "language": None, "diarize": False,
        }
        handler.upload_sessions["t-fin2"] = session
        ws = MagicMock()
        ws.send = AsyncMock()

        with patch.object(handler, "_calculate_file_hash", return_value="match"), \
             patch("src.core.database.db_manager") as mock_db, \
             patch("src.utils.file_utils.save_uploaded_file",
                   new=AsyncMock(return_value=(str(tmp_path / "saved.wav"), None))), \
             patch("src.core.task_manager.task_manager") as mock_tm:
            mock_db.get_cached_result = AsyncMock(return_value=None)
            mock_tm.create_task = AsyncMock(return_value=TranscriptionTask(
                task_id="t-fin2", file_name="x.wav", file_path="",
                file_size=8, file_hash="match",
            ))
            mock_tm.submit_task = AsyncMock(return_value=None)
            await handler._finalize_chunked_upload(ws, "t-fin2")

        kwargs = mock_db.get_cached_result.call_args.kwargs
        assert kwargs.get("engine") == "qwen3+nospk"
        assert kwargs.get("allow_cross_engine") is False
