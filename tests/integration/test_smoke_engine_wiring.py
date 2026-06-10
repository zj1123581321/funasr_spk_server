"""
冒烟测试 — task_manager + dispatch + db 端到端 wiring

设计:
- 不起 WebSocket server, 直接调 task_manager API (绕开 socket 复杂度)
- 不依赖真模型, mock resolve_transcriber 返 fake transcribe
- 真走 worker queue + 真 cache 写读 + 真 dispatch + 真 strict validate

覆盖:
1. happy path: create → submit → worker → resolve → fake.transcribe → save_result → COMPLETED
2. cache 命中: 同 hash 同 engine 第二次 → fake 不被调
3. 跨引擎缓存命中: server=funasr 留缓存 → 切 server=qwen3 同 hash → fake 不被调
4. engine mismatch reject: server=funasr, request=qwen3 → ValueError 在 create_task

跑得快(秒级, 不加载模型), 默认 enabled, 用作 wiring 回归.
模型质量验证留给 tests/integration/test_*_e2e.py (FUNASR_RUN_INTEGRATION=1 启用).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.config import config
from src.core.database import DatabaseManager
from src.core.task_manager import TaskManager
from src.models.schemas import (
    FileUploadRequest,
    TaskStatus,
    TranscriptionResult,
    TranscriptionSegment,
)


# ==================== fake transcriber ====================

def _make_fake_transcriber(text: str = "冒烟测试文本", file_hash: str = "h"):
    """构造 fake transcriber, 接口与 FunASR / Qwen3 同形"""
    fake = MagicMock()

    async def fake_transcribe(audio_path, task_id, progress_callback=None, output_format="json", options=None):
        if progress_callback:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(50)
                await progress_callback(100)
            else:
                progress_callback(50)
                progress_callback(100)
        file_name = Path(audio_path).name
        if output_format == "srt":
            return {
                "format": "srt",
                "content": f"1\n00:00:00,000 --> 00:00:01,000\nSpeaker1:{text}\n",
                "file_name": file_name,
                "file_hash": file_hash,
                "duration": 1.0,
                "processing_time": 0.01,
                "raw_result": {"asr_text": text, "engine": "fake"},
            }
        result = TranscriptionResult(
            task_id=task_id,
            file_name=file_name,
            file_hash=file_hash,
            duration=1.0,
            segments=[
                TranscriptionSegment(
                    start_time=0.0, end_time=1.0, text=text, speaker="Speaker1"
                )
            ],
            speakers=["Speaker1"],
            processing_time=0.01,
        )
        return (result, {"asr_text": text, "engine": "fake"})

    fake.transcribe = fake_transcribe
    return fake


# ==================== fixtures ====================

@pytest.fixture
async def smoke_env(tmp_path):
    """临时 db / dirs + 重置 db_manager + 起独立 TaskManager + cleanup"""
    # 1. 临时目录
    db_path = tmp_path / "smoke.db"
    upload_dir = tmp_path / "uploads"
    temp_dir = tmp_path / "temp"
    upload_dir.mkdir()
    temp_dir.mkdir()

    # 2. 切 config 指向临时路径
    orig_db_path = config.database.path
    orig_upload = config.server.upload_dir
    orig_temp = config.server.temp_dir
    orig_delete_after = config.transcription.delete_after_transcription
    config.database.path = str(db_path)
    config.server.upload_dir = str(upload_dir)
    config.server.temp_dir = str(temp_dir)
    config.transcription.delete_after_transcription = False  # 留文件方便排错

    # 3. 替换全局 db_manager 单例(注意: 必须同时改所有 import 了它的 module namespace,
    #    不只 db_mod, task_manager 顶层 `from src.core.database import db_manager`
    #    已经把名字绑到自己 namespace, 改 db_mod 不会传递)
    import src.core.database as db_mod
    import src.core.task_manager as tm_mod
    orig_db_mgr_in_db_mod = db_mod.db_manager
    orig_db_mgr_in_tm_mod = tm_mod.db_manager
    new_db = DatabaseManager(db_path=str(db_path))
    await new_db.init_db()
    db_mod.db_manager = new_db
    tm_mod.db_manager = new_db

    # 4. 独立 TaskManager (用 config 临时值, 启 worker)
    mgr = TaskManager()
    await mgr.start()

    # 5. 替换 task_manager 模块单例(其它模块 import 进来的也要同步替换)
    orig_tm = tm_mod.task_manager
    tm_mod.task_manager = mgr

    try:
        yield mgr, new_db
    finally:
        await mgr.stop()
        tm_mod.task_manager = orig_tm
        db_mod.db_manager = orig_db_mgr_in_db_mod
        tm_mod.db_manager = orig_db_mgr_in_tm_mod
        config.database.path = orig_db_path
        config.server.upload_dir = orig_upload
        config.server.temp_dir = orig_temp
        config.transcription.delete_after_transcription = orig_delete_after


@pytest.fixture
def server_engine_funasr():
    orig = config.transcription.default_engine
    config.transcription.default_engine = "funasr"
    yield
    config.transcription.default_engine = orig


@pytest.fixture
def server_engine_qwen3():
    orig = config.transcription.default_engine
    config.transcription.default_engine = "qwen3"
    yield
    config.transcription.default_engine = orig


async def _wait_for_task(mgr, task_id, timeout=5.0):
    """轮询等 task COMPLETED / FAILED, 防死等"""
    interval = 0.05
    elapsed = 0.0
    while elapsed < timeout:
        task = mgr.get_task(task_id)
        if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return task
        await asyncio.sleep(interval)
        elapsed += interval
    raise AssertionError(f"task {task_id} 未在 {timeout}s 内完成, 当前 status={task.status if task else 'None'}")


# ==================== 测试用例 ====================

class TestSmokeHappyPath:
    """worker 完整链路: create → submit → resolve_transcriber → transcribe → save"""

    @pytest.mark.asyncio
    async def test_funasr_engine_end_to_end(self, smoke_env, server_engine_funasr, tmp_path):
        mgr, db = smoke_env
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 100)
        fake = _make_fake_transcriber(text="第一遍", file_hash="h-funasr-1")

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake):
            req = FileUploadRequest(
                file_name="a.wav", file_size=100, file_hash="h-funasr-1", output_format="json"
            )
            task = await mgr.create_task(req, task_id="ts-fn-1")
            await mgr.submit_task("ts-fn-1", str(audio))
            await _wait_for_task(mgr, "ts-fn-1")

        assert task.status == TaskStatus.COMPLETED, f"status={task.status} error={task.error}"
        assert task.engine == "funasr"
        assert task.result is not None
        assert task.result.segments[0].text == "第一遍"
        assert task.result.speakers == ["Speaker1"]

    @pytest.mark.asyncio
    async def test_qwen3_engine_end_to_end(self, smoke_env, server_engine_qwen3, tmp_path):
        mgr, db = smoke_env
        audio = tmp_path / "b.wav"
        audio.write_bytes(b"\x00" * 100)
        fake = _make_fake_transcriber(text="千问引擎", file_hash="h-qwen3-1")

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake):
            req = FileUploadRequest(
                file_name="b.wav", file_size=100, file_hash="h-qwen3-1", output_format="json"
            )
            task = await mgr.create_task(req, task_id="ts-qw-1")
            await mgr.submit_task("ts-qw-1", str(audio))
            await _wait_for_task(mgr, "ts-qw-1")

        assert task.status == TaskStatus.COMPLETED
        assert task.engine == "qwen3"
        assert task.result.segments[0].text == "千问引擎"


class TestSmokeCacheHit:
    """同 hash 重复请求 → cache 命中, fake transcribe 只被调一次"""

    @pytest.mark.asyncio
    async def test_same_engine_same_hash_second_request_hits_cache(
        self, smoke_env, server_engine_funasr, tmp_path
    ):
        mgr, db = smoke_env
        audio = tmp_path / "c.wav"
        audio.write_bytes(b"\x00" * 100)

        call_counter = {"n": 0}
        fake = _make_fake_transcriber(text="缓存内容", file_hash="h-cache-1")
        orig_transcribe = fake.transcribe

        async def counting_transcribe(*a, **kw):
            call_counter["n"] += 1
            return await orig_transcribe(*a, **kw)
        fake.transcribe = counting_transcribe

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake):
            # 第一次跑: 真转录
            req1 = FileUploadRequest(
                file_name="c.wav", file_size=100, file_hash="h-cache-1", output_format="json"
            )
            await mgr.create_task(req1, task_id="ts-c-1")
            await mgr.submit_task("ts-c-1", str(audio))
            await _wait_for_task(mgr, "ts-c-1")

            assert call_counter["n"] == 1, "首次应触发真转录"

            # 第二次同 hash, 同 engine: 应命中 cache, fake 不再调
            audio2 = tmp_path / "c2.wav"
            audio2.write_bytes(b"\x00" * 100)
            req2 = FileUploadRequest(
                file_name="c2.wav", file_size=100, file_hash="h-cache-1", output_format="json"
            )
            await mgr.create_task(req2, task_id="ts-c-2")
            await mgr.submit_task("ts-c-2", str(audio2))
            await _wait_for_task(mgr, "ts-c-2")

            assert call_counter["n"] == 1, f"第二次应命中 cache, 实际调用 {call_counter['n']} 次"
            task2 = mgr.get_task("ts-c-2")
            assert task2.status == TaskStatus.COMPLETED
            assert task2.result.segments[0].text == "缓存内容"

    @pytest.mark.asyncio
    async def test_same_hash_concurrent_submission_both_run_no_dedup(
        self, smoke_env, server_engine_funasr, tmp_path
    ):
        """[现状回归] 同 hash 并发提交无 in-flight dedup, 两 task 都真转录.

        背景: cache lookup 在 submit_task 串行查 db, 没有 by-hash 的
        in-flight task 合并机制. 两个并发请求查 cache 都 miss 时, 两个
        都会真跑转录, 浪费一份算力 (正确性不影响, 后写覆盖先写, 内容相同).

        此 test 显式记录现状: 若未来加 in-flight dedup, 此 assert 会红,
        作为 trigger 提醒维护者把断言改为 == 1.

        触发 race 的关键: fake transcribe 含 await asyncio.sleep, 让两 task
        真的并发 (而不是 A 跑完写 cache → B 才看 cache → 命中).
        """
        mgr, db = smoke_env
        audio1 = tmp_path / "race-a.wav"
        audio2 = tmp_path / "race-b.wav"
        audio1.write_bytes(b"\x00" * 100)
        audio2.write_bytes(b"\x00" * 100)

        call_counter = {"n": 0}
        fake = _make_fake_transcriber(text="并发同 hash", file_hash="h-race-1")
        orig = fake.transcribe

        async def slow_counting_transcribe(*a, **kw):
            call_counter["n"] += 1
            # 模拟真转录耗时, 让两 task 真的并发 (不是 A 完了 B 才开始)
            await asyncio.sleep(0.2)
            return await orig(*a, **kw)
        fake.transcribe = slow_counting_transcribe

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake):
            req1 = FileUploadRequest(
                file_name="race-a.wav", file_size=100, file_hash="h-race-1", output_format="json"
            )
            req2 = FileUploadRequest(
                file_name="race-b.wav", file_size=100, file_hash="h-race-1", output_format="json"
            )
            await mgr.create_task(req1, task_id="ts-race-1")
            await mgr.create_task(req2, task_id="ts-race-2")
            # 几乎同时 submit, 让两个都在 A 写 cache 前查 cache
            await asyncio.gather(
                mgr.submit_task("ts-race-1", str(audio1)),
                mgr.submit_task("ts-race-2", str(audio2)),
            )
            await _wait_for_task(mgr, "ts-race-1")
            await _wait_for_task(mgr, "ts-race-2")

        # 现状回归: 两次都跑了 (no dedup)
        assert call_counter["n"] == 2, (
            f"现状: 同 hash 并发应都真转录 (no in-flight dedup), 实际 {call_counter['n']} 次. "
            "若 == 1, 说明加了 in-flight dedup, 请更新此 test 断言改为 == 1."
        )
        t1 = mgr.get_task("ts-race-1")
        t2 = mgr.get_task("ts-race-2")
        assert t1.status == TaskStatus.COMPLETED
        assert t2.status == TaskStatus.COMPLETED
        # 内容相同 (同 hash, 同 fake)
        assert t1.result.segments[0].text == "并发同 hash"
        assert t2.result.segments[0].text == "并发同 hash"

    @pytest.mark.asyncio
    async def test_cross_engine_cache_hit_default(self, smoke_env, tmp_path):
        """server=funasr 留缓存 → 切 server=qwen3 同 hash → 跨引擎命中, fake 不被调"""
        mgr, db = smoke_env
        audio = tmp_path / "d.wav"
        audio.write_bytes(b"\x00" * 100)

        # round 1: server=funasr
        config.transcription.default_engine = "funasr"
        call_counter = {"funasr": 0, "qwen3": 0}
        fake_funasr = _make_fake_transcriber(text="funasr 缓存", file_hash="h-cross-1")
        orig_fn = fake_funasr.transcribe

        async def fn_counting(*a, **kw):
            call_counter["funasr"] += 1
            return await orig_fn(*a, **kw)
        fake_funasr.transcribe = fn_counting

        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_funasr):
            req = FileUploadRequest(
                file_name="d.wav", file_size=100, file_hash="h-cross-1", output_format="json"
            )
            await mgr.create_task(req, task_id="ts-cross-1")
            await mgr.submit_task("ts-cross-1", str(audio))
            await _wait_for_task(mgr, "ts-cross-1")
        assert call_counter["funasr"] == 1

        # round 2: 切 server=qwen3, 同 hash 上传
        config.transcription.default_engine = "qwen3"
        fake_qwen3 = _make_fake_transcriber(text="qwen3 不应被调", file_hash="h-cross-1")
        orig_qw = fake_qwen3.transcribe

        async def qw_counting(*a, **kw):
            call_counter["qwen3"] += 1
            return await orig_qw(*a, **kw)
        fake_qwen3.transcribe = qw_counting

        audio2 = tmp_path / "d2.wav"
        audio2.write_bytes(b"\x00" * 100)
        with patch("src.core.transcriber_dispatch.resolve_transcriber", return_value=fake_qwen3):
            req2 = FileUploadRequest(
                file_name="d2.wav", file_size=100, file_hash="h-cross-1", output_format="json"
            )
            await mgr.create_task(req2, task_id="ts-cross-2")
            await mgr.submit_task("ts-cross-2", str(audio2))
            await _wait_for_task(mgr, "ts-cross-2")

        assert call_counter["qwen3"] == 0, \
            f"跨引擎缓存命中, qwen3 不应被调用, 实际 {call_counter['qwen3']} 次"
        task2 = mgr.get_task("ts-cross-2")
        assert task2.status == TaskStatus.COMPLETED
        # 跨引擎命中返回的 segments 来自 funasr 缓存
        assert task2.result.segments[0].text == "funasr 缓存"


class TestSmokeEngineMismatchReject:
    """create_task strict validate: request.engine != server engine → 立即 ValueError"""

    @pytest.mark.asyncio
    async def test_request_qwen3_when_server_funasr_rejects(
        self, smoke_env, server_engine_funasr
    ):
        mgr, _ = smoke_env
        req = FileUploadRequest(
            file_name="x.wav", file_size=100, file_hash="h-rej-1",
            engine="qwen3", output_format="json"
        )
        with pytest.raises(ValueError) as exc:
            await mgr.create_task(req, task_id="ts-rej-1")
        msg = str(exc.value)
        assert "funasr" in msg and "qwen3" in msg
        # 没创建 task
        assert mgr.get_task("ts-rej-1") is None
