"""异步轮询契约 (TODOS #20) — T2/T3: task_status_batch handler

覆盖（红→绿）：
1. 空/缺 task_ids → error
2. 超 50 截断 + warn（控帧大小）
3. processing/pending → 小帧（result/srt_content/error 全 None）
4. completed-JSON → 内联 result
5. completed-SRT → 内联 srt_content（非 result，codex #11）
6. failed/timed_out/cancelled → 终态 + error（codex #10）
7. expired → error task_expired；not_found → error task_not_found（不整批失败）
8. 混合批 → 逐 id 各自正确
9. _handle_message 路由 task_status_batch
10. 读 status+result 同步无 await（原子性：build item 是 sync 方法，钉 task_manager.py:499-500 不变量）
11. 上传协议不变回归（无 wait 字段、上传消息类型集合不变）

mock task_manager 隔离。
"""
import inspect
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from src.api.websocket_handler import WebSocketHandler
from src.models.schemas import (
    TaskStatus,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTask,
)


@pytest.fixture
def handler():
    return WebSocketHandler()


@pytest.fixture
def fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.remote_address = ("127.0.0.1", 1234)
    return ws


def make_task(task_id, status, output_format="json", result=None, srt_content=None,
              error=None, progress=0.0):
    t = TranscriptionTask(
        task_id=task_id, file_name="a.wav", file_path="/tmp/a",
        file_size=10, file_hash="h", engine="qwen3",
        status=status, output_format=output_format, progress=progress,
    )
    t.result = result
    t.srt_content = srt_content
    t.error = error
    return t


def make_result(task_id):
    return TranscriptionResult(
        task_id=task_id, file_name="a.wav", file_hash="h", duration=1.0,
        segments=[TranscriptionSegment(start_time=0.0, end_time=1.0, text="hi", speaker="Speaker1")],
        speakers=["Speaker1"], processing_time=0.5,
    )


def last_batch_payload(send_msg_mock):
    """取最后一条 task_status_batch 消息的 data dict"""
    calls = [c for c in send_msg_mock.call_args_list
             if len(c.args) >= 2 and c.args[1] == "task_status_batch"]
    assert calls, "未发出 task_status_batch 消息"
    return calls[-1].args[2]


def items_by_id(payload):
    return {it["task_id"]: it for it in payload["items"]}


class TestBadInput:
    @pytest.mark.asyncio
    async def test_empty_task_ids_errors(self, handler, fake_ws):
        with patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_task_status_batch(fake_ws, [])
            assert se.called
            sm.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_task_ids_errors(self, handler, fake_ws):
        with patch.object(handler, "_send_message", new=AsyncMock()), \
             patch.object(handler, "_send_error", new=AsyncMock()) as se:
            await handler._handle_task_status_batch(fake_ws, None)
            assert se.called


class TestTruncation:
    @pytest.mark.asyncio
    async def test_over_50_truncated_and_warns(self, handler, fake_ws):
        ids = [f"t{i}" for i in range(80)]
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm, \
             patch("src.api.websocket_handler.logger") as lg:
            tm.get_task = MagicMock(return_value=None)
            tm.was_evicted = MagicMock(return_value=False)
            await handler._handle_task_status_batch(fake_ws, ids)
            payload = last_batch_payload(sm)
            assert len(payload["items"]) == 50
            assert lg.warning.called


class TestPerStatus:
    @pytest.mark.asyncio
    async def test_processing_small_frame(self, handler, fake_ws):
        task = make_task("p1", TaskStatus.PROCESSING, progress=42.0)
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(return_value=task)
            await handler._handle_task_status_batch(fake_ws, ["p1"])
            it = items_by_id(last_batch_payload(sm))["p1"]
            assert it["status"] == TaskStatus.PROCESSING.value
            assert it["progress"] == 42.0
            assert it["result"] is None
            assert it["srt_content"] is None
            assert it["error"] is None

    @pytest.mark.asyncio
    async def test_completed_json_inlines_result(self, handler, fake_ws):
        task = make_task("c1", TaskStatus.COMPLETED, output_format="json",
                         result=make_result("c1"), progress=100.0)
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(return_value=task)
            await handler._handle_task_status_batch(fake_ws, ["c1"])
            it = items_by_id(last_batch_payload(sm))["c1"]
            assert it["status"] == TaskStatus.COMPLETED.value
            assert it["result"] is not None
            assert it["result"]["segments"][0]["text"] == "hi"
            assert it["srt_content"] is None

    @pytest.mark.asyncio
    async def test_completed_srt_inlines_srt_content(self, handler, fake_ws):
        srt = "1\n00:00:00,000 --> 00:00:01,000\nhi\n"
        task = make_task("s1", TaskStatus.COMPLETED, output_format="srt",
                         result=make_result("s1"), srt_content=srt, progress=100.0)
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(return_value=task)
            await handler._handle_task_status_batch(fake_ws, ["s1"])
            it = items_by_id(last_batch_payload(sm))["s1"]
            assert it["status"] == TaskStatus.COMPLETED.value
            assert it["srt_content"] == srt
            # codex #11: SRT 不塞进 result 字段
            assert it["result"] is None

    @pytest.mark.asyncio
    async def test_terminal_states_carry_error(self, handler, fake_ws):
        tasks = {
            "f1": make_task("f1", TaskStatus.FAILED, error="模型崩了"),
            "to1": make_task("to1", TaskStatus.TIMED_OUT, error="超时"),
            "cx1": make_task("cx1", TaskStatus.CANCELLED, error=None),
        }
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(side_effect=lambda tid: tasks.get(tid))
            await handler._handle_task_status_batch(fake_ws, list(tasks))
            its = items_by_id(last_batch_payload(sm))
            assert its["f1"]["status"] == TaskStatus.FAILED.value
            assert its["f1"]["error"] == "模型崩了"
            assert its["to1"]["status"] == TaskStatus.TIMED_OUT.value
            assert its["cx1"]["status"] == TaskStatus.CANCELLED.value


class TestPollMiss:
    @pytest.mark.asyncio
    async def test_expired_and_not_found(self, handler, fake_ws):
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(return_value=None)
            tm.was_evicted = MagicMock(side_effect=lambda tid: tid == "gone")
            await handler._handle_task_status_batch(fake_ws, ["gone", "never"])
            its = items_by_id(last_batch_payload(sm))
            assert its["gone"]["status"] is None
            assert its["gone"]["error"] == "task_expired"
            assert its["never"]["status"] is None
            assert its["never"]["error"] == "task_not_found"


class TestMixedBatch:
    @pytest.mark.asyncio
    async def test_mixed_batch_per_id(self, handler, fake_ws):
        done = make_task("d", TaskStatus.COMPLETED, result=make_result("d"), progress=100.0)
        proc = make_task("p", TaskStatus.PROCESSING, progress=10.0)
        store = {"d": done, "p": proc}
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(side_effect=lambda tid: store.get(tid))
            tm.was_evicted = MagicMock(return_value=False)
            await handler._handle_task_status_batch(fake_ws, ["d", "p", "x"])
            its = items_by_id(last_batch_payload(sm))
            assert its["d"]["result"] is not None
            assert its["p"]["result"] is None and its["p"]["status"] == TaskStatus.PROCESSING.value
            assert its["x"]["error"] == "task_not_found"


class TestAtomicity:
    def test_build_item_is_sync_no_await(self, handler):
        """钉 task_manager.py:499-500 不变量：读 status+result 同步块、中间不 await，
        否则会撞 COMPLETED 翻转在 result 组装后的窗口返回 completed+null。
        build item 必须是 sync 方法（无 await 即不可能跨协程切换）。"""
        assert not inspect.iscoroutinefunction(handler._build_task_status_batch_item)

    @pytest.mark.asyncio
    async def test_processing_never_completed_null(self, handler, fake_ws):
        """PROCESSING 任务（result 尚未组装）→ 永不返回 completed+null"""
        task = make_task("w", TaskStatus.PROCESSING, result=None, progress=50.0)
        with patch("src.core.task_manager.task_manager") as tm, \
             patch.object(handler, "_send_message", new=AsyncMock()) as sm:
            tm.get_task = MagicMock(return_value=task)
            await handler._handle_task_status_batch(fake_ws, ["w"])
            it = items_by_id(last_batch_payload(sm))["w"]
            assert it["status"] != TaskStatus.COMPLETED.value
            assert it["result"] is None


class TestRouting:
    @pytest.mark.asyncio
    async def test_handle_message_routes_batch(self, handler, fake_ws):
        with patch.object(handler, "_handle_task_status_batch", new=AsyncMock()) as bh:
            await handler._handle_message(
                fake_ws, "c1",
                {"type": "task_status_batch", "data": {"task_ids": ["a", "b"]}},
            )
            bh.assert_awaited_once()
            # task_ids 透传
            assert bh.call_args.args[-1] == ["a", "b"]


class TestUploadProtocolUnchanged:
    def test_upload_message_types_unchanged(self):
        """钉死：新增 batch 不改任何上传/单查消息分支（部署安全命脉）"""
        import src.api.websocket_handler as wh
        src = inspect.getsource(wh.WebSocketHandler._handle_message)
        for mt in ["ping", "upload_request", "upload_data", "upload_chunk",
                   "finalize_upload", "task_status", "cancel_task"]:
            assert f'"{mt}"' in src
        # 无 wait 开关泄漏到上传请求
        from src.models.schemas import FileUploadRequest
        assert "wait" not in FileUploadRequest.model_fields
