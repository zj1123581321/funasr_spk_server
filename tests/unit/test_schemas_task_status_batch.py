"""异步轮询契约 (TODOS #20) — T1: batch 查询 schema

钉死 TaskStatusBatchResponse / TaskStatusBatchItem 的字段形态：
- item 含 task_id / status(可空, expired/not_found 时) / progress / result? / srt_content? / error?
- COMPLETED-JSON 走 result, COMPLETED-SRT 走 srt_content (codex #11: result 装不下 SRT)
- 终态全集 failed/timed_out/cancelled 带 error (codex #10)
- FileUploadRequest 不加 wait 字段 (codex 战略简化: 上传协议零改动)
"""
from src.models.schemas import (
    TaskStatus,
    TaskStatusBatchItem,
    TaskStatusBatchResponse,
    TranscriptionResult,
)


class TestBatchItemShape:
    def test_processing_item_small_frame(self):
        """PENDING/PROCESSING → result/srt_content/error 全 None（小帧）"""
        item = TaskStatusBatchItem(task_id="t1", status=TaskStatus.PROCESSING, progress=42.0)
        assert item.task_id == "t1"
        assert item.status == TaskStatus.PROCESSING
        assert item.progress == 42.0
        assert item.result is None
        assert item.srt_content is None
        assert item.error is None

    def test_completed_json_carries_result(self):
        result = TranscriptionResult(
            task_id="t2", file_name="a.wav", file_hash="h",
            duration=1.0, segments=[], speakers=[], processing_time=0.5,
        )
        item = TaskStatusBatchItem(
            task_id="t2", status=TaskStatus.COMPLETED, progress=100.0, result=result,
        )
        assert item.result is not None
        assert item.srt_content is None

    def test_completed_srt_carries_srt_content(self):
        item = TaskStatusBatchItem(
            task_id="t3", status=TaskStatus.COMPLETED, progress=100.0,
            srt_content="1\n00:00:00,000 --> 00:00:01,000\nhi\n",
        )
        assert item.srt_content.startswith("1\n")
        assert item.result is None

    def test_failed_item_carries_error(self):
        item = TaskStatusBatchItem(
            task_id="t4", status=TaskStatus.FAILED, error="模型崩了",
        )
        assert item.error == "模型崩了"

    def test_expired_item_status_none(self):
        """expired/not_found → status 可空，用 error 标记（不整批失败）"""
        item = TaskStatusBatchItem(task_id="t5", status=None, error="task_expired")
        assert item.status is None
        assert item.error == "task_expired"


class TestBatchResponse:
    def test_response_holds_items_and_serializes(self):
        resp = TaskStatusBatchResponse(items=[
            TaskStatusBatchItem(task_id="a", status=TaskStatus.PROCESSING, progress=10.0),
            TaskStatusBatchItem(task_id="b", status=None, error="task_not_found"),
        ])
        d = resp.model_dump()
        assert len(d["items"]) == 2
        assert d["items"][0]["task_id"] == "a"
        assert d["items"][1]["error"] == "task_not_found"


class TestUploadProtocolUnchanged:
    def test_file_upload_request_has_no_wait_field(self):
        """codex 战略简化：上传协议零改动，无 wait 开关"""
        from src.models.schemas import FileUploadRequest
        assert "wait" not in FileUploadRequest.model_fields
