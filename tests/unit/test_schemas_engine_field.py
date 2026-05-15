"""
PR1 — schemas engine 字段测试

验证 FileUploadRequest 和 TranscriptionTask 支持 engine 字段，
且不带 engine 字段的旧 client 请求仍能正常解析（向后兼容）。
"""
import pytest
from pydantic import ValidationError

from src.models.schemas import FileUploadRequest, TranscriptionTask, TaskStatus


class TestFileUploadRequestEngineField:
    def test_accepts_engine_field(self):
        """upload request 能携带 engine 字段"""
        req = FileUploadRequest(
            file_name="test.wav",
            file_size=1024,
            file_hash="abc123",
            engine="qwen3",
        )
        assert req.engine == "qwen3"

    def test_engine_is_optional_default_none(self):
        """不带 engine 字段时默认 None，由 task_manager 解析为 default_engine"""
        req = FileUploadRequest(
            file_name="test.wav",
            file_size=1024,
            file_hash="abc123",
        )
        assert req.engine is None

    def test_engine_accepts_arbitrary_string(self):
        """engine 字段不在 schema 层做白名单校验（让 dispatch 层做）"""
        # 这样未来加新引擎不需要改 schema
        req = FileUploadRequest(
            file_name="test.wav",
            file_size=1024,
            file_hash="abc123",
            engine="experimental_v2",
        )
        assert req.engine == "experimental_v2"


class TestTranscriptionTaskEngineField:
    def test_task_carries_engine(self):
        """task 实例化时必须能携带 engine（由 task_manager 解析后填入）"""
        task = TranscriptionTask(
            task_id="t-1",
            file_name="x.wav",
            file_path="/tmp/x.wav",
            file_size=100,
            file_hash="h",
            engine="funasr",
        )
        assert task.engine == "funasr"

    def test_task_engine_default_funasr(self):
        """task engine 默认 funasr（与 default_engine 一致，保证向后兼容）"""
        task = TranscriptionTask(
            task_id="t-1",
            file_name="x.wav",
            file_path="/tmp/x.wav",
            file_size=100,
            file_hash="h",
        )
        assert task.engine == "funasr"
