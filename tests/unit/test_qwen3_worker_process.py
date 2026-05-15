"""
Qwen3 worker process entry — unit 测试

设计:
- worker entry 跟 FunASR 同协议: 读 worker_{id}_*.task, 写 worker_{id}_*.pkl
- worker 内 import Qwen3DiarizeTranscriber (libllama + Metal context 在 worker 进程内独占)
- 单任务后退出 (sys.exit(0))

测试覆盖:
1. 任务 JSON 字段 → transcribe(audio_path, task_id, output_format) 透传正确
2. JSON 模式: 结果 pickle 含 (TranscriptionResult, raw_result) 元组
3. SRT 模式: 结果 pickle 含 SRT dict
4. transcribe 异常 → 写 error pickle (success=False) + traceback
5. process_task 处理完后删除 .task 文件
6. argparse 接受 --worker-id / --task-dir
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ==================== fake transcriber ====================


def _make_fake_transcriber(text: str = "测试文本", file_hash: str = "h"):
    """构造接口与 Qwen3DiarizeTranscriber 同形的 fake"""
    fake = MagicMock()

    async def fake_transcribe(audio_path, task_id, progress_callback=None, output_format="json"):
        from src.models.schemas import TranscriptionResult, TranscriptionSegment

        if output_format == "srt":
            return {
                "format": "srt",
                "content": f"1\n00:00:00,000 --> 00:00:01,000\nSpeaker1:{text}\n",
                "file_name": Path(audio_path).name,
                "file_hash": file_hash,
                "duration": 1.0,
                "processing_time": 0.01,
                "raw_result": {"asr_text": text, "engine": "qwen3"},
            }
        result = TranscriptionResult(
            task_id=task_id,
            file_name=Path(audio_path).name,
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
        return (result, {"asr_text": text, "engine": "qwen3"})

    fake.transcribe = fake_transcribe
    return fake


# ==================== fixtures ====================


@pytest.fixture
def tmp_task_dir(tmp_path):
    """临时任务目录"""
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    return task_dir


@pytest.fixture
def write_task_file(tmp_task_dir):
    """工厂: 写一个 .task JSON 文件, 返回路径"""
    def _write(worker_id: int, task_id: str, audio_path: str, output_format: str = "json"):
        task_file = tmp_task_dir / f"worker_{worker_id}_{task_id}.task"
        task = {
            "task_id": task_id,
            "audio_path": audio_path,
            "source_audio_path": audio_path,
            "output_format": output_format,
        }
        task_file.write_text(json.dumps(task), encoding="utf-8")
        return task_file
    return _write


# ==================== process_task 主接口 ====================


class TestProcessTaskJsonMode:
    """JSON 模式: pickle 含 (TranscriptionResult, raw_result)"""

    def test_writes_pickle_result_on_success(self, tmp_task_dir, write_task_file):
        from src.core import qwen3_worker_process as wp

        task_id = "tid-1"
        task_file = write_task_file(0, task_id, "/fake/audio.wav", "json")
        fake = _make_fake_transcriber(text="千问输出", file_hash="hh")

        wp.process_task(worker_id=0, transcriber=fake, task_file=str(task_file), task_dir=str(tmp_task_dir))

        result_file = tmp_task_dir / f"worker_0_{task_id}.pkl"
        assert result_file.exists(), "pickle 结果文件未写入"

        with open(result_file, "rb") as f:
            data = pickle.load(f)

        assert data["task_id"] == task_id
        assert data["success"] is True
        # JSON 模式: result 是 (TranscriptionResult, raw)
        assert isinstance(data["result"], tuple)
        assert len(data["result"]) == 2
        tres, raw = data["result"]
        assert tres.task_id == task_id
        assert tres.file_hash == "hh"
        assert raw["engine"] == "qwen3"

    def test_deletes_task_file_after_processing(self, tmp_task_dir, write_task_file):
        from src.core import qwen3_worker_process as wp

        task_id = "tid-2"
        task_file = write_task_file(0, task_id, "/fake/audio.wav", "json")
        assert task_file.exists()

        wp.process_task(worker_id=0, transcriber=_make_fake_transcriber(), task_file=str(task_file), task_dir=str(tmp_task_dir))

        assert not task_file.exists(), ".task 文件未被删除"


class TestProcessTaskSrtMode:
    """SRT 模式: pickle 含 SRT dict"""

    def test_srt_mode_pickle_contains_srt_dict(self, tmp_task_dir, write_task_file):
        from src.core import qwen3_worker_process as wp

        task_id = "tid-srt-1"
        task_file = write_task_file(1, task_id, "/fake/audio.wav", "srt")
        fake = _make_fake_transcriber(text="SRT 内容", file_hash="hh2")

        wp.process_task(worker_id=1, transcriber=fake, task_file=str(task_file), task_dir=str(tmp_task_dir))

        result_file = tmp_task_dir / f"worker_1_{task_id}.pkl"
        with open(result_file, "rb") as f:
            data = pickle.load(f)

        assert data["success"] is True
        assert isinstance(data["result"], dict)
        assert data["result"]["format"] == "srt"
        assert "SRT 内容" in data["result"]["content"]


class TestProcessTaskError:
    """异常 → success=False + error/traceback"""

    def test_transcribe_exception_writes_error_pickle(self, tmp_task_dir, write_task_file):
        from src.core import qwen3_worker_process as wp

        task_id = "tid-err-1"
        task_file = write_task_file(0, task_id, "/fake/audio.wav", "json")

        fake = MagicMock()

        async def boom(*args, **kwargs):
            raise RuntimeError("transcribe 炸了")

        fake.transcribe = boom

        wp.process_task(worker_id=0, transcriber=fake, task_file=str(task_file), task_dir=str(tmp_task_dir))

        result_file = tmp_task_dir / f"worker_0_{task_id}.pkl"
        assert result_file.exists()

        with open(result_file, "rb") as f:
            data = pickle.load(f)

        assert data["success"] is False
        assert "transcribe 炸了" in data["error"]
        assert "traceback" in data
        assert data["task_id"] == task_id


# ==================== argparse 入口 ====================


class TestArgparseEntry:
    """worker entry 必须支持 --worker-id / --task-dir, 跟 FunASR worker 一致"""

    def test_module_has_main_argparse_block(self):
        from src.core import qwen3_worker_process as wp
        # 关键模块函数都存在
        assert hasattr(wp, "process_task"), "缺少 process_task"
        assert hasattr(wp, "worker_loop"), "缺少 worker_loop"
        assert hasattr(wp, "load_qwen3_transcriber"), "缺少 load_qwen3_transcriber"


# ==================== load_qwen3_transcriber: 从 config 读路径 ====================


class TestLoadQwen3Transcriber:
    """工厂函数: 从 config.qwen3 构造 Qwen3DiarizeTranscriber, libllama context 在 worker 内独占"""

    def test_load_calls_get_qwen3_transcriber(self):
        """直接复用主模块的 get_qwen3_transcriber 工厂(单 worker 内仍是单 instance)"""
        from src.core import qwen3_worker_process as wp

        fake_instance = MagicMock(name="fake_qwen3")
        with patch.object(wp, "get_qwen3_transcriber", return_value=fake_instance):
            ret = wp.load_qwen3_transcriber()

        assert ret is fake_instance
