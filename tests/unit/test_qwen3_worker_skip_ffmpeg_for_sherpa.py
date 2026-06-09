"""Qwen3 worker_process: sherpa-supported 格式跳过 ffmpeg 转换 (PR 修 over-detect).

调研结论 (docs/开发/archive/spk-over-detect-归因调研结果.md):
- cd578a8 worker 加 ffmpeg convert_to_wav 修了 m4a 不能被 sherpa 直读的 prod bug,
  但 mp3 也被卷进去转码, 转码改变了 audio 字节, 触发 sherpa diarize FastClustering over-detect.
- mp3/flac/ogg 这类格式 sherpa diarize 通过 `_load_audio_mono_16k` (qwen3/diarize.py) 的
  librosa fallback 可以直接读, 不必走 ffmpeg.
- ASR vendor 历史上能跑 mp3 (audio_149min.mp3 经 PoC 多次验证), 同样不必转.

新契约: worker 只对 sherpa 真正读不了的格式 (m4a/aac/mp4/mov/webm 等) 走 ffmpeg.
sherpa-supported (wav/flac/ogg/mp3/opus) 一律跳过 convert_to_wav, 原文件透传给 transcribe.

修复前: mp3/flac/ogg 都被 convert_to_wav → 触发 over-detect 风险
修复后: mp3/flac/ogg 不调 convert_to_wav, audio_path 透传给 transcribe
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_transcriber_capturing(captured_path: dict):
    """fake transcriber, 把 transcribe 实际拿到的 audio_path 记录到 captured_path['actual']"""
    fake = MagicMock()

    async def fake_transcribe(audio_path, task_id, progress_callback=None, output_format="json", language=None):
        captured_path["actual"] = audio_path
        from src.models.schemas import TranscriptionResult, TranscriptionSegment
        return (
            TranscriptionResult(
                task_id=task_id,
                file_name=Path(audio_path).name,
                file_hash="h",
                duration=1.0,
                segments=[
                    TranscriptionSegment(
                        start_time=0.0, end_time=1.0, text="x", speaker="Speaker1"
                    )
                ],
                speakers=["Speaker1"],
                processing_time=0.01,
            ),
            {"engine": "qwen3"},
        )

    fake.transcribe = fake_transcribe
    return fake


def _write_task_file(tmp_path: Path, worker_id: int, task_id: str, audio_ext: str):
    """工厂: 写一份 task JSON, audio_path 以 audio_ext 结尾, 返回 (task_file, audio_path)"""
    audio_path = f"/temp/tasks_qwen3/abc-uuid{audio_ext}"
    task_file = tmp_path / f"worker_{worker_id}_{task_id}.task"
    task = {
        "task_id": task_id,
        "audio_path": audio_path,
        "source_audio_path": f"/orig/upload/my{audio_ext}",
        "output_format": "json",
    }
    task_file.write_text(json.dumps(task), encoding="utf-8")
    return task_file, audio_path


# ==================== 跳过 ffmpeg 的格式 (sherpa libsndfile / librosa 能读) ====================


class TestSherpaSupportedFormatsSkipFfmpeg:
    """wav/flac/ogg/mp3/opus: 不应该调 convert_to_wav, audio_path 原样透传给 transcribe"""

    @pytest.mark.parametrize(
        "audio_ext",
        [".wav", ".mp3", ".flac", ".ogg", ".opus"],
    )
    def test_sherpa_supported_extension_skips_convert_to_wav(self, tmp_path, audio_ext):
        from src.core import qwen3_worker_process as wp

        task_id = f"t-skip{audio_ext.replace('.', '-')}"
        task_file, audio_path = _write_task_file(tmp_path, 0, task_id, audio_ext)

        captured = {}
        fake = _make_fake_transcriber_capturing(captured)

        with patch("src.utils.file_utils.convert_to_wav") as mock_conv:
            wp.process_task(
                worker_id=0, transcriber=fake,
                task_file=str(task_file), task_dir=str(tmp_path),
            )

        # 1. convert_to_wav 不应被调用
        mock_conv.assert_not_called()

        # 2. transcribe 拿到的是原始 audio_path, 不是 .converted.wav
        assert captured["actual"] == audio_path, (
            f"transcribe 应拿到原始 {audio_path}, 实际 {captured.get('actual')}"
        )


# ==================== 非 sherpa-supported 格式仍然 ffmpeg 转 wav ====================


class TestNonSherpaFormatsStillConvert:
    """m4a/aac/mp4/mov/webm: sherpa 读不了, 必须 ffmpeg 转 wav (现有契约保留)"""

    @pytest.mark.parametrize(
        "audio_ext",
        [".m4a", ".aac", ".mp4", ".mov", ".webm"],
    )
    def test_non_sherpa_extension_triggers_convert_to_wav(self, tmp_path, audio_ext):
        from src.core import qwen3_worker_process as wp

        task_id = f"t-conv{audio_ext.replace('.', '-')}"
        task_file, audio_path = _write_task_file(tmp_path, 0, task_id, audio_ext)

        captured = {}
        fake = _make_fake_transcriber_capturing(captured)

        with patch("src.utils.file_utils.convert_to_wav") as mock_conv:
            mock_conv.side_effect = lambda inp, output_path=None: output_path
            wp.process_task(
                worker_id=0, transcriber=fake,
                task_file=str(task_file), task_dir=str(tmp_path),
            )

        # convert_to_wav 必须被调用, 输入是原始 audio
        mock_conv.assert_called_once()
        called_input = mock_conv.call_args.args[0]
        assert called_input == audio_path

        # transcribe 拿到的是转换后的 .converted.wav
        assert captured["actual"].endswith(".converted.wav")


# ==================== 边界 case: 大小写 / 多重后缀 ====================


class TestCaseInsensitiveExtension:
    """audio_path 后缀大小写不敏感: AUDIO.MP3 同样跳过 ffmpeg"""

    def test_uppercase_mp3_extension_skips_convert(self, tmp_path):
        from src.core import qwen3_worker_process as wp

        task_id = "t-upper-mp3"
        task_file, audio_path = _write_task_file(tmp_path, 0, task_id, ".MP3")

        captured = {}
        fake = _make_fake_transcriber_capturing(captured)

        with patch("src.utils.file_utils.convert_to_wav") as mock_conv:
            wp.process_task(
                worker_id=0, transcriber=fake,
                task_file=str(task_file), task_dir=str(tmp_path),
            )

        mock_conv.assert_not_called()
        assert captured["actual"].endswith(".MP3")
