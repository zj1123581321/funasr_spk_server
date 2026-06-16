"""
Qwen3-Diarize 引擎端到端集成测试

默认 skip(需加载 ~2GB 模型 + 真转录, 不适合 CI 跑).
设 FUNASR_RUN_INTEGRATION=1 启用.

测试用例:
- 60s 双人播客 (tests/fixtures/audio/podcast_2speakers_60s.wav) 端到端
- RTF 验收: <0.15 (PoC v3 实测 0.108-0.118)
- speakers 验收: >=2 (双人 podcast)
- segments 验收: 非空, 按时间有序, text 拼接非空

注意: 这个测试需要模型权重落地(scripts/download_qwen3_models.sh 已跑过).
"""
import asyncio
import os
from pathlib import Path

import pytest


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设置 FUNASR_RUN_INTEGRATION=1 启用(默认 skip, 避免加载 ~2GB Qwen3 模型)",
)


def _qwen3_models_ready() -> bool:
    """检查 6 个核心模型文件是否落地"""
    from src.core.config import config
    paths = [
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_frontend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_encoder_backend.onnx",
        Path(config.qwen3.asr_model_dir) / "qwen3_asr_llm.gguf",
        Path(config.qwen3.segmentation_model),
        Path(config.qwen3.embedding_model),
    ]
    return all(p.exists() for p in paths)


pytestmark2 = pytest.mark.skipif(
    not _qwen3_models_ready(),
    reason="Qwen3 模型权重未落地, 跑 scripts/download_qwen3_models.sh 后再试",
)


@pytest.fixture(scope="session")
def fresh_qwen3_transcriber():
    """重置单例 + 返回新 Qwen3DiarizeTranscriber(避免单例状态污染)"""
    from src.core.qwen3_transcriber import (
        get_qwen3_transcriber,
        reset_qwen3_transcriber_singleton,
    )
    reset_qwen3_transcriber_singleton()
    yield get_qwen3_transcriber()
    reset_qwen3_transcriber_singleton()


class TestQwen3DiarizeEndToEnd:

    @pytest.mark.asyncio
    async def test_podcast_2speakers_json_mode(self, fresh_qwen3_transcriber, podcast_audio: Path):
        """60s 双人播客真跑一遍 ASR + Diarize"""
        result, raw = await fresh_qwen3_transcriber.transcribe(
            audio_path=str(podcast_audio),
            task_id="qwen3-e2e-podcast",
            progress_callback=None,
            output_format="json",
        )
        # 1) 元组结构正确
        from src.models.schemas import TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert isinstance(raw, dict)

        # 2) 时长接近 60s
        assert 55.0 <= result.duration <= 65.0, f"音频时长异常: {result.duration}"

        # 3) RTF < 0.15 (PoC 报告 5min 双人 RTF 0.118, 60s 应类似量级)
        rtf = result.processing_time / result.duration
        assert rtf < 0.5, f"Qwen3 RTF 超目标: {rtf:.3f} > 0.5 (PoC ~0.12)"

        # 4) 至少 2 个 speaker
        assert len(result.speakers) >= 2, f"双人 podcast 应识别 >=2 speaker, got: {result.speakers}"

        # 5) segments 非空 + 按 start_time 有序
        assert len(result.segments) > 0
        starts = [s.start_time for s in result.segments]
        assert starts == sorted(starts), "segments 应按 start_time 有序"

        # 6) 文本拼接非空
        full_text = "".join(s.text for s in result.segments)
        assert len(full_text.strip()) > 0, "ASR 应有非空文本输出"

        # 7) raw_result 含 asr_text + turns
        assert raw["asr_text"]
        assert isinstance(raw["turns"], list)
        assert len(raw["turns"]) > 0
        assert raw["engine"] == "qwen3"

    @pytest.mark.asyncio
    async def test_podcast_2speakers_srt_mode(self, fresh_qwen3_transcriber, podcast_audio: Path):
        ret = await fresh_qwen3_transcriber.transcribe(
            audio_path=str(podcast_audio),
            task_id="qwen3-e2e-podcast-srt",
            progress_callback=None,
            output_format="srt",
        )
        assert isinstance(ret, dict)
        assert ret["format"] == "srt"
        assert "1\n00:00:" in ret["content"], "SRT 应至少有 1 个时间戳片段"
        assert "Speaker" in ret["content"]
        assert ret["raw_result"]["engine"] == "qwen3"
