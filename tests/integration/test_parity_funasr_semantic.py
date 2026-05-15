"""
PR1 — FunASR semantic parity test

意图：验证 PR1 改动后，FunASR 路径的语义输出和 PR1 前一致。

设计：
- 默认 skip（FunASR 模型 ~2GB，加载 + 转录耗时数分钟，不适合 CI 默认跑）
- 设 FUNASR_RUN_INTEGRATION=1 后才执行
- 首次运行：当 golden 文件不存在 → 跑转录 + 写入 golden（建立 baseline）
- 后续运行：跑转录 + 与 golden 做 semantic 对比

Semantic 算法（修 codex review T4：byte-equal 不现实）：
- 文本：每段 text 严格相等
- 时间窗：start_time / end_time 容差 ±50ms
- speakers：数量相等（标签允许重映射）
- 忽略：created_at / processing_time / task_id 等运行时字段
"""
import asyncio
import json
import os
from pathlib import Path

import pytest

from src.models.schemas import TranscriptionResult


RUN_INTEGRATION = os.getenv("FUNASR_RUN_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="设置 FUNASR_RUN_INTEGRATION=1 环境变量启用（默认 skip，避免加载 ~2GB FunASR 模型）",
)


# 容差：50ms
TIMESTAMP_TOLERANCE_SECONDS = 0.05


def assert_semantic_equal(actual: TranscriptionResult, golden: dict):
    """语义级别对比 — 容忍非语义差异（时间戳轻微波动、运行时字段）"""
    actual_segs = actual.segments
    golden_segs = golden["segments"]
    assert len(actual_segs) == len(golden_segs), \
        f"segment 数量不一致: actual={len(actual_segs)} golden={len(golden_segs)}"

    for i, (a, g) in enumerate(zip(actual_segs, golden_segs)):
        assert a.text == g["text"], f"segment[{i}] text 不一致: {a.text!r} vs {g['text']!r}"
        assert abs(a.start_time - g["start_time"]) <= TIMESTAMP_TOLERANCE_SECONDS, \
            f"segment[{i}] start_time 差超过 {TIMESTAMP_TOLERANCE_SECONDS}s: {a.start_time} vs {g['start_time']}"
        assert abs(a.end_time - g["end_time"]) <= TIMESTAMP_TOLERANCE_SECONDS, \
            f"segment[{i}] end_time 差超过 {TIMESTAMP_TOLERANCE_SECONDS}s: {a.end_time} vs {g['end_time']}"

    assert len(actual.speakers) == len(golden["speakers"]), \
        f"speaker 数量不一致: actual={len(actual.speakers)} golden={len(golden['speakers'])}"


async def _run_funasr_once(audio_path: Path) -> TranscriptionResult:
    """走 PR1 真实路径（resolve_transcriber → FunASREngine → transcribe）跑一次"""
    from src.core.transcriber_dispatch import resolve_transcriber
    transcriber = resolve_transcriber("funasr")
    if not transcriber.is_initialized:
        await transcriber.initialize()
    result = await transcriber.transcribe(
        audio_path=str(audio_path),
        task_id="parity-test",
        progress_callback=None,
        output_format="json",
    )
    # JSON 模式返回 (TranscriptionResult, raw_result)
    return result[0]


def _golden_path(golden_dir: Path, audio_name: str) -> Path:
    return golden_dir / f"{audio_name}.golden.json"


@pytest.mark.integration
class TestParityFunasrSemantic:
    @pytest.mark.asyncio
    async def test_tts_1speaker_5s(self, tts_audio: Path, golden_dir: Path):
        await self._parity_check(tts_audio, golden_dir)

    @pytest.mark.asyncio
    async def test_silence_5s(self, silence_audio: Path, golden_dir: Path):
        await self._parity_check(silence_audio, golden_dir)

    @pytest.mark.asyncio
    async def test_podcast_2speakers_60s(self, podcast_audio: Path, golden_dir: Path):
        await self._parity_check(podcast_audio, golden_dir)

    async def _parity_check(self, audio_path: Path, golden_dir: Path):
        golden_file = _golden_path(golden_dir, audio_path.stem)
        actual = await _run_funasr_once(audio_path)

        if not golden_file.exists():
            # 首次运行：把当前输出当作 baseline 保存
            golden_dir.mkdir(parents=True, exist_ok=True)
            golden_data = {
                "segments": [
                    {"start_time": s.start_time, "end_time": s.end_time, "text": s.text, "speaker": s.speaker}
                    for s in actual.segments
                ],
                "speakers": actual.speakers,
            }
            golden_file.write_text(json.dumps(golden_data, ensure_ascii=False, indent=2))
            pytest.skip(f"首次运行：已写入 golden baseline {golden_file}，复跑即生效")
        else:
            golden = json.loads(golden_file.read_text())
            assert_semantic_equal(actual, golden)
