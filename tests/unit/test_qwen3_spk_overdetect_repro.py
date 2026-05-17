"""复现 60min-2spk over-detect regression — 调研产出, red test.

背景:
    `audio_2spk_60min.mp3` (60min 真实 2 人对话) 在 production 走过
    `ffmpeg → 16kHz mono wav` 转换后, sherpa-onnx 离线 diarize 输出
    11 个 cluster, 其中两个噪声 cluster 时长 (43.2s / 61.6s) 恰好
    突破 `filter_spurious_speakers` 的 `1% × 3600s = 36s` 阈值, 未被
    过滤; cluster_merge 后仍剩 4 个 speaker (production bench 实测
    Speaker[1, 3, 4, 6]).

    同一 audio 在 PR3 PoC 时期 (直接 mp3 → librosa fallback, 不走 ffmpeg)
    sherpa-onnx 只吐 2 个主 cluster + 5 个 <3s 噪声 cluster, filter
    清干净后剩 2 spk ✓.

根因:
    `cd578a8` 提交在 `qwen3_worker_process.py` 中加了 `convert_to_wav`
    (ffmpeg → 16kHz mono wav), 改变了喂给 sherpa diarize 的 audio bytes:
    样本数一致 (57.6M samples / 3600s) 但 rms_diff ~ 0.004 (0.4%), 仅
    4.8% 的样本接近一致. 这点差异通过 pyannote / NeMo embedding 放大,
    最终在 FastClustering@threshold=0.9 边界 case 上把 cluster 数从 7
    推到 11, 并把两个噪声 cluster 时长推高到刚刚突破 spurious 阈值.

详细调研:
    `docs/开发/archive/spk-over-detect-归因调研结果.md`

本测试:
    在 *后处理层* 复现 over-detect — 用实测 sherpa 输出的 turns
    驱动 `filter_spurious_speakers`, 期望 2 speaker, 实测得到 4
    (xfail 标记, 修复后取消标记即变绿).
"""
from __future__ import annotations

import pytest

from src.core.qwen3.merge import filter_spurious_speakers


def _build_observed_turns() -> list[dict]:
    """构造实测 ffmpeg→wav 路径下 sherpa diarize 输出 turns 结构.

    时长分布来自实验:
        spikes/qwen3_silence_align 调研 + /tmp/diarize_wav_result.json
        commit 92e8442, cluster_threshold=0.9, num_threads=4, mp3 → wav.

        cluster 0: 3089.7s (86.0% main — 主讲嘉宾)
        cluster 5:  394.4s (11.0% main — 另一位嘉宾)
        cluster 3:   61.6s ( 1.7% 噪声但突破 36s 阈值)
        cluster 2:   43.2s ( 1.2% 噪声但突破 36s 阈值)
        其余 7 个 cluster: 0.3-1.7s 合计 ~4s, 占 0.11%

    为单元测试简洁起见, 这里把每个 cluster 用单条 turn 表达
    (filter_spurious 只看 sum, 行为等价).
    """
    return [
        {"start":    0.0, "end": 3089.7, "speaker": 0},
        {"start": 3089.7, "end": 3484.1, "speaker": 5},
        {"start": 3484.1, "end": 3545.7, "speaker": 3},
        {"start": 3545.7, "end": 3588.9, "speaker": 2},
        # 7 个 <2s 噪声 cluster (会被 abs 阈值 2s 滤掉, 这里仅做现实性)
        {"start": 3588.9, "end": 3590.6, "speaker": 18},
        {"start": 3590.6, "end": 3591.4, "speaker": 15},
        {"start": 3591.4, "end": 3591.8, "speaker": 6},
        {"start": 3591.8, "end": 3592.2, "speaker": 7},
        {"start": 3592.2, "end": 3592.6, "speaker": 12},
        {"start": 3592.6, "end": 3593.0, "speaker": 16},
        {"start": 3593.0, "end": 3593.3, "speaker": 17},
    ]


AUDIO_DURATION_60MIN = 3600.0
EXPECTED_SPK_COUNT = 2  # 真实是 2 人对话, PR3 PoC 输出也是 2


@pytest.mark.xfail(
    reason=(
        "OVER-DETECT REGRESSION: ffmpeg → wav 后 sherpa diarize 多吐两个 "
        "43.2s / 61.6s 噪声 cluster, 时长 > filter_spurious 的 36s "
        "(1% × 3600s) 阈值, 滤不掉. 修复方向: 调高 min_speaker_share "
        "/ 走 librosa fallback 避开 ffmpeg / cluster_merge 多人模式. "
        "见 docs/开发/archive/spk-over-detect-归因调研结果.md."
    ),
    strict=True,
)
def test_filter_spurious_60min_2spk_ffmpeg_wav_over_detect() -> None:
    """60min-2spk 走 ffmpeg→wav 路径后, filter_spurious 无法收敛到 2 spk."""
    turns = _build_observed_turns()
    filtered = filter_spurious_speakers(
        turns,
        min_speaker_total=2.0,
        min_speaker_share=0.01,
        audio_duration=AUDIO_DURATION_60MIN,
    )
    speakers = {t["speaker"] for t in filtered}
    assert len(speakers) == EXPECTED_SPK_COUNT, (
        f"expected {EXPECTED_SPK_COUNT} speakers after filter, got "
        f"{len(speakers)}: {sorted(speakers)}"
    )


def test_filter_spurious_60min_2spk_clean_path_passes() -> None:
    """对照: PR3 PoC 时期 (mp3 直读) sherpa 只吐 7 cluster + 主 2 大 5 小, 应正常过滤到 2 spk."""
    clean_turns = [
        # 主 2 个 (跟 ffmpeg 路径下 cluster 0 / 5 对应, 时长一致)
        {"start":    0.0, "end": 3128.6, "speaker": 0},
        {"start": 3128.6, "end": 3580.2, "speaker": 2},
        # 5 个 <3s 噪声 (mp3 直读路径下的真实 sherpa 输出)
        {"start": 3580.2, "end": 3580.6, "speaker": 1},
        {"start": 3580.6, "end": 3580.9, "speaker": 3},
        {"start": 3580.9, "end": 3583.7, "speaker": 8},
        {"start": 3583.7, "end": 3585.3, "speaker": 9},
        {"start": 3585.3, "end": 3585.7, "speaker": 18},
    ]
    filtered = filter_spurious_speakers(
        clean_turns,
        min_speaker_total=2.0,
        min_speaker_share=0.01,
        audio_duration=AUDIO_DURATION_60MIN,
    )
    speakers = {t["speaker"] for t in filtered}
    assert len(speakers) == EXPECTED_SPK_COUNT, (
        f"clean path 应过滤到 {EXPECTED_SPK_COUNT} spk, 实测 {len(speakers)}: "
        f"{sorted(speakers)}"
    )
