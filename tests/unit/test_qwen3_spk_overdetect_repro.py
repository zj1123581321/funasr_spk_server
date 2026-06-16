"""60min-2spk over-detect regression guard — 修复后保留作历史档案.

历史背景 (修复前):
    `audio_2spk_60min.mp3` (60min 真实 2 人对话) 在 production 走过
    `ffmpeg → 16kHz mono wav` 转换后, sherpa-onnx 离线 diarize 输出
    11 个 cluster, 其中两个噪声 cluster 时长 (43.2s / 61.6s) 恰好
    突破 `filter_spurious_speakers` 的 `1% × 3600s = 36s` 阈值, 未被
    过滤; cluster_merge 后仍剩 4 个 speaker (production bench 实测
    Speaker[1, 3, 4, 6]).

    同一 audio 在 PR3 PoC 时期 (直接 mp3 → librosa fallback, 不走 ffmpeg)
    sherpa-onnx 只吐 2 个主 cluster + 5 个 <3s 噪声 cluster, filter
    清干净后剩 2 spk ✓.

根因 (调研报告 docs/开发/archive/spk-over-detect-归因调研结果.md):
    `cd578a8` 提交在 `qwen3_worker_process.py` 中加了 `convert_to_wav`
    (ffmpeg → 16kHz mono wav), 改变了喂给 sherpa diarize 的 audio bytes;
    embedding 漂移把 cluster 数从 7 推到 11, 两个噪声 cluster 时长越过阈值.

修复 (本 PR fix/qwen3-spk-overdetect):
    方向 2 (治本): worker 对 sherpa-supported 格式 (wav/flac/ogg/mp3/opus) 跳过
        ffmpeg, mp3 走 librosa fallback 直读, sherpa 拿到的 audio 跟 PR3 PoC 一致
    方向 4 (兜底): cluster_merge dominant 模式扩展到 minor — 即使将来又有 audio
        触发同形 over-detect, 这些 minor 噪声 cluster 跟 dominant cos ≥ 0.5 会被合掉

本 test 保留作 *活档案 + regression guard*:
1. clean_path: PR3 PoC 时期形态 (mp3 直读), filter_spurious 正常收敛 2 spk (修复前后都应过)
2. full_pipeline_minor_fold: 即使 sherpa 又吐出 over-detect 形态 (反事实 4-spk turns),
   cluster_merge minor-fold 兜底应恢复 2 spk (验证方向 4 兜底真的能用)
"""
from __future__ import annotations

import numpy as np

from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge
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


def test_over_detect_turns_recovered_by_cluster_merge_minor_fold() -> None:
    """over-detect 形态 turns 通过 cluster_merge dominant-minor-fold (方向 4) 兜底恢复 2 spk.

    场景: filter_spurious 漏掉 43.2s/61.6s 噪声 cluster (突破 36s 阈值), 留下 4 spk.
    cluster_merge minor-fold 应识别这两个 cluster 跟 dominant cos ≥ 0.5 → 合到 dominant.
    main cluster 5 (跟 dominant cos = 0) 保留. 最终 2 spk.

    意义 (regression guard): 即使将来又有 audio 触发同形 over-detect (新解码器 /
    新 audio profile), 兜底层能拦截. 修复前 (no minor-fold) 4 spk, 修复后 2 spk.
    """
    turns = _build_observed_turns()
    filtered = filter_spurious_speakers(
        turns,
        min_speaker_total=2.0,
        min_speaker_share=0.01,
        audio_duration=AUDIO_DURATION_60MIN,
    )
    # filter 后剩 4 spk: 0 (dominant 86%), 5 (main 11%), 3 (minor 1.7%), 2 (minor 1.2%)
    pre_merge_speakers = {t["speaker"] for t in filtered}
    assert len(pre_merge_speakers) == 4, (
        f"filter_spurious 阶段应留 4 spk (over-detect 形态), 实测 "
        f"{len(pre_merge_speakers)}: {sorted(pre_merge_speakers)}"
    )

    # 转 segments 形式 (cluster_merge 内部按 str 比较 speaker)
    segments = [
        {
            "start": float(t["start"]),
            "end": float(t["end"]),
            "speaker": str(t["speaker"]),
            "text": "",
        }
        for t in filtered
    ]

    # mock embedding: 噪声 cluster 3/2 跟 dominant cos = 0.55 (≥ dominant_minor=0.5);
    # main cluster 5 跟 dominant cos = 0 (保留 main, 不被合).
    sqrt_orth = float(np.sqrt(1.0 - 0.55 * 0.55))
    emb_by_spk = {
        "0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "5": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "3": np.array([0.55, 0.0, sqrt_orth], dtype=np.float32),
        "2": np.array([0.55, 0.0, sqrt_orth], dtype=np.float32),
    }
    start_to_sp = {round(s["start"], 6): s["speaker"] for s in segments}

    def extractor(audio_16k, start, end):
        return emb_by_spk.get(start_to_sp.get(round(float(start), 6)))

    new_segs, _log = apply_cluster_centroid_merge(
        segments,
        extractor_fn=extractor,
        audio_16k=np.zeros(16000, dtype=np.float32),
        dominant_minor_threshold=0.5,
    )

    final_speakers = {s["speaker"] for s in new_segs}
    assert len(final_speakers) == EXPECTED_SPK_COUNT, (
        f"cluster_merge 兜底后应恢复 {EXPECTED_SPK_COUNT} spk, "
        f"实测 {len(final_speakers)}: {sorted(final_speakers)}. "
        f"如果 > 2, 说明 dominant minor-fold 没生效 (修复方向 4 回归)."
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
