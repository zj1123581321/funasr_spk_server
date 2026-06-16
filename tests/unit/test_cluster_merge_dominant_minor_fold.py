"""Cluster merge dominant 模式吃 minor cluster 的兜底测试 (PR 修 over-detect 方向 4).

背景:
- 当前 `apply_cluster_centroid_merge` 流程:
    1) minor → main (cos ≥ relabel_threshold=0.55)
    2) main-to-main high-conf (cos ≥ main_threshold=0.78)
    3) dominant mode (share ≥ 0.6 时, 其它 main → dominant, cos ≥ dominant_threshold=0.6)
  → minor cluster 如果跟 main 的 cos 在 (dominant_minor, relabel) 之间 (例如 0.5 < cos < 0.55),
    步骤 1 不合, 步骤 3 又不处理 minor, 漏成独立 speaker.

调研结论 (spk-over-detect-归因调研结果.md):
- 60min-2spk audio over-detect 的两个 43.2s/61.6s 噪声 cluster 就是这种情况:
  share 1.2% / 1.7% (都 < min_main_share=0.03), 跟主 speaker cos 略低于 0.55,
  minor→main 漏, dominant_mode 也不管 minor.

方向 4 修复: dominant 模式 (share ≥ 0.6) 触发时, 用更宽松的阈值 dominant_minor_threshold
(默认 0.5) 把所有 minor cluster 跟 dominant 比较, 接近就合并到 dominant.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge


# ==================== fixture: 5 speaker 1000s 场景 ====================
#
# speakers:
#   A: 700s (70% — dominant)
#   B: 200s (20% — main)
#   C:  80s (8%  — main)
#   D:  12s (1.2% — minor, cos with A = 0.52)  ← 旧算法漏, 新算法合到 A
#   E:   8s (0.8% — minor, cos with A = 0.20)  ← 任何算法都保留独立
#
# 5 维向量, A/B/C 互相正交 (cos=0):
#   A = [1, 0, 0, 0, 0]
#   B = [0, 1, 0, 0, 0]
#   C = [0, 0, 1, 0, 0]
#   D = [0.52, 0, 0, sqrt(1-0.52^2), 0]  → cos(A,D) = 0.52
#   E = [0.20, 0, 0, 0, sqrt(1-0.20^2)]  → cos(A,E) = 0.20


_EMB = {
    "A": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "B": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "C": np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "D": np.array(
        [0.52, 0.0, 0.0, float(np.sqrt(1 - 0.52 * 0.52)), 0.0], dtype=np.float32
    ),
    "E": np.array(
        [0.20, 0.0, 0.0, 0.0, float(np.sqrt(1 - 0.20 * 0.20))], dtype=np.float32
    ),
}

_SPEAKER_DURATIONS = [("A", 700.0), ("B", 200.0), ("C", 80.0), ("D", 12.0), ("E", 8.0)]


def _build_segments():
    """构造 segments + start→speaker map, 每 speaker 拆 10 段"""
    segs = []
    start_to_sp = {}
    t = 0.0
    for sp, dur in _SPEAKER_DURATIONS:
        per = dur / 10
        for _ in range(10):
            start = round(t, 6)
            segs.append({"start": start, "end": round(t + per, 6), "speaker": sp, "text": "x"})
            start_to_sp[start] = sp
            t += per
    return segs, start_to_sp


def _make_extractor(start_to_sp):
    """fake extractor: 根据 segment start 查 speaker, 返回固定单位向量"""
    def extractor(audio_16k, start, end):
        sp = start_to_sp.get(round(float(start), 6))
        if sp is None:
            return None
        return _EMB[sp]
    return extractor


# ==================== red test: 新参数 + 期望行为 ====================


class TestDominantModeFoldsCloseMinor:
    """dominant ≥ 0.6 时, minor 跟 dominant cos ≥ dominant_minor_threshold 应合并"""

    def test_minor_close_to_dominant_is_folded_minor_far_kept_isolated(self):
        segments, start_to_sp = _build_segments()
        extractor = _make_extractor(start_to_sp)
        audio_16k = np.zeros(16000, dtype=np.float32)  # 占位 (fake extractor 不用真 audio)

        new_segs, log = apply_cluster_centroid_merge(
            segments,
            extractor_fn=extractor,
            audio_16k=audio_16k,
            min_main_share=0.03,
            relabel_threshold=0.55,
            main_threshold=0.78,
            dominant_share=0.6,
            dominant_threshold=0.6,
            dominant_minor_threshold=0.5,  # 新参数, 当前函数签名没有 → TypeError red
        )

        final_speakers = {s["speaker"] for s in new_segs}

        # A 必须保留 (dominant)
        assert "A" in final_speakers, f"A (dominant) 应保留, 实际 {final_speakers}"
        # B/C 是 main, 跟 A cos=0, dominant_threshold=0.6 不触发, 应保留
        assert "B" in final_speakers, f"B (main, cos with A = 0) 应保留, 实际 {final_speakers}"
        assert "C" in final_speakers, f"C (main, cos with A = 0) 应保留, 实际 {final_speakers}"

        # D: minor, cos with A = 0.52, > dominant_minor_threshold=0.5, 应合到 A
        assert "D" not in final_speakers, (
            f"D (minor, cos with A = 0.52 ≥ dominant_minor_threshold=0.5) 应被合到 A, "
            f"实际仍独立: {final_speakers}. "
            f"如果存在, 说明 dominant 模式没扩展到 minor (修复方向 4 未实现)."
        )
        # E: minor, cos with A = 0.20, < dominant_minor_threshold=0.5, 应保留
        assert "E" in final_speakers, (
            f"E (minor, cos with A = 0.20 < 0.5) 不应被合, 实际被合: {final_speakers}"
        )

    def test_minor_folded_into_dominant_logged(self):
        """log 应该有 action=minor_folded_into_dominant 条目, 标记 D→A"""
        segments, start_to_sp = _build_segments()
        extractor = _make_extractor(start_to_sp)
        audio_16k = np.zeros(16000, dtype=np.float32)

        _new_segs, log = apply_cluster_centroid_merge(
            segments,
            extractor_fn=extractor,
            audio_16k=audio_16k,
            dominant_minor_threshold=0.5,
        )

        fold_events = [e for e in log if e.get("action") == "minor_folded_into_dominant"]
        assert any(e.get("from") == "D" and e.get("to") == "A" for e in fold_events), (
            f"应有 D→A 的 minor_folded_into_dominant 日志, 实际 log: {log}"
        )


class TestDominantBelowThresholdDoesNotFoldMinor:
    """dominant share < 0.6: 不触发 dominant 模式, 也不应吃 minor"""

    def test_dominant_share_below_threshold_keeps_all_minor(self):
        # 改 share: A 只占 40% (低于 dominant_share=0.6)
        segments = []
        start_to_sp = {}
        t = 0.0
        for sp, dur in [("A", 400.0), ("B", 400.0), ("C", 180.0), ("D", 12.0), ("E", 8.0)]:
            per = dur / 10
            for _ in range(10):
                start = round(t, 6)
                segments.append({"start": start, "end": round(t + per, 6), "speaker": sp, "text": "x"})
                start_to_sp[start] = sp
                t += per

        extractor = _make_extractor(start_to_sp)
        audio_16k = np.zeros(16000, dtype=np.float32)

        new_segs, _log = apply_cluster_centroid_merge(
            segments,
            extractor_fn=extractor,
            audio_16k=audio_16k,
            min_main_share=0.03,
            relabel_threshold=0.55,
            main_threshold=0.78,
            dominant_share=0.6,
            dominant_threshold=0.6,
            dominant_minor_threshold=0.5,
        )

        final_speakers = {s["speaker"] for s in new_segs}
        # D 也应该保留 (因为 dominant 模式没触发, 不该吃 minor)
        # 注意: 仍可能被 minor->main 救; 但 cos=0.52 < relabel_threshold=0.55, 不会救, 应留
        assert "D" in final_speakers, (
            f"dominant<0.6 不触发, D 不应被吃, 实际 {final_speakers}"
        )
