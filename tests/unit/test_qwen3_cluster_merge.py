"""
src/core/qwen3/cluster_merge.py 单元测试 (PR3 cluster centroid merge).

严格 TDD: 每个测试先写, 看红, 最小实现, 看绿. 不复用 PoC 单测.

Sherpa embedding extractor 用 mock fixture, 不依赖真模型.
"""
from __future__ import annotations

import math

import numpy as np


class TestCosine:
    """cosine: numpy 向量余弦相似度."""

    def test_cosine_identical_vectors_is_1(self) -> None:
        """同方向同长度向量, cos=1."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert math.isclose(cosine(a, b), 1.0, abs_tol=1e-9)

    def test_cosine_opposite_vectors_is_minus_1(self) -> None:
        """反方向向量, cos=-1."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert math.isclose(cosine(a, b), -1.0, abs_tol=1e-9)

    def test_cosine_orthogonal_vectors_is_0(self) -> None:
        """正交向量, cos=0."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert math.isclose(cosine(a, b), 0.0, abs_tol=1e-9)

    def test_cosine_null_or_zero_input_returns_minus_1(self) -> None:
        """None / 零向量 视为 "完全不相似" 返回 -1.0 (PoC 语义)."""
        from src.core.qwen3.cluster_merge import cosine

        a = np.array([1.0, 0.0])
        zero = np.array([0.0, 0.0])
        assert cosine(None, a) == -1.0
        assert cosine(a, None) == -1.0
        assert cosine(zero, a) == -1.0
        assert cosine(a, zero) == -1.0


def _seg(start: float, end: float) -> dict:
    """测试辅助: build_centroids 只关心 start/end."""
    return {"start": start, "end": end}


class TestBuildCentroids:
    """build_centroids: 用 mock extractor_fn 算各 cluster centroid."""

    def test_single_cluster_returns_normalized_centroid(self) -> None:
        """1 speaker 多段, centroid = normalize(mean(embeddings))."""
        from src.core.qwen3.cluster_merge import build_centroids

        emb_map = {
            (0.0, 2.0): np.array([3.0, 0.0, 0.0]),
            (2.0, 4.0): np.array([1.0, 0.0, 0.0]),
        }

        def extractor_fn(audio, start, end):
            return emb_map.get((start, end))

        audio_16k = np.zeros(64000)  # dummy
        segs_by_spk = {"0": [_seg(0.0, 2.0), _seg(2.0, 4.0)]}
        centroids = build_centroids(extractor_fn, audio_16k, segs_by_spk)

        assert set(centroids.keys()) == {"0"}
        c = centroids["0"]
        # mean = [2.0, 0.0, 0.0], normalize -> [1.0, 0.0, 0.0]
        assert np.allclose(c, np.array([1.0, 0.0, 0.0]))

    def test_multiple_clusters_return_separate_centroids(self) -> None:
        """多 speaker, 每个独立 centroid."""
        from src.core.qwen3.cluster_merge import build_centroids

        emb_map = {
            (0.0, 2.0): np.array([1.0, 0.0]),
            (2.0, 4.0): np.array([0.0, 1.0]),
        }

        def extractor_fn(audio, start, end):
            return emb_map.get((start, end))

        segs_by_spk = {
            "0": [_seg(0.0, 2.0)],
            "1": [_seg(2.0, 4.0)],
        }
        centroids = build_centroids(extractor_fn, np.zeros(64000), segs_by_spk)
        assert set(centroids.keys()) == {"0", "1"}
        assert np.allclose(centroids["0"], np.array([1.0, 0.0]))
        assert np.allclose(centroids["1"], np.array([0.0, 1.0]))

    def test_speaker_with_all_none_embeddings_is_skipped(self) -> None:
        """所有段都拿不到 embedding 的 speaker, 不出现在 centroids 中."""
        from src.core.qwen3.cluster_merge import build_centroids

        def extractor_fn(audio, start, end):
            # speaker "1" 的段总是返回 None (e.g. 段太短)
            if start >= 5.0:
                return None
            return np.array([1.0, 0.0])

        segs_by_spk = {
            "0": [_seg(0.0, 2.0)],
            "1": [_seg(5.0, 5.5), _seg(6.0, 6.3)],  # 都返回 None
        }
        centroids = build_centroids(extractor_fn, np.zeros(64000), segs_by_spk)
        assert set(centroids.keys()) == {"0"}


class TestMergeMainHighConf:
    """merge_main_high_conf: 两个 main cluster cos ≥ threshold 合并, dominant 吃 recessive."""

    def test_two_mains_above_threshold_merged(self) -> None:
        """cos≥0.78, share 大的吃 share 小的."""
        from src.core.qwen3.cluster_merge import merge_main_high_conf

        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.99, 0.01]) / np.linalg.norm([0.99, 0.01]),  # cos~1
        }
        shares = {"0": 0.6, "1": 0.4}
        main_set = ["0", "1"]
        mapping_updates, remaining_mains, log = merge_main_high_conf(
            centroids, shares, main_set, merge_threshold=0.78
        )
        # share 大的 "0" 吃 "1"
        assert mapping_updates == {"1": "0"}
        assert remaining_mains == ["0"]
        assert len(log) == 1
        assert log[0]["action"] == "main_merged_high_conf"
        assert log[0]["from"] == "1" and log[0]["to"] == "0"

    def test_two_mains_below_threshold_not_merged(self) -> None:
        """cos<0.78, 不合并."""
        from src.core.qwen3.cluster_merge import merge_main_high_conf

        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.0, 1.0]),  # cos=0
        }
        shares = {"0": 0.6, "1": 0.4}
        main_set = ["0", "1"]
        mapping_updates, remaining_mains, log = merge_main_high_conf(
            centroids, shares, main_set, merge_threshold=0.78
        )
        assert mapping_updates == {}
        assert set(remaining_mains) == {"0", "1"}
        assert log == []

    def test_multi_round_until_stable(self) -> None:
        """3 个 main 都相似 (cos~1), 多轮合并到 1 个 (share 最大者)."""
        from src.core.qwen3.cluster_merge import merge_main_high_conf

        # 3 个相似 cluster
        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.99, 0.01]) / np.linalg.norm([0.99, 0.01]),
            "2": np.array([0.98, 0.02]) / np.linalg.norm([0.98, 0.02]),
        }
        shares = {"0": 0.5, "1": 0.3, "2": 0.2}
        main_set = ["0", "1", "2"]
        mapping_updates, remaining_mains, log = merge_main_high_conf(
            centroids, shares, main_set, merge_threshold=0.78
        )
        # 都合并到 "0"
        assert remaining_mains == ["0"]
        assert mapping_updates.get("1") == "0"
        assert mapping_updates.get("2") == "0"
        assert len(log) == 2


class TestMergeMinorToMain:
    """merge_minor_to_main: minor cluster -> 最近 main (cos ≥ relabel_threshold)."""

    def test_minor_above_threshold_merged_to_nearest_main(self) -> None:
        """minor 与 main_A cos=0.9, main_B cos=0.3, 合到 main_A."""
        from src.core.qwen3.cluster_merge import merge_minor_to_main

        centroids = {
            "0": np.array([1.0, 0.0]),  # main_A
            "1": np.array([0.0, 1.0]),  # main_B
            "2": np.array([0.9, 0.4359]) / np.linalg.norm([0.9, 0.4359]),  # minor, ~main_A
        }
        shares = {"0": 0.5, "1": 0.4, "2": 0.1}
        main_set = ["0", "1"]
        minor_set = ["2"]
        mapping_updates, log = merge_minor_to_main(
            centroids, shares, main_set, minor_set, relabel_threshold=0.55
        )
        assert mapping_updates == {"2": "0"}
        assert len(log) == 1
        assert log[0]["action"] == "minor_to_main"

    def test_minor_below_threshold_kept_isolated(self) -> None:
        """minor 与所有 main 相似度都低 (cos < 0.55), 保留 (e.g. 音乐 cluster)."""
        from src.core.qwen3.cluster_merge import merge_minor_to_main

        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.0, 1.0]),
            # 与 main_A cos=-0.7071, 与 main_B cos=-0.7071, 都 < 0.55
            "2": np.array([-0.7071, -0.7071]),
        }
        shares = {"0": 0.5, "1": 0.4, "2": 0.1}
        main_set = ["0", "1"]
        minor_set = ["2"]
        mapping_updates, log = merge_minor_to_main(
            centroids, shares, main_set, minor_set, relabel_threshold=0.55
        )
        assert mapping_updates == {}
        assert len(log) == 1
        assert log[0]["action"] == "minor_kept_isolated"

    def test_multiple_minors_each_assigned(self) -> None:
        """多个 minor, 各自分配到最近 main."""
        from src.core.qwen3.cluster_merge import merge_minor_to_main

        centroids = {
            "0": np.array([1.0, 0.0]),  # main_A
            "1": np.array([0.0, 1.0]),  # main_B
            "2": np.array([0.95, 0.31]) / np.linalg.norm([0.95, 0.31]),  # minor ~A
            "3": np.array([0.31, 0.95]) / np.linalg.norm([0.31, 0.95]),  # minor ~B
        }
        shares = {"0": 0.4, "1": 0.4, "2": 0.1, "3": 0.1}
        main_set = ["0", "1"]
        minor_set = ["2", "3"]
        mapping_updates, log = merge_minor_to_main(
            centroids, shares, main_set, minor_set, relabel_threshold=0.55
        )
        assert mapping_updates == {"2": "0", "3": "1"}
        assert len(log) == 2


class TestMergeDominantMode:
    """merge_dominant_mode: 合并后 dominant cluster share>=0.6 时, 其他 main 用更低阈值合到 dom."""

    def test_dominant_share_above_threshold_merges_with_low_sim(self) -> None:
        """dom share=0.7, 其他 main 与 dom cos=0.65 (>0.6) 应合并."""
        from src.core.qwen3.cluster_merge import merge_dominant_mode

        # dom = "0", other = "1", cos=0.65
        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.65, 0.76]) / np.linalg.norm([0.65, 0.76]),  # cos~0.65 with [1,0]
        }
        current_shares = {"0": 0.7, "1": 0.3}
        remaining_mains = ["0", "1"]
        mapping_updates, log = merge_dominant_mode(
            centroids, current_shares, remaining_mains,
            dominant_share=0.6, dominant_merge_threshold=0.6,
        )
        assert mapping_updates == {"1": "0"}
        assert len(log) == 1
        assert log[0]["action"] == "main_merged_dominant_mode"

    def test_dominant_share_below_threshold_not_triggered(self) -> None:
        """dom share=0.55 < 0.6, dominant 模式不触发, 即使高相似也不合."""
        from src.core.qwen3.cluster_merge import merge_dominant_mode

        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.99, 0.01]) / np.linalg.norm([0.99, 0.01]),  # cos~1
        }
        current_shares = {"0": 0.55, "1": 0.45}  # 没有 dominant
        remaining_mains = ["0", "1"]
        mapping_updates, log = merge_dominant_mode(
            centroids, current_shares, remaining_mains,
            dominant_share=0.6, dominant_merge_threshold=0.6,
        )
        assert mapping_updates == {}
        assert log == []

    def test_dominant_triggered_but_sim_below_threshold_not_merged(self) -> None:
        """dom share=0.8, 但 other main 与 dom cos=0.3 (<0.6 阈值), 不合."""
        from src.core.qwen3.cluster_merge import merge_dominant_mode

        centroids = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.3, 0.954]) / np.linalg.norm([0.3, 0.954]),  # cos~0.3
        }
        current_shares = {"0": 0.8, "1": 0.2}
        remaining_mains = ["0", "1"]
        mapping_updates, log = merge_dominant_mode(
            centroids, current_shares, remaining_mains,
            dominant_share=0.6, dominant_merge_threshold=0.6,
        )
        # dominant 触发了, 但 sim 0.3 < 0.6 阈值, 不合
        assert mapping_updates == {}
        # log 可能为空或记录"考虑但 reject", 这里至少没有 merge entry
        assert all(entry.get("action") != "main_merged_dominant_mode" for entry in log)


def _spk_seg(start: float, end: float, speaker: str, text: str = "x") -> dict:
    return {"start": start, "end": end, "speaker": speaker, "text": text}


class TestApplyClusterCentroidMerge:
    """apply_cluster_centroid_merge: 入口函数, 串联 build_centroids + 3 个 merge step."""

    def test_single_speaker_no_op(self) -> None:
        """1 speaker, 没有 minor, 1 main, 无任何合并."""
        from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge

        # 4 段全部 "0"
        segments = [
            _spk_seg(0.0, 10.0, "0"),
            _spk_seg(10.0, 20.0, "0"),
            _spk_seg(20.0, 30.0, "0"),
        ]

        def extractor_fn(audio, start, end):
            return np.array([1.0, 0.0])

        out, log = apply_cluster_centroid_merge(
            segments, extractor_fn, np.zeros(64000)
        )
        # speaker 标签全部 "0"
        assert all(s["speaker"] == "0" for s in out)
        # 没有合并事件
        assert not any(e.get("action", "").startswith("main_merged") for e in log)

    def test_two_speakers_below_threshold_not_merged(self) -> None:
        """2 main, cos 低, 不合并."""
        from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge

        segments = [
            _spk_seg(0.0, 10.0, "0"),
            _spk_seg(10.0, 20.0, "1"),
            _spk_seg(20.0, 30.0, "0"),
            _spk_seg(30.0, 40.0, "1"),
        ]
        emb_map = {
            "0": np.array([1.0, 0.0]),
            "1": np.array([0.0, 1.0]),  # 与 "0" 正交
        }

        def extractor_fn(audio, start, end):
            # 通过 start 时间窗推断 speaker
            for seg in segments:
                if seg["start"] <= start < seg["end"]:
                    return emb_map[seg["speaker"]]
            return None

        out, log = apply_cluster_centroid_merge(
            segments, extractor_fn, np.zeros(64000)
        )
        # 2 speaker 保留
        assert set(s["speaker"] for s in out) == {"0", "1"}
        assert not any(e.get("action", "").startswith("main_merged") for e in log)

    def test_music_cluster_kept_isolated_when_dissimilar(self) -> None:
        """3 main + 1 minor "music" (低 share + cos < relabel_threshold), minor 保留独立."""
        from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge

        # 3 个主 speaker 各 30s, 1 个音乐 cluster 1s 但 cos 极低
        segments = [
            _spk_seg(0.0, 30.0, "0"),
            _spk_seg(30.0, 60.0, "1"),
            _spk_seg(60.0, 90.0, "2"),
            _spk_seg(90.0, 91.0, "3"),  # share=1/91 ~1.1% < 3% min_main_share -> minor
        ]
        emb_map = {
            "0": np.array([1.0, 0.0, 0.0]),
            "1": np.array([0.0, 1.0, 0.0]),
            "2": np.array([0.0, 0.0, 1.0]),
            "3": np.array([-1.0, -1.0, -1.0]) / np.linalg.norm([-1.0, -1.0, -1.0]),  # 反方向, cos<0
        }

        def extractor_fn(audio, start, end):
            for seg in segments:
                if seg["start"] <= start < seg["end"]:
                    return emb_map[seg["speaker"]]
            return None

        out, log = apply_cluster_centroid_merge(
            segments, extractor_fn, np.zeros(16000 * 100),
            min_main_share=0.03, relabel_threshold=0.55, main_threshold=0.78,
        )
        # "3" 保持独立 (kept_isolated)
        assert any(e.get("action") == "minor_kept_isolated" and e.get("from") == "3" for e in log)
        # 最终 speakers 包含 "3"
        assert "3" in set(s["speaker"] for s in out)

    def test_six_speakers_with_dup_centroids_merged(self) -> None:
        """6 个 cluster, 其中 2 对相似, 合并到 4 个."""
        from src.core.qwen3.cluster_merge import apply_cluster_centroid_merge

        # 6 个 speaker, 每个 20s
        segments = [_spk_seg(i * 20.0, (i + 1) * 20.0, str(i)) for i in range(6)]
        # 4 个独立方向, 2 个 dup (speaker 4 = speaker 0, speaker 5 = speaker 1)
        emb_map = {
            "0": np.array([1.0, 0.0, 0.0, 0.0]),
            "1": np.array([0.0, 1.0, 0.0, 0.0]),
            "2": np.array([0.0, 0.0, 1.0, 0.0]),
            "3": np.array([0.0, 0.0, 0.0, 1.0]),
            "4": np.array([1.0, 0.0, 0.0, 0.0]),  # 同 "0"
            "5": np.array([0.0, 1.0, 0.0, 0.0]),  # 同 "1"
        }

        def extractor_fn(audio, start, end):
            for seg in segments:
                if seg["start"] <= start < seg["end"]:
                    return emb_map[seg["speaker"]]
            return None

        out, log = apply_cluster_centroid_merge(
            segments, extractor_fn, np.zeros(16000 * 200),
            main_threshold=0.78,
        )
        # 6 -> 4 speakers
        assert len(set(s["speaker"] for s in out)) == 4
        # 至少有 2 个 main_merged_high_conf 事件
        merged_evs = [e for e in log if e.get("action") == "main_merged_high_conf"]
        assert len(merged_evs) >= 2
