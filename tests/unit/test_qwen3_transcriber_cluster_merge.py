"""
PR3 — qwen3_transcriber 集成 cluster_merge 的 helper.

测试 apply_cluster_centroid_merge_to_turns(turns, audio_path, cfg_like):
- enabled=False 直接返回原 turns
- enabled=True 加载 audio + build extractor + 调 apply_cluster_centroid_merge
- 返回类型保持 List[dict]
"""
from __future__ import annotations

import numpy as np

from src.core.config import Qwen3Config


class TestApplyClusterCentroidMergeToTurns:
    def test_disabled_returns_original_turns(self) -> None:
        from src.core.qwen3_transcriber import apply_cluster_centroid_merge_to_turns

        cfg = Qwen3Config(cluster_merge_enabled=False)
        turns = [
            {"speaker": 0, "start": 0.0, "end": 5.0},
            {"speaker": 1, "start": 5.0, "end": 10.0},
        ]
        out, log = apply_cluster_centroid_merge_to_turns(turns, "/fake/audio.wav", cfg)
        assert out == turns
        assert log == []

    def test_enabled_loads_audio_and_merges_dup_clusters(self, monkeypatch) -> None:
        """enabled=True 时, 加载 audio + extractor, 把相似的 speaker cluster 合并."""
        import src.core.qwen3_transcriber as qwen3_tx_mod
        from src.core.qwen3.merge import Segment  # 不需要, 只是确保 import 不破

        # 1) Stub _load_audio_mono_16k 返回 dummy audio
        monkeypatch.setattr(
            qwen3_tx_mod,
            "_load_audio_mono_16k",
            lambda path: (np.zeros(16000 * 30), 16000),
        )

        # 2) Stub build_embedding_extractor 返回固定 extractor_fn (映射 speaker -> embedding)
        # 6 speaker, 但 4/5 与 0/1 同向 → 应合并到 4
        emb_map = {
            0: np.array([1.0, 0.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 0.0, 1.0]),
            4: np.array([1.0, 0.0, 0.0, 0.0]),  # ~ 0
            5: np.array([0.0, 1.0, 0.0, 0.0]),  # ~ 1
        }

        # turns: 各 speaker 各 20s
        turns = [
            {"speaker": sp, "start": sp * 20.0, "end": (sp + 1) * 20.0}
            for sp in range(6)
        ]

        def fake_extractor_fn(audio, start, end):
            for t in turns:
                if t["start"] <= start < t["end"]:
                    return emb_map[t["speaker"]]
            return None

        monkeypatch.setattr(
            qwen3_tx_mod,
            "build_embedding_extractor_fn",
            lambda cfg_like: fake_extractor_fn,
        )

        cfg = Qwen3Config(cluster_merge_enabled=True)
        out, log = qwen3_tx_mod.apply_cluster_centroid_merge_to_turns(
            turns, "/fake/audio.wav", cfg
        )

        # 6 -> 4 speakers
        assert len(set(t["speaker"] for t in out)) == 4
        # 至少 2 个 merge 事件
        merged_evs = [e for e in log if e.get("action") == "main_merged_high_conf"]
        assert len(merged_evs) >= 2
