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

    async def test_initialize_eager_warms_up_extractor(self, monkeypatch) -> None:
        """transcriber.initialize() 应同时 eager warm ASR engine + sherpa extractor.

        worker_loop 启动时调 initialize() 让两者都 ready, 第一个 task 不再含
        extractor build 的 3-5s overhead.
        """
        import src.core.qwen3_transcriber as qwen3_tx_mod
        from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

        # 桩两个 build 函数, 跟踪调用次数
        engine_build_count = {"n": 0}
        extractor_build_count = {"n": 0}

        def _stub_build_engine(model_dir):
            engine_build_count["n"] += 1
            return object()

        def _stub_build_extractor(cfg_like):
            extractor_build_count["n"] += 1
            return lambda audio, start, end: None

        monkeypatch.setattr(qwen3_tx_mod, "build_engine", _stub_build_engine)
        monkeypatch.setattr(qwen3_tx_mod, "build_embedding_extractor_fn", _stub_build_extractor)

        transcriber = Qwen3DiarizeTranscriber(
            asr_model_dir="/fake",
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
        # 状态前: 两个都 None
        assert transcriber._asr_engine is None
        assert transcriber._embedding_extractor_fn is None

        await transcriber.initialize()

        # 两个都被 eager build 一次
        assert engine_build_count["n"] == 1
        assert extractor_build_count["n"] == 1
        # 后续调用 ensure 应复用, 不再 build
        transcriber._ensure_embedding_extractor_fn()
        assert extractor_build_count["n"] == 1

    async def test_initialize_extractor_failure_does_not_break_engine(
        self, monkeypatch
    ) -> None:
        """extractor warmup 失败不影响 ASR engine warmup — fail-soft 设计.

        生产场景: sherpa 模型缺失时, ASR 应仍可用, cluster_merge 在 task 内
        再 lazy 尝试 (那里再失败有 try/except 兜底).
        """
        import src.core.qwen3_transcriber as qwen3_tx_mod
        from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

        monkeypatch.setattr(qwen3_tx_mod, "build_engine", lambda md: object())

        def _stub_build_extractor_fails(cfg_like):
            raise RuntimeError("sherpa model missing")

        monkeypatch.setattr(
            qwen3_tx_mod, "build_embedding_extractor_fn", _stub_build_extractor_fails
        )

        transcriber = Qwen3DiarizeTranscriber(
            asr_model_dir="/fake",
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
        # initialize 不应抛错 (extractor 失败被 catch)
        await transcriber.initialize()
        # ASR engine 仍 warm
        assert transcriber._asr_engine is not None
        # extractor 仍 None (warmup 失败, 留给 task 内 lazy 重试)
        assert transcriber._embedding_extractor_fn is None

    def test_ensure_embedding_extractor_fn_is_lazy_singleton(self, monkeypatch) -> None:
        """transcriber._ensure_embedding_extractor_fn 首次 build 后 cache, 后续 task 复用."""
        import src.core.qwen3_transcriber as qwen3_tx_mod
        from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

        build_call_count = {"n": 0}

        def _stub_build(cfg_like):
            build_call_count["n"] += 1
            return lambda audio, start, end: None

        monkeypatch.setattr(qwen3_tx_mod, "build_embedding_extractor_fn", _stub_build)

        transcriber = Qwen3DiarizeTranscriber(
            asr_model_dir="/fake",
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
        )
        fn1 = transcriber._ensure_embedding_extractor_fn()
        fn2 = transcriber._ensure_embedding_extractor_fn()
        fn3 = transcriber._ensure_embedding_extractor_fn()
        assert fn1 is fn2 is fn3
        assert build_call_count["n"] == 1, (
            f"应仅 build 一次, 实际 {build_call_count['n']}"
        )

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
