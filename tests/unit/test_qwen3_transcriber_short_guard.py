"""
PR2 — qwen3_transcriber 集成 short-segment guard 的 helper.

测试 apply_short_segment_guard_to_segments(merged_segments, qwen3_config):
- enabled=False 直接返回原 List[Segment]
- enabled=True 转 dict -> apply guard -> 转回 List[Segment]
- 转换前后 Segment 字段语义保持 (start/end/speaker/text)
"""
from __future__ import annotations

from src.core.config import Qwen3Config
from src.core.qwen3.merge import Segment


class TestApplyShortSegmentGuardToSegments:
    def test_disabled_returns_original_segments_list(self) -> None:
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config(short_segment_guard_enabled=False)
        segs = [
            Segment(start=0.0, end=0.1, speaker=0, text="ghost"),  # tiny 但应保留
            Segment(start=0.1, end=2.0, speaker=1, text="正常"),
        ]
        out, stats = apply_short_segment_guard_to_segments(segs, cfg)
        assert out == segs
        assert stats.get("enabled") is False

    def test_enabled_drops_ghost_and_returns_segment_objects(self) -> None:
        """enabled=True 时, 0s ghost 段被 drop, 返回类型仍是 List[Segment]."""
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config(
            short_segment_guard_enabled=True,
            short_segment_drop_sec=0.5,  # ghost dur=0 < 0.5
            short_segment_aba_max_mid_sec=1.5,
            short_segment_merge_same=True,
        )
        segs = [
            Segment(start=0.0, end=3.0, speaker=0, text="前段"),
            Segment(start=3.0, end=3.0, speaker=1, text="幽灵"),  # 0s ghost
            Segment(start=3.5, end=6.0, speaker=0, text="后段"),
        ]
        out, stats = apply_short_segment_guard_to_segments(segs, cfg)
        # 转换类型保持
        assert all(isinstance(s, Segment) for s in out)
        # ghost 段已被合并
        assert len(out) <= 2
        assert stats.get("enabled") is True
        assert stats.get("drop", {}).get("dropped_total", 0) >= 1

    def test_enabled_preserves_segment_field_semantics(self) -> None:
        """guard 后 Segment 的 start/end 为 float, speaker 为 int, text 为 str."""
        from src.core.qwen3_transcriber import apply_short_segment_guard_to_segments

        cfg = Qwen3Config()
        segs = [
            Segment(start=0.0, end=3.0, speaker=0, text="A"),
            Segment(start=3.0, end=6.0, speaker=1, text="B"),
        ]
        out, _stats = apply_short_segment_guard_to_segments(segs, cfg)
        for s in out:
            assert isinstance(s.start, float)
            assert isinstance(s.end, float)
            assert isinstance(s.speaker, int)
            assert isinstance(s.text, str)


class TestTranscribePipelineAppliesGuard:
    """mock 全链路: 注入含 ghost 段的 merged_segments, 验证 transcribe() 输出已被 guard 处理.

    不依赖真 ASR / sherpa 模型 — 验证 guard 在 transcribe 主流程被调.
    """

    async def test_ghost_segment_dropped_in_json_output(self, monkeypatch) -> None:
        """注入 ghost 段, transcribe(json) 返回 segments 不含 ghost."""
        import src.core.qwen3_transcriber as qwen3_tx_mod
        from src.core.qwen3_transcriber import Qwen3DiarizeTranscriber

        # 1) 桩 ASR 引擎
        class _StubChunk:
            def __init__(self, idx, start, end, text):
                self.index = idx
                self.start = start
                self.end = end
                self.text = text

        class _StubASRResult:
            def __init__(self):
                self.text = "完整音频文本"
                self.duration = 10.0
                self.elapsed = 1.0
                self.rtf = 0.1
                self.chunks = [_StubChunk(0, 0.0, 10.0, "完整音频文本")]

        class _StubEngine:
            pass

        monkeypatch.setattr(qwen3_tx_mod, "build_engine", lambda model_dir: _StubEngine())

        def _stub_run_asr(*args, **kwargs):
            return _StubASRResult()

        monkeypatch.setattr(qwen3_tx_mod, "run_asr", _stub_run_asr)

        def _stub_run_diarization(*args, **kwargs):
            return [
                {"speaker": 0, "start": 0.0, "end": 3.0},
                {"speaker": 0, "start": 3.5, "end": 10.0},
            ]

        monkeypatch.setattr(qwen3_tx_mod, "run_diarization_dispatched", _stub_run_diarization)

        # 关键: mock merge_asr_chunks_and_diarize 注入 0s ghost 段
        from src.core.qwen3.merge import Segment

        def _stub_merge(chunks, turns):
            return [
                Segment(start=0.0, end=3.0, speaker=0, text="前段"),
                Segment(start=3.0, end=3.0, speaker=1, text="幽灵"),  # 0s ghost
                Segment(start=3.5, end=10.0, speaker=0, text="后段"),
            ]

        monkeypatch.setattr(qwen3_tx_mod, "merge_asr_chunks_and_diarize", _stub_merge)
        monkeypatch.setattr(
            qwen3_tx_mod,
            "filter_spurious_speakers",
            lambda turns, **kw: turns,
        )

        async def _stub_calc_hash(path):
            return "fake-hash"

        monkeypatch.setattr(qwen3_tx_mod, "calculate_file_hash", _stub_calc_hash)

        # 2) 构造 transcriber, short_drop_sec=0.5 让 0s ghost 被 drop
        transcriber = Qwen3DiarizeTranscriber(
            asr_model_dir="/fake/model_dir",
            segmentation_model="/fake/seg.onnx",
            embedding_model="/fake/emb.onnx",
            short_segment_guard_enabled=True,
            short_segment_drop_sec=0.5,
        )

        # 3) 跑 transcribe(json)
        result, raw = await transcriber.transcribe(
            audio_path="/fake/audio.wav",
            task_id="ghost-guard-test",
            progress_callback=None,
            output_format="json",
        )

        # 4) ghost segment 不应出现在最终输出
        assert all(s.text != "幽灵" or "幽灵" in s.text and len(s.text) > 2 for s in result.segments)
        # 更严格: 不应有 0s duration 段
        assert all(s.end_time > s.start_time for s in result.segments)
        # 总段数应 <= 2 (3 段, ghost 合并掉)
        assert len(result.segments) <= 2
