"""
词级时间戳 — 缓存 key 加 word_align 维

word_align 加 segment.words 是契约变化: 开启后的结果跟未开启的不能互相命中.
把 word_align 状态 (enabled + 有效语言) 折进缓存用的 engine tag:
- qwen3 + word_align 开 → "qwen3+wa:<lang>" (lang = per-request language or config 兜底)
- qwen3 + word_align 关 → "qwen3" (老 key, 向后兼容)
- 非 qwen3 (funasr) → 原样 (FunASR 无 word_align)
- 带 wa tag 时 strict 查 (allow_cross_engine=False), 避免跨引擎回退到无词结果.
"""
from __future__ import annotations

import pytest

from src.core.database import compute_cache_engine, cache_lookup_params, cache_params
from src.models.schemas import TranscribeOptions


class TestComputeCacheEngine:
    def test_qwen3_word_align_on_with_request_language(self):
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language="eng", word_align_language="chi"
            )
            == "qwen3+wa:eng"
        )

    def test_qwen3_word_align_on_falls_back_to_config_language(self):
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language=None, word_align_language="chi"
            )
            == "qwen3+wa:chi"
        )

    def test_qwen3_word_align_off_plain_engine(self):
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=False, language="eng", word_align_language="chi"
            )
            == "qwen3"
        )

    def test_funasr_never_tagged(self):
        assert (
            compute_cache_engine(
                "funasr", word_align_enabled=True, language="eng", word_align_language="chi"
            )
            == "funasr"
        )

    # ---- 决策 2A: SRT 强制降 +wa (word_align JSON-only, SRT 行无词) ----

    def test_srt_drops_wa_tag(self):
        """SRT + word_align=true: 该行实际无词 (word_align JSON-only), tag 降回纯 qwen3,
        避免被未来 JSON +wa 请求误命中 (无词却以为对齐完成)."""
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language="eng",
                word_align_language="chi", output_format="srt",
            )
            == "qwen3"
        )

    def test_json_keeps_wa_tag(self):
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language="eng",
                word_align_language="chi", output_format="json",
            )
            == "qwen3+wa:eng"
        )

    def test_srt_wa_plus_nospk_keeps_only_nospk(self):
        """SRT + word_align=true + diarize=false → +wa 降, 仅保留 +nospk."""
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language="eng",
                word_align_language="chi", diarize=False, output_format="srt",
            )
            == "qwen3+nospk"
        )

    def test_default_output_format_is_json_backcompat(self):
        """不传 output_format 默认 json (向后兼容老调用)."""
        assert (
            compute_cache_engine(
                "qwen3", word_align_enabled=True, language="eng", word_align_language="chi"
            )
            == "qwen3+wa:eng"
        )


class TestCacheParamsReadsOptions:
    """决策 1A: cache_params 读 options.word_align (非全局 config)."""

    def test_cache_params_uses_options_word_align_on(self):
        opts = TranscribeOptions(language="eng", word_align=True)
        engine, allow_cross = cache_params("qwen3", opts)
        assert engine == "qwen3+wa:eng"
        assert allow_cross is False

    def test_cache_params_uses_options_word_align_off(self):
        opts = TranscribeOptions(language="eng", word_align=False)
        engine, allow_cross = cache_params("qwen3", opts)
        assert engine == "qwen3"
        assert allow_cross is None

    def test_cache_params_srt_drops_wa(self):
        opts = TranscribeOptions(language="eng", word_align=True)
        engine, _ = cache_params("qwen3", opts, output_format="srt")
        assert engine == "qwen3"


class TestCrossEngineExcludesFoldedRows:
    """决策 C (codex #4): 跨引擎 file_hash 回退只命中裸 engine 行, 排除 +wa/+nospk 折维行.

    反向污染: base qwen3 (word_align=false) 请求 allow_cross=None 跟随 config.cache_cross_engine,
    开启时若回退命中 qwen3+wa 行 → 没要词却返回带词. 修法: 回退查询排除带 '+' 的折维 tag.
    """

    @pytest.mark.asyncio
    async def test_base_request_does_not_cross_hit_wa_row(self, tmp_path):
        from src.core.database import DatabaseManager
        from src.core.config import config
        from src.models.schemas import TranscriptionResult, TranscriptionSegment, WordTimestamp

        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        await db.init_db()

        original = config.transcription.cache_cross_engine
        config.transcription.cache_cross_engine = True
        try:
            # 只存一行带词的 +wa 折维行
            wa_result = TranscriptionResult(
                task_id="t", file_name="a.wav", file_hash="hX", duration=1.0,
                segments=[TranscriptionSegment(
                    start_time=0, end_time=1, text="hi", speaker="Speaker1",
                    words=[WordTimestamp(text="hi", start=0.0, end=0.5)],
                )],
                speakers=["Speaker1"], processing_time=0.1,
            )
            await db.save_result(wa_result, raw_result={"engine": "qwen3"}, engine="qwen3+wa:chi")

            # base qwen3 (无 wa) 请求, 允许跨引擎. 折维行须被排除 → miss (None)
            hit = await db.get_cached_result(
                "hX", "json", engine="qwen3", allow_cross_engine=True,
                options=TranscribeOptions(word_align=False),
            )
            assert hit is None, "base 请求不应跨引擎命中 +wa 折维行 (反向污染)"
        finally:
            config.transcription.cache_cross_engine = original


class TestCacheLookupParams:
    def test_word_align_tag_forces_strict_cross_engine(self):
        engine, allow_cross = cache_lookup_params(
            "qwen3", word_align_enabled=True, language="eng", word_align_language="chi"
        )
        assert engine == "qwen3+wa:eng"
        assert allow_cross is False

    def test_plain_engine_keeps_default_cross_engine(self):
        engine, allow_cross = cache_lookup_params(
            "qwen3", word_align_enabled=False, language=None, word_align_language="chi"
        )
        assert engine == "qwen3"
        assert allow_cross is None
