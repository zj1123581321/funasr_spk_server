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

from src.core.database import compute_cache_engine, cache_lookup_params


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
