"""Lightweight Qwen3 ASR text post-processing helpers.

These helpers are intentionally deterministic.  They are not a replacement for
an LLM proofreader; they target high-value, repeated technical terms that ASR
models often render phonetically in long podcast/interview audio.
"""
from __future__ import annotations

import re


TECH_PODCAST_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("B端的路线", "闭源的路线"),
    ("B端路线", "闭源路线"),
    ("后普", "开源"),
    ("候补", "开源"),
    ("曼奇", "曼祺"),
    ("肖红", "肖宏"),
    ("Malice", "Manus"),
    ("malice", "Manus"),
    ("OpenCLow", "OpenClaw"),
    ("OpenCLoud", "OpenClaw"),
    ("OpenCloud", "OpenClaw"),
    ("OpenAI引起个人Agent", "OpenClaw引起个人Agent"),
    ("Open Coding", "OpenClaw"),
    ("Open coding", "OpenClaw"),
    ("Open Claw", "OpenClaw"),
    ("OpenCL", "OpenClaw"),
    ("Cloud Code", "Claude Code"),
    ("cloud code", "Claude Code"),
    ("Cloud Work", "Claude Work"),
    ("cloud work", "Claude Work"),
    ("Cloud CoWork", "Claude Work"),
    ("astropeg", "Anthropic"),
    ("Astropeg", "Anthropic"),
    ("astro peak", "Anthropic"),
    ("Astro peak", "Anthropic"),
    ("Agentic Se\nrvice", "Agentic Service"),
    ("agentic se\nrvice", "Agentic Service"),
    ("张鱼", "章鱼"),
    ("张域", "章鱼"),
    ("且微信", "企业微信"),
    ("写微信", "企业微信"),
    ("奔驰马克", "benchmark"),
    ("朗兰斯纲领", "朗兰兹纲领"),
    ("solve 和 memory", "soul 和 memory"),
    ("solve和memory", "soul和memory"),
    ("solve 和 Memory", "soul 和 memory"),
    ("secret 都能改掉", "soul 都能改掉"),
    ("secret都能改掉", "soul都能改掉"),
    ("开了 C L I", "开了 API"),
    ("开了CLI", "开了 API"),
    ("通过 C L I", "通过 API"),
    ("通过CLI", "通过 API"),
    ("Deep Miner", "Deep Minor"),
    ("data intention", "Data Intelligence"),
    ("data intelligence", "Data Intelligence"),
)


def apply_tech_podcast_glossary(text: str) -> str:
    """Apply conservative glossary corrections for the PR4 tech podcast sample."""
    out = text
    for src, dst in TECH_PODCAST_REPLACEMENTS:
        out = out.replace(src, dst)
    out = re.sub(r"\bCloud\s+Code\b", "Claude Code", out)
    return out
