"""nospk 输出投影 — 把 diarized TranscriptionResult 投影成无说话人区分形态

设计定案 D3+T-A: 纯函数, 引擎代码零改动. 应用点两个 (双出口):
1. db_manager.get_cached_result 出口 — 覆盖 websocket_handler 早返回
   (upload_request / chunked finalize) + task_manager 缓存读;
   E1: qwen3 nospk 请求 exact tag miss → 同引擎同 wa-tag diarized 行现场投影
   (标 projected:true, **不回写** — 缓存里永远只有真算结果, T2);
   funasr 免折维行 (D4) 本身 diarized, 出口投影一行通吃.
2. task_manager fresh 结果出口 — funasr 照算后投影给客户端, 缓存仍存 diarized 原结果.

对外契约 (D8): diarize=false ⇒ segments[].speaker=null + speakers=[] +
SRT 无 "SpeakerN:" 前缀. words / 时间 / 文本不动.
"""
from __future__ import annotations

from typing import Iterable, List

from src.models.schemas import TranscriptionResult, TranscriptionSegment


def build_result_metadata(*, engine: str, options, projected: bool = False) -> dict:
    """E2 effective options 回显块 (serve 层组装, 不入库).

    字段: engine / diarize / word_align / language / projected.
    合并优先级 (E2 定义): request > 分片 session 回填 > config > 引擎默认 —
    request 与 session 回填已在 TranscribeOptions 收拢 (session 值在
    FileUploadRequest 构造时回填), 本函数只做 options 与 config / 引擎默认的合并:

    - word_align: 仅 qwen3 引擎且 server config 开启时为 True (funasr 无此能力)
    - language: request 值 strip 规范化优先; word_align 开启时空值回退
      config.qwen3.word_align_language (与缓存折维同一规范化规则)
    - projected: 请求级属性 — 本响应是否由 diarized 结果投影而来
      (缓存投影回退命中 / funasr 照算出口投影)
    """
    from src.core.config import config

    word_align = engine == "qwen3" and config.qwen3.word_align_enabled
    language = (options.language or "").strip() or None
    if word_align and language is None:
        language = config.qwen3.word_align_language
    return {
        "engine": engine,
        "diarize": options.diarize,
        "word_align": word_align,
        "language": language,
        "projected": projected,
    }


def project_result_nospk(result: TranscriptionResult) -> TranscriptionResult:
    """把 diarized 结果投影成 nospk 形态 (纯函数, 不动入参, 幂等).

    speaker → None (null = 未区分, 与"真只有一人"可区分), speakers → [].
    其余字段 (segments 时间/文本/words, duration, metadata 等) 原样保留.
    """
    projected = result.model_copy(deep=True)
    projected.segments = [
        seg.model_copy(update={"speaker": None}) for seg in projected.segments
    ]
    projected.speakers = []
    return projected


def segments_to_srt_text(segments: Iterable[TranscriptionSegment]) -> str:
    """schema 层 segments → SRT 字符串 (引擎中立渲染点).

    speaker 非空 → "SpeakerN:文本" (与 FunASR 字节级对齐, 冒号后无空格);
    speaker=None (nospk) → 纯文本行, 无前缀.
    空文本片段跳过, 索引按非空 segment 重新编号.
    """
    def fmt(ms: int) -> str:
        seconds = ms / 1000
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        msec = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

    lines: List[str] = []
    idx = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx += 1
        start_ms = int(round(seg.start_time * 1000))
        end_ms = int(round(seg.end_time * 1000))
        lines.append(str(idx))
        lines.append(f"{fmt(start_ms)} --> {fmt(end_ms)}")
        lines.append(f"{seg.speaker}:{text}" if seg.speaker else text)
        lines.append("")
    return "\n".join(lines)
