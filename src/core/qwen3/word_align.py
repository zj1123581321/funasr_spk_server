"""词级时间戳对齐 wrapper — MMS-300M CTC-FA (deskpai ctc_forced_aligner ONNX fork)

给 Qwen3 引擎增量挂 segment.words: 拿 (audio, ASR文本) 出每个词的绝对秒时间.
不替换现有段边界 merge, 只在干净段上挂词 (见 docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md).

设计要点:
- 自建 onnxruntime.InferenceSession + runtime-aware providers, 不用 deskpai
  AlignmentSingleton (它运行时下载 + 写死 CPU). 模型预下到本地路径.
- 按 ASR chunk 喂 (每个 chunk 的 audio 切片 + chunk.text → 词时间 + offset chunk.start),
  不喂整文件 monolith (避 trellis OOM).
- 逐 window fallback: 某 chunk 对齐失败 → 跳过该 chunk (不挂词), stats 记失败数 + 原因,
  段照常出, 不崩.
- 中文 language="chi" (preprocess_text 对 chi 逐字切, 中英混排能吃英文).

ctc_forced_aligner 低层 API (不用 AlignmentSingleton):
    emissions, stride = generate_emissions(session, audio_1d_16k, batch_size=N)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language="chi")
    segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank)
    word_ts = postprocess_results(text_starred, spans, stride, scores)  # [{text,start,end,score}] 秒
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

from loguru import logger

# MMS sample rate 固定 16kHz, 跟 ASR/diarize 同采样率.
_SR = 16000


def resolve_word_align_providers(provider: str, runtime: Any = None) -> List[str]:
    """把 config 的 word_align_provider 解析成 onnxruntime providers 列表.

    - "auto" → [runtime.recommend_word_align_provider()] (默认 detect_runtime())
    - "cpu" / "cuda" → 对应 ExecutionProvider 别名
    - 已是 EP 名 (含 'ExecutionProvider') → 原样透传

    放在 wrapper 解析 (不在 Pydantic), 区别于 sherpa provider="auto"→"cpu" 的解析.
    """
    v = (provider or "auto").strip()
    if v.lower() == "auto":
        if runtime is None:
            from src.core.runtime import detect_runtime

            runtime = detect_runtime()
        return [runtime.recommend_word_align_provider()]
    alias = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    if v.lower() in alias:
        return [alias[v.lower()]]
    return [v]


def build_alignment_session(model_path: str, providers: List[str]):
    """构造 MMS ONNX InferenceSession (fail-fast: 模型缺失直接报错, 不运行时下载).

    Args:
        model_path: 本地 MMS ONNX 路径 (download_qwen3_models.sh 预下).
        providers: onnxruntime ExecutionProvider 列表.

    Raises:
        FileNotFoundError: model_path 不存在 (附带提示如何预下).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"word_align MMS ONNX 模型不存在: {model_path}. "
            f"请用 scripts/download_qwen3_models.sh 预下, 或从 ~/ctc_forced_aligner/model.onnx 拷贝. "
            f"(不走 deskpai 运行时下载, 避免首个请求卡 1.2GB 下载)"
        )
    import onnxruntime

    logger.info(f"加载 word_align MMS ONNX: {model_path} providers={providers}")
    return onnxruntime.InferenceSession(model_path, providers=providers)


def _build_tokenizer():
    """构造 ctc_forced_aligner.Tokenizer (轻量, vocab dict, 无 IO)."""
    import ctc_forced_aligner as ctc

    return ctc.Tokenizer()


def align_window(
    audio_window_16k,
    text: str,
    *,
    session,
    tokenizer,
    language: str = "chi",
    batch_size: int = 16,
) -> List[dict]:
    """对单个音频窗口 + 文本跑 CTC-FA, 返回相对窗口起点的词时间戳.

    Args:
        audio_window_16k: 1D float32 numpy, 16kHz mono (单 chunk 的音频切片).
        text: 该窗口对应的 ASR 原文 (保标点/格式, 不用归一化 token).
        session: build_alignment_session 出的 ONNX session.
        tokenizer: ctc_forced_aligner.Tokenizer 实例.
        language: ISO 码 (chi/eng/jpn/kor...).
        batch_size: generate_emissions ONNX 推理 batch.

    Returns:
        [{text, start, end, score}, ...], start/end 是相对窗口起点的秒. 空文本 → [].
    """
    if not text or not text.strip():
        return []
    import ctc_forced_aligner as ctc

    emissions, stride = ctc.generate_emissions(
        session, audio_window_16k, batch_size=batch_size
    )
    tokens_starred, text_starred = ctc.preprocess_text(
        text, romanize=True, language=language
    )
    segments, scores, blank = ctc.get_alignments(emissions, tokens_starred, tokenizer)
    spans = ctc.get_spans(tokens_starred, segments, blank)
    return ctc.postprocess_results(text_starred, spans, stride, scores)


def align_chunks(
    audio_16k,
    chunks,
    *,
    session,
    tokenizer,
    language: str = "chi",
    batch_size: int = 16,
) -> Tuple[List[dict], dict]:
    """按 ASR chunk 逐窗口喂 CTC-FA, 拼出整段词级时间戳 (绝对秒).

    每个 chunk 用其 [start, end] 切 audio_16k, 对 chunk.text 对齐, 词时间 offset chunk.start.
    逐 window fallback: 某 chunk 抛异常 → 跳过 (不挂词), stats 记 failed_windows + reason,
    其它 chunk 照常出.

    Args:
        audio_16k: 1D float32 numpy, 16kHz mono (整文件波形).
        chunks: list of objs/dicts with start/end (秒) + text (ASR chunk).
        session/tokenizer/language/batch_size: 同 align_window.

    Returns:
        (words, stats):
          words: [{text, start, end, score}, ...] 绝对秒, 按 chunk 顺序拼接.
          stats: {total_windows, failed_windows, total_words, failures:[{index, reason}]}.
    """
    words: List[dict] = []
    failures: List[dict] = []
    total_windows = 0

    for idx, chunk in enumerate(chunks):
        c_start = float(getattr(chunk, "start", None) if not isinstance(chunk, dict) else chunk["start"])
        c_end = float(getattr(chunk, "end", None) if not isinstance(chunk, dict) else chunk["end"])
        c_text = (getattr(chunk, "text", None) if not isinstance(chunk, dict) else chunk.get("text")) or ""
        if c_end <= c_start or not c_text.strip():
            continue
        total_windows += 1
        a = int(max(0.0, c_start) * _SR)
        b = int(min(len(audio_16k) / _SR, c_end) * _SR)
        window = audio_16k[a:b]
        try:
            win_words = align_window(
                window,
                c_text,
                session=session,
                tokenizer=tokenizer,
                language=language,
                batch_size=batch_size,
            )
        except Exception as exc:  # 逐 window fallback, 不阻塞整段
            failures.append({"index": idx, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        for w in win_words:
            words.append(
                {
                    "text": w["text"],
                    "start": float(w["start"]) + c_start,
                    "end": float(w["end"]) + c_start,
                    "score": w.get("score"),
                }
            )

    stats = {
        "total_windows": total_windows,
        "failed_windows": len(failures),
        "total_words": len(words),
        "failures": failures,
    }
    return words, stats


class WordAligner:
    """MMS CTC-FA 对齐器 (per-worker 单例): lazy build session + tokenizer, 复用.

    用法跟 sherpa embedding extractor 单例一致 (qwen3_transcriber._ensure_*):
    首次 align_chunks 时 build session (~1-2s warm), 后续 task 复用.
    """

    def __init__(
        self,
        model_path: str,
        provider: str = "auto",
        language: str = "chi",
        batch_size: int = 16,
        runtime: Any = None,
    ):
        self.model_path = model_path
        self.providers = resolve_word_align_providers(provider, runtime=runtime)
        self.language = language
        self.batch_size = batch_size
        self._session = None
        self._tokenizer = None

    def _ensure(self):
        if self._session is None:
            self._session = build_alignment_session(self.model_path, self.providers)
            self._tokenizer = _build_tokenizer()
        return self._session, self._tokenizer

    def align_chunks(
        self, audio_16k, chunks, language: Optional[str] = None
    ) -> Tuple[List[dict], dict]:
        """对整文件波形 + ASR chunks 出词级时间戳 (绝对秒). language 缺省走构造时的兜底."""
        session, tokenizer = self._ensure()
        return align_chunks(
            audio_16k,
            chunks,
            session=session,
            tokenizer=tokenizer,
            language=language or self.language,
            batch_size=self.batch_size,
        )
