"""Qwen3-Diarize 转录器

调用形状与 FunASRTranscriber 对齐:
    async def transcribe(audio_path, task_id, progress_callback, output_format)
        JSON 模式 -> (TranscriptionResult, raw_result_dict)
        SRT  模式 -> {format, content, file_name, file_hash, duration, processing_time, raw_result}

内部组装:
    1. ASR  (src.core.qwen3.asr.run_asr) — 拿全文 text + duration
    2. Diarize (src.core.qwen3.diarize.run_diarization) — 拿 turns
    3. filter_spurious + merge — 把 ASR 文本切到各 speaker turn

ASR 引擎(libllama.dylib + Metal context) 实例级 lazy 单例, 进程内复用.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Optional, Tuple, Union

from loguru import logger

from src.core.qwen3.asr import build_engine, run_asr
from src.core.qwen3.diarize import run_diarization
from src.core.qwen3.merge import (
    Segment,
    filter_spurious_speakers,
    merge_asr_chunks_and_diarize,
    merge_asr_and_diarize,
    segments_to_srt,
)
from src.core.qwen3.postprocess import apply_short_segment_guard
from src.models.schemas import TranscriptionResult, TranscriptionSegment
from src.utils.file_utils import calculate_file_hash


def apply_short_segment_guard_to_segments(
    merged_segments: list[Segment],
    qwen3_config,
) -> tuple[list[Segment], dict]:
    """把 List[Segment] 经 apply_short_segment_guard 后转回 List[Segment].

    enabled=False 时直接返回原列表 (零拷贝).

    Args:
        merged_segments: 来自 merge_asr_chunks_and_diarize 的 List[Segment].
        qwen3_config: Qwen3Config 实例 (读 short_segment_* 字段).

    Returns:
        (new_segments, stats).
    """
    if not qwen3_config.short_segment_guard_enabled:
        return merged_segments, {"enabled": False}
    # 转 dict (speaker int -> str, postprocess 用字符串比较)
    seg_dicts = [
        {
            "start": float(s.start),
            "end": float(s.end),
            "speaker": str(s.speaker),
            "text": s.text,
        }
        for s in merged_segments
    ]
    out_dicts, stats = apply_short_segment_guard(
        seg_dicts,
        enabled=True,
        short_drop_sec=qwen3_config.short_segment_drop_sec,
        aba_max_mid_sec=qwen3_config.short_segment_aba_max_mid_sec,
        merge_same=qwen3_config.short_segment_merge_same,
    )
    # 转回 Segment (speaker str -> int)
    out_segments = [
        Segment(
            start=float(d["start"]),
            end=float(d["end"]),
            speaker=int(d["speaker"]),
            text=d["text"],
        )
        for d in out_dicts
    ]
    return out_segments, stats


class Qwen3DiarizeTranscriber:
    """Qwen3-ASR + sherpa Speaker Diarization 离线转录器

    参数注入式: 模型路径 / preset 由调用方传入(典型从 config 读), 类本身不绑 config.
    """

    def __init__(
        self,
        asr_model_dir: str,
        segmentation_model: str,
        embedding_model: str,
        num_speakers: Optional[int] = None,
        cluster_threshold: float = 0.9,
        num_threads: int = 8,
        provider: str = "cpu",
        language: str = "Chinese",
        temperature: float = 0.4,
        # filter_spurious 参数
        spurious_min_total: float = 2.0,
        spurious_min_share: float = 0.01,
        # PR2 short-segment guard 参数 (与 Qwen3Config 字段对齐, 鸭子类型供 helper 读)
        short_segment_guard_enabled: bool = True,
        short_segment_drop_sec: float = 1.5,
        short_segment_aba_max_mid_sec: float = 1.5,
        short_segment_merge_same: bool = True,
    ):
        self.asr_model_dir = asr_model_dir
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.num_threads = num_threads
        self.provider = provider
        self.language = language
        self.temperature = temperature
        self.spurious_min_total = spurious_min_total
        self.spurious_min_share = spurious_min_share
        self.short_segment_guard_enabled = short_segment_guard_enabled
        self.short_segment_drop_sec = short_segment_drop_sec
        self.short_segment_aba_max_mid_sec = short_segment_aba_max_mid_sec
        self.short_segment_merge_same = short_segment_merge_same

        # 引擎单例 — 第一次 transcribe 时构造, 后续复用
        self._asr_engine = None

    # ==================== 引擎管理 ====================

    def _ensure_engine(self):
        """惰性构造 ASR 引擎(libllama + Metal context 加载耗时, 进程内复用)"""
        if self._asr_engine is None:
            logger.info(f"首次构造 Qwen3-ASR 引擎(asr_model_dir={self.asr_model_dir})")
            self._asr_engine = build_engine(self.asr_model_dir)
        return self._asr_engine

    async def initialize(self):
        """提前触发引擎加载(可选, 不调也会在 transcribe 时 lazy 加载)"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_engine)

    # ==================== 进度回调 ====================

    @staticmethod
    async def _report_progress(callback: Optional[Callable], pct: int, task_id: str):
        if callback is None:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(pct)
            else:
                callback(pct)
            logger.debug(f"[{task_id}] 进度: {pct}%")
        except Exception as e:
            logger.warning(f"[{task_id}] progress_callback 异常 (忽略): {e}")

    # ==================== 主入口 ====================

    async def transcribe(
        self,
        audio_path: str,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        output_format: str = "json",
    ) -> Union[Tuple[TranscriptionResult, dict], dict]:
        """跑一遍 ASR + Diarize + Merge, 返回 (TranscriptionResult, raw_result) 或 SRT dict"""
        start_time = time.time()
        loop = asyncio.get_event_loop()

        await self._report_progress(progress_callback, 5, task_id)

        # 引擎在线程池 lazy 加载, 不阻塞 event loop
        engine = await loop.run_in_executor(None, self._ensure_engine)

        await self._report_progress(progress_callback, 10, task_id)

        # ASR + Diarize 并行(PoC 实测真并行节省 ~50% wall time)
        asr_future = loop.run_in_executor(
            None,
            lambda: run_asr(
                audio_path,
                engine,
                language=self.language,
                temperature=self.temperature,
            ),
        )
        diarize_future = loop.run_in_executor(
            None,
            lambda: run_diarization(
                audio_path,
                segmentation_model=self.segmentation_model,
                embedding_model=self.embedding_model,
                num_speakers=self.num_speakers,
                cluster_threshold=self.cluster_threshold,
                num_threads=self.num_threads,
                provider=self.provider,
            ),
        )

        # progress 在等待时定期报(简化: gather 完成时直接跳到 80)
        asr_result, turns = await asyncio.gather(asr_future, diarize_future)
        await self._report_progress(progress_callback, 80, task_id)

        # 过滤假说话人(短时长 spurious cluster)
        filtered_turns = filter_spurious_speakers(
            turns,
            min_speaker_total=self.spurious_min_total,
            min_speaker_share=self.spurious_min_share,
            audio_duration=asr_result.duration,
        )

        # 文本切分到 turns。优先使用 Qwen3-ASR 内部 chunk 时间窗，避免整段线性切字漂移。
        if asr_result.chunks:
            merged_segments = merge_asr_chunks_and_diarize(asr_result.chunks, filtered_turns)
        else:
            merged_segments = merge_asr_and_diarize(asr_result.text, filtered_turns)

        # PR2: short-segment guard 后处理 (drop tiny / ABA 平滑 / 合并同 spk)
        # 由 config 控制开关与阈值, 默认全开. SRT/JSON 都走 guard 后的 segments.
        merged_segments, guard_stats = apply_short_segment_guard_to_segments(
            merged_segments, self
        )
        if guard_stats.get("enabled"):
            logger.info(
                f"[{task_id}] short-guard: drop={guard_stats.get('drop', {}).get('dropped_total', 0)} "
                f"aba={guard_stats.get('aba', {}).get('changed', 0)} "
                f"merge_same={guard_stats.get('merge', {}).get('merged', 0)} "
                f"final_segments={len(merged_segments)}"
            )

        await self._report_progress(progress_callback, 90, task_id)

        file_hash = await calculate_file_hash(audio_path)
        processing_time = time.time() - start_time

        raw_result = {
            "asr_text": asr_result.text,
            "asr_rtf": asr_result.rtf,
            "asr_duration": asr_result.duration,
            "asr_elapsed": asr_result.elapsed,
            "asr_chunks": [
                {
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }
                for c in asr_result.chunks
            ],
            "turns": turns,
            "filtered_turns": filtered_turns,
            "engine": "qwen3",
        }

        if output_format == "srt":
            srt_content = segments_to_srt(merged_segments)
            await self._report_progress(progress_callback, 100, task_id)
            logger.info(
                f"[{task_id}] Qwen3 转录完成 (SRT): "
                f"duration={asr_result.duration:.2f}s, "
                f"耗时={processing_time:.2f}s, "
                f"RTF={processing_time / asr_result.duration:.3f}"
            )
            return {
                "format": "srt",
                "content": srt_content,
                "file_name": os.path.basename(audio_path),
                "file_hash": file_hash,
                "duration": asr_result.duration,
                "processing_time": processing_time,
                "raw_result": raw_result,
            }

        # JSON: 转 TranscriptionSegment, speaker int -> "Speaker{i+1}"
        segments = [
            TranscriptionSegment(
                start_time=round(s.start, 2),
                end_time=round(s.end, 2),
                text=s.text,
                speaker=f"Speaker{s.speaker + 1}",
            )
            for s in merged_segments
        ]
        speakers = sorted(set(seg.speaker for seg in segments))

        transcription_result = TranscriptionResult(
            task_id=task_id,
            file_name=os.path.basename(audio_path),
            file_hash=file_hash,
            duration=asr_result.duration,
            segments=segments,
            speakers=speakers,
            processing_time=processing_time,
        )

        await self._report_progress(progress_callback, 100, task_id)
        logger.info(
            f"[{task_id}] Qwen3 转录完成 (JSON): "
            f"segments={len(segments)}, speakers={len(speakers)}, "
            f"duration={asr_result.duration:.2f}s, 耗时={processing_time:.2f}s"
        )
        return (transcription_result, raw_result)


# ==================== 全局单例(C3 配置接通后从 config 读路径) ====================

_qwen3_singleton: Optional[Qwen3DiarizeTranscriber] = None


def get_qwen3_transcriber() -> Qwen3DiarizeTranscriber:
    """获取 Qwen3 转录器单例 (从 config.qwen3 读模型路径 + preset)"""
    global _qwen3_singleton
    if _qwen3_singleton is not None:
        return _qwen3_singleton

    from src.core.config import config

    q = config.qwen3
    _qwen3_singleton = Qwen3DiarizeTranscriber(
        asr_model_dir=q.asr_model_dir,
        segmentation_model=q.segmentation_model,
        embedding_model=q.embedding_model,
        num_speakers=q.num_speakers,
        cluster_threshold=q.cluster_threshold,
        num_threads=q.num_threads,
        provider=q.provider,
        language=q.language,
        temperature=q.temperature,
        short_segment_guard_enabled=q.short_segment_guard_enabled,
        short_segment_drop_sec=q.short_segment_drop_sec,
        short_segment_aba_max_mid_sec=q.short_segment_aba_max_mid_sec,
        short_segment_merge_same=q.short_segment_merge_same,
    )
    return _qwen3_singleton


def reset_qwen3_transcriber_singleton():
    """重置单例(仅测试用)"""
    global _qwen3_singleton
    _qwen3_singleton = None
