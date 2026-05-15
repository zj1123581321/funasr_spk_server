"""
Transcriber Dispatch (PR1)
=============================
轻量 dispatch 函数 —— **不是** EngineRouter 类。

为什么不用 ABC + factory？
- Qwen3 的可行性还未通过 spike 验证（codex review T7）
- 在不确定需求上建抽象 = 过早优化
- PR1 阶段只需让 task.engine 字段端到端流转，30 行函数足够

未来 PR2（条件触发）会替换为完整的 ASREngine 抽象 + factory 注册表。

调用约定：
    transcriber = resolve_transcriber(task.engine)
    result = await transcriber.transcribe(audio_path, task_id, callback, output_format)

返回的 transcriber 对象签名保持和现有 FunASRTranscriber 一致：
    transcribe(audio_path, task_id, progress_callback, output_format)
    返回 (TranscriptionResult, raw_result) 元组（JSON 模式） | dict（SRT 模式）
"""
from typing import Optional, Any

from loguru import logger

from src.core.config import config


def resolve_transcriber(engine_name: Optional[str]) -> Any:
    """根据 engine 名解析到对应的 transcriber 单例

    Args:
        engine_name: 引擎标识。None 或空串视为未指定，回退到 config.transcription.default_engine

    Returns:
        transcriber 单例对象（鸭子类型；签名见模块注释）

    Raises:
        ValueError: engine_name 解析后仍为未知引擎名
    """
    name = (engine_name or "").strip()
    if not name:
        name = config.transcription.default_engine
        logger.debug(f"resolve_transcriber: 未指定引擎，使用默认 {name}")

    if name == "funasr":
        from src.core.funasr_transcriber import get_transcriber
        return get_transcriber()

    if name == "qwen3":
        from src.core.qwen3_transcriber import get_qwen3_transcriber
        return get_qwen3_transcriber()

    raise ValueError(
        f"未知的 ASR 引擎: {name!r}。"
        f"当前支持: funasr, qwen3（PR1 阶段 qwen3 是占位）"
    )
