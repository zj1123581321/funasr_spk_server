"""
Transcriber Dispatch — 全局唯一引擎模式

策略:
- 服务器启动时由 config.transcription.default_engine 决定**全局唯一**引擎
  (历史字段名沿用 default_engine, 语义上等同于 "engine")
- upload_request.engine 字段只是用于 client-side 表达期望, 严格 validate:
  * 不传 / 空串 → 走 server engine, 通过
  * 等于 server engine → 通过
  * 不等于 → ValueError(明确写 server 配的是 X, requested 是 Y)
- 全局只 instance 一个 transcriber 单例

为什么不再走 ABC 抽象?
- PR2 阶段 Qwen3 已经落地, 但只有 2 个引擎, 不引入 EngineRouter 类抽象
- 见 docs/开发/重构计划-ASR引擎抽象.md 第 8 节

调用约定:
    transcriber = resolve_transcriber(upload_request.engine)
    result = await transcriber.transcribe(audio_path, task_id, callback, output_format)

返回的 transcriber 对象签名(鸭子类型, FunASR / Qwen3 同形):
    transcribe(audio_path, task_id, progress_callback, output_format)
    JSON 模式 -> (TranscriptionResult, raw_result) 元组
    SRT  模式 -> dict {format, content, file_name, file_hash, duration, processing_time, raw_result}
"""
from typing import Any, Optional

from loguru import logger

from src.core.config import config

# 注册表: 引擎名 -> get_singleton 函数. 引擎名变动时只改这里
_ENGINE_REGISTRY = {
    "funasr": lambda: __import__("src.core.funasr_transcriber", fromlist=["get_transcriber"]).get_transcriber(),
    "qwen3":  lambda: __import__("src.core.qwen3_transcriber", fromlist=["get_qwen3_transcriber"]).get_qwen3_transcriber(),
}


def resolve_transcriber(requested_engine: Optional[str]) -> Any:
    """根据 server engine + requested engine 解析到对应 transcriber 单例.

    Args:
        requested_engine: client 通过 upload_request.engine 传来的引擎名.
                          None / "" 视为未指定, 走 server 配置.

    Returns:
        transcriber 单例对象(鸭子类型, 见模块注释).

    Raises:
        ValueError: server engine 未知 OR requested engine 与 server 不匹配.
    """
    server_engine = config.transcription.default_engine
    requested = (requested_engine or "").strip()

    if requested and requested != server_engine:
        raise ValueError(
            f"Server configured with engine={server_engine!r}, "
            f"cannot accept engine={requested!r}. "
            f"Please omit the engine field or set it to {server_engine!r}."
        )

    if not requested:
        logger.debug(f"resolve_transcriber: 未指定引擎, 走 server 配置 {server_engine!r}")

    factory = _ENGINE_REGISTRY.get(server_engine)
    if factory is None:
        raise ValueError(
            f"未知 ASR 引擎: {server_engine!r}. 当前支持: {list(_ENGINE_REGISTRY)}. "
            f"请检查 config.transcription.default_engine 或 FUNASR_DEFAULT_ENGINE 环境变量."
        )
    return factory()
