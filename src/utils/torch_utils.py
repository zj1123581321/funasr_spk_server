"""
与 PyTorch 设备相关的通用工具函数
"""
from __future__ import annotations

import gc
from typing import Callable, Dict, Optional


def _bytes_to_mb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 * 1024), 2)


def _safe_invoke(fn, *args, **kwargs) -> Optional[int]:
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn()
        except Exception:
            return None
    except Exception:
        return None


def collect_torch_memory_stats() -> Dict[str, float]:
    """
    收集当前进程相关的内存指标，覆盖 RSS、MPS 与 CUDA 运行时
    """
    stats: Dict[str, float] = {}

    # 进程常驻内存
    try:
        import psutil

        process = psutil.Process()
        stats["rss_mb"] = _bytes_to_mb(process.memory_info().rss)
    except Exception:
        pass

    try:
        import torch  # 延迟导入
    except ImportError:
        return stats

    # Apple Silicon MPS 指标
    if hasattr(torch, "backends") and getattr(torch.backends, "mps", None):
        mps_backend = torch.backends.mps
        if getattr(mps_backend, "is_available", lambda: False)():
            allocated = _bytes_to_mb(
                _safe_invoke(getattr(torch.mps, "current_allocated_memory", lambda: None))
            )
            reserved = _bytes_to_mb(
                _safe_invoke(getattr(torch.mps, "current_reserved_memory", lambda: None))
            )
            driver = _bytes_to_mb(
                _safe_invoke(getattr(torch.mps, "driver_allocated_memory", lambda: None))
            )

            if allocated is not None:
                stats["mps_allocated_mb"] = allocated
            if reserved is not None:
                stats["mps_reserved_mb"] = reserved
            if driver is not None:
                stats["mps_driver_mb"] = driver

    # CUDA 指标（为未来扩展保留，当前 Mac MPS 环境不会触发）
    if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)):
        try:
            if torch.cuda.is_available():
                stats["cuda_allocated_mb"] = _bytes_to_mb(
                    _safe_invoke(torch.cuda.memory_allocated)
                )
                stats["cuda_reserved_mb"] = _bytes_to_mb(
                    _safe_invoke(torch.cuda.memory_reserved)
                )
        except Exception:
            pass

    return stats


def _format_stats_diff(before: Dict[str, float], after: Dict[str, float]) -> str:
    keys = sorted(set(before.keys()) | set(after.keys()))
    segments = []

    for key in keys:
        before_val = before.get(key)
        after_val = after.get(key)
        if before_val is None and after_val is None:
            continue
        segments.append(
            f"{key}: "
            f"{before_val if before_val is not None else 'N/A'} → "
            f"{after_val if after_val is not None else 'N/A'} MB"
        )

    return " | ".join(segments)


def release_accelerator_memory(
    tag: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    在一次推理完成后主动释放加速设备缓存，避免 MPS/CUDA 内存占用持续增长

    Args:
        tag: 可选的标识字符串，用于日志或调试
        log_fn: 可选的日志函数，用于输出释放前后的统计信息

    Returns:
        dict: {"before": {...}, "after": {...}}，包含释放前后的内存指标
    """
    try:
        import torch  # 延迟导入，确保环境变量已就绪
    except ImportError:
        return {"before": {}, "after": {}}

    before = collect_torch_memory_stats()

    # 清理 Apple Silicon MPS 设备缓存
    if hasattr(torch, "backends") and getattr(torch.backends, "mps", None):
        try:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception as mps_error:
            identifier = f"[{tag}]" if tag else ""
            message = f"{identifier}释放 MPS 缓存失败: {mps_error}"
            if log_fn:
                log_fn(message)
            else:
                print(message)

    # 可扩展到 CUDA/其他后端
    if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # 回收 Python 对象引用，进一步减小常驻内存
    gc.collect()

    after = collect_torch_memory_stats()

    if log_fn:
        prefix = f"[{tag}] " if tag else ""
        log_fn(f"{prefix}加速缓存释放完成 | {_format_stats_diff(before, after)}")

    return {"before": before, "after": after}
