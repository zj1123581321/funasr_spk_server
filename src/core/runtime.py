"""Runtime environment detection — 集中表达"环境感知配置".

提供:
- detect_runtime() → 按 platform + CUDA lib 探测 + env override 返回 RuntimeEnvironment
- RuntimeEnvironment 三个实现: MacRuntime / CudaRuntime / CpuRuntime
- validate() — 启动 sanity check, 缺关键 lib 时 fail-fast
- recommend_diarize_backend() — 给 dispatch 一个推荐 backend (sherpa / ort_cuda)
- recommend_num_threads() — 按 cpu_count 给 sherpa num_threads 推荐

设计目的:
- Mac 路径 100% 兼容当前行为, 不做任何破坏性改动
- Linux + CUDA 路径下推荐 ort_cuda diarize backend, 实现 wall RTF 阶跃下降
- 强制 escape hatch: FUNASR_RUNTIME=cpu 让 dev/CI 一键回退到 CPU baseline
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class RuntimeEnvironment(Protocol):
    """运行时环境的鸭子契约.

    所有 runtime 实现都暴露相同 API, 让 dispatch / config inject 不分支具体类型.
    """

    name: str

    def validate(self) -> None:
        """启动 sanity check, 关键 lib 缺失时 raise (替代 silent fallback)."""
        ...

    def recommend_diarize_backend(self) -> str:
        """返回 'sherpa' / 'ort_cuda' / ... 推荐值, 由 config 字段 + env override 兜底."""
        ...

    def recommend_num_threads(self) -> int:
        """根据物理核数算 sherpa num_threads 推荐."""
        ...


def _available_ort_providers() -> list[str]:
    """返回 onnxruntime 当前可用 provider 列表; 加载失败时返回空 list.

    跟 _has_cuda_runtime_available() 分开是为了让 validate() mock 更细 (能区分
    "ort 装了但 CUDA EP 缺" vs "ort 完全没装").
    """
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers())
    except Exception:
        return []


def _has_cuda_runtime_available() -> bool:
    """探测 onnxruntime-gpu CUDAExecutionProvider 是否可加载.

    在 import 阶段不真实 init session, 只看 ort.get_available_providers() 列表是否含 CUDA.
    silent fallback 留给 validate(), 这里只做能力探测.
    """
    return "CUDAExecutionProvider" in _available_ort_providers()


@dataclass
class MacRuntime:
    name: str = "mac_ane"

    def validate(self) -> None:
        return None

    def recommend_diarize_backend(self) -> str:
        return "sherpa"

    def recommend_num_threads(self) -> int:
        return 4


@dataclass
class CudaRuntime:
    name: str = "cuda"

    def validate(self) -> None:
        """fail-fast: ORT 默认会 silent 回 CPU, 这里显式校验 CUDAExecutionProvider 真的在 list 里."""
        providers = _available_ort_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "CudaRuntime.validate(): onnxruntime CUDAExecutionProvider 不可用. "
                f"available_providers={providers}. "
                "通常意味着 onnxruntime-gpu 未装、LD_LIBRARY_PATH 缺 cudnn/cublas, "
                "或者 cuda 版本不匹配. 用 FUNASR_RUNTIME=cpu 强制 fallback 或修依赖."
            )

    def recommend_diarize_backend(self) -> str:
        return "ort_cuda"

    def recommend_num_threads(self) -> int:
        return 2


@dataclass
class CpuRuntime:
    name: str = "cpu"

    def validate(self) -> None:
        return None

    def recommend_diarize_backend(self) -> str:
        return "sherpa"

    def recommend_num_threads(self) -> int:
        return 2


def detect_runtime() -> RuntimeEnvironment:
    """根据 sys.platform + CUDA 探测 + env override 选 runtime.

    env override 优先级最高 — 任何平台都可通过 FUNASR_RUNTIME=cpu / mac_ane / cuda
    强制走指定 runtime, 给 dev/CI 留 escape hatch.
    """
    forced = os.environ.get("FUNASR_RUNTIME", "").strip().lower()
    if forced == "cpu":
        return CpuRuntime()
    if forced == "mac_ane":
        return MacRuntime()
    if forced == "cuda":
        return CudaRuntime()

    if sys.platform == "darwin":
        return MacRuntime()
    if sys.platform.startswith("linux") and _has_cuda_runtime_available():
        return CudaRuntime()
    return CpuRuntime()
