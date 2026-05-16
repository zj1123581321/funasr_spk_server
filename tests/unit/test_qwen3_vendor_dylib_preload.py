"""
PR1: vendor dylib preload 保护性 smoke 测试

背景: 新版 ggml 把 backend 拆成多个 dylib (cpu/blas/metal). libggml.dylib 通过
@rpath 引用它们; nohup/SIP 等场景 DYLD_LIBRARY_PATH 会被 strip, 必须由
bind_llama_lib() 显式 ctypes.CDLL preload 使后端 symbol 进入全局 namespace.

本文件只验证"已经修复的行为不被悄悄回退":
- vendor/bin/ 下关键 backend dylib 物理存在 (防文件被误删)
- bind_llama_lib() 调用不抛异常 (防 preload 列表回退)
- bind_llama_lib() 幂等 (防重复加载触发 dyld 警告)

无 production 改动 — 测试已存在行为, commit 信息需注明.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


VENDOR_BIN = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "core"
    / "vendor"
    / "qwen_asr_gguf"
    / "inference"
    / "bin"
)

DARWIN_BACKEND_DYLIBS = [
    "libggml-base.dylib",
    "libggml-cpu.dylib",
    "libggml-metal.dylib",  # Apple Silicon GPU 后端
]


class TestVendorBinHasBackendDylibs:
    """vendor/bin/ 下关键 backend dylib 必须物理存在."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="dylib 仅 macOS 适用")
    @pytest.mark.parametrize("dylib_name", DARWIN_BACKEND_DYLIBS)
    def test_required_darwin_backend_dylib_exists(self, dylib_name: str) -> None:
        dylib_path = VENDOR_BIN / dylib_name
        assert dylib_path.exists(), (
            f"vendor backend dylib 缺失: {dylib_path}. "
            f"PR4 修复要求 darwin 平台必须 preload 这些 backend dylib, "
            f"如果文件不在 — 要么 vendor 没装完整, 要么 rpath 已经改了别的方案."
        )

    @pytest.mark.skipif(sys.platform != "darwin", reason="dylib 仅 macOS 适用")
    def test_main_llama_dylib_exists(self) -> None:
        """libllama.dylib 是核心入口, 必须存在."""
        assert (VENDOR_BIN / "libllama.dylib").exists()


class TestBindLlamaLibLoadsWithoutCrash:
    """bind_llama_lib() 调用必须成功 — 这是 PR4 dylib preload 修复的精确入口."""

    @pytest.mark.skipif(sys.platform != "darwin", reason="dev 路径仅在 macOS 验证")
    def test_bind_llama_lib_first_call_does_not_crash(self) -> None:
        """首次调用 bind_llama_lib() 必须能加载完所有 backend + 主 libs.

        这个测试一旦红 — 几乎可以肯定是 preload 列表退回 (libggml-cpu 或 metal
        被从 _preload_dlls 里删了), 或者 RTLD_GLOBAL 模式被改成 RTLD_LOCAL.
        """
        from src.core.vendor.qwen_asr_gguf.inference.llama import bind_llama_lib

        bind_llama_lib()

    @pytest.mark.skipif(sys.platform != "darwin", reason="dev 路径仅在 macOS 验证")
    def test_bind_llama_lib_is_idempotent(self) -> None:
        """重复调用 bind_llama_lib() 不抛 — 防 short-circuit (if llama is not None: return) 被改坏."""
        from src.core.vendor.qwen_asr_gguf.inference.llama import bind_llama_lib

        bind_llama_lib()
        bind_llama_lib()
        bind_llama_lib()
