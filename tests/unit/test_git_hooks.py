"""
P4 A4 — pre-push 质量闸 hook 的 sanity 测试

hook 是 shell, 不做完整 shell 单测;但它的**载荷逻辑 = FunASR 核心路径检测正则**,
错了会让"碰 FunASR 路径漏跑 parity"(质量闸失效)。本测试钉:
- hook / installer 文件存在且可执行
- pre-push 调用 run_checks.sh
- FUNASR_PATHS_RE 覆盖 CLAUDE.md 列的全部核心路径 + requirements.txt, 且不误判纯文档改动
"""
import os
import re
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts" / "git-hooks" / "pre-push"
INSTALLER = ROOT / "scripts" / "install-git-hooks.sh"


def _is_executable(p: Path) -> bool:
    return bool(p.stat().st_mode & stat.S_IXUSR)


def _extract_regex() -> str:
    for line in HOOK.read_text().splitlines():
        m = re.match(r"\s*FUNASR_PATHS_RE='([^']+)'", line)
        if m:
            return m.group(1)
    raise AssertionError("pre-push 未找到 FUNASR_PATHS_RE")


class TestHookFiles:
    def test_pre_push_exists_executable(self):
        assert HOOK.is_file(), f"缺 {HOOK}"
        assert _is_executable(HOOK), "pre-push 须可执行"

    def test_installer_exists_executable(self):
        assert INSTALLER.is_file(), f"缺 {INSTALLER}"
        assert _is_executable(INSTALLER), "installer 须可执行"

    def test_pre_push_calls_run_checks(self):
        body = HOOK.read_text()
        assert "scripts/run_checks.sh --parity" in body
        assert "scripts/run_checks.sh" in body
        assert "FUNASR_SKIP_PREPUSH" in body, "须有紧急放行开关"


class TestFunasrPathDetection:
    @pytest.mark.parametrize("path", [
        "src/models/schemas.py",
        "src/core/database.py",
        "src/core/task_manager.py",
        "src/api/websocket_handler.py",
        "src/core/funasr_transcriber.py",
        "src/core/transcriber_dispatch.py",
        "src/core/qwen3/diarize.py",
        "requirements.txt",
    ])
    def test_critical_paths_trigger_parity(self, path):
        rx = _extract_regex()
        assert re.search(rx, path), f"核心路径 {path} 未被检测 → 会漏跑 parity"

    @pytest.mark.parametrize("path", [
        "docs/开发/some-doc.md",
        "tests/unit/test_foo.py",
        "README.md",
    ])
    def test_non_critical_paths_skip_parity(self, path):
        rx = _extract_regex()
        assert not re.search(rx, path), f"非核心路径 {path} 不该触发 parity"
