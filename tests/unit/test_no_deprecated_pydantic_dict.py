"""#12 回归 guard: 禁止 Pydantic v2 弃用的 `.dict()` API。

Pydantic v2 下 `BaseModel.dict()` 仍可用但每次调用打 DeprecationWarning
(`PydanticDeprecatedSince20`), 日志噪声大。正解是 `.model_dump()`, 默认行为与旧 API
等价 (mode="python", enum/datetime 保持 python 对象)。

只盯 `.dict()`: 它在本仓零误报 (无 stdlib/httpx 对象有此方法)。弃用的 `.json()`
不进 guard —— 静态无法与 httpx `Response.json()` (合法) 区分。

本测试扫描 src/ 源码, 任何残留的 `.dict()` 调用都让它失败, 既驱动本次清理 (#12),
又防后续回归。
"""
import re
from pathlib import Path

# 项目根 = tests/unit/ 上两级
_SRC_DIR = Path(__file__).resolve().parents[2] / "src"

# 匹配 `<expr>.dict(` —— 前面紧贴 `)`/`]` 或标识符 (对象方法调用)。本仓内只有
# pydantic model 实例有 `.dict()`, 故无误报。
_DEPRECATED_CALL = re.compile(r"[\w\)\]]\.dict\s*\(")


def _scan_deprecated_calls() -> list[str]:
    """返回 `<相对路径>:<行号>: <行内容>` 列表, 命中弃用调用。"""
    offenders: list[str] = []
    for py_file in sorted(_SRC_DIR.rglob("*.py")):
        for lineno, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):  # 整行注释跳过
                continue
            if _DEPRECATED_CALL.search(line):
                rel = py_file.relative_to(_SRC_DIR.parent)
                offenders.append(f"{rel}:{lineno}: {stripped}")
    return offenders


def test_no_deprecated_pydantic_dict_calls():
    """src/ 内不得残留 `.dict()` (Pydantic v2 用 .model_dump())。"""
    offenders = _scan_deprecated_calls()
    assert not offenders, (
        "发现 Pydantic v2 弃用的 .dict() 调用, 请改用 .model_dump():\n  "
        + "\n  ".join(offenders)
    )
