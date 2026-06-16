#!/usr/bin/env bash
# 本地测试便利运行器 — 把"改 FunASR 路径必跑 parity"从口头约定变成一条命令。
#
# 设计来源: docs/开发/2026-06-16-可观测性仪表盘与测试加固-设计定案与落地计划.md (P3 质量闸门)。
# 目标: 问题在 dev 暴露, 不到生产。
#
# 用法 (在项目根目录):
#   bash scripts/run_checks.sh            # 仅 unit (毫秒级, 默认; push 前快速自检)
#   bash scripts/run_checks.sh --parity   # unit + FunASR parity (改 FunASR 路径后必跑)
#   bash scripts/run_checks.sh --all      # unit + 全部 integration (含 qwen3, 慢)
#
# 约定 (CLAUDE.md):
#   改 schemas / database / task_manager / websocket_handler / funasr_transcriber
#   后必跑 --parity。通过 = 改动安全; 失败 = 改坏了 FunASR 路径。
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$0")")"

PY="venv/bin/python"   # venv pytest binary shebang 漂移, 统一走 venv python -m pytest
MODE="${1:-unit}"

echo "==> unit tests (hermetic, 毫秒级)"
"$PY" -m pytest tests/unit/ -q

case "$MODE" in
  unit)
    echo "==> 完成 (仅 unit)。改了 FunASR 路径? 跑: bash scripts/run_checks.sh --parity"
    ;;
  --parity)
    echo "==> FunASR parity + ws e2e (真模型, ~2-3min)"
    # 显式钉 funasr 引擎 + 关 .env 引擎覆盖污染 (见 docs project_test_env_pitfalls)
    FUNASR_RUN_INTEGRATION=1 FUNASR_DEFAULT_ENGINE=funasr \
      "$PY" -m pytest \
      tests/integration/test_parity_funasr_semantic.py \
      tests/integration/test_funasr_server_websocket_e2e.py -q
    echo "==> parity 通过 = FunASR 路径改动安全"
    ;;
  --all)
    echo "==> 全部 integration (含 qwen3, 较慢; ort_parity 在 Mac 上可能挂, 见 docs)"
    FUNASR_RUN_INTEGRATION=1 "$PY" -m pytest tests/integration/ -q
    ;;
  *)
    echo "未知参数: $MODE (可选: 无参 / --parity / --all)" >&2
    exit 2
    ;;
esac
