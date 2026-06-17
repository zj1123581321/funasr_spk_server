#!/usr/bin/env bash
# 安装版本控制的 git hooks (P4 A4 / P3 质量闸门)
# 用 core.hooksPath 指向 repo 内的 scripts/git-hooks, 让 hook 进版本控制、团队共享。
#
# 用法:  bash scripts/install-git-hooks.sh
# 卸载:  git config --unset core.hooksPath
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

chmod +x scripts/git-hooks/* 2>/dev/null || true
git config core.hooksPath scripts/git-hooks

echo "✓ 已设 core.hooksPath = scripts/git-hooks"
echo "  pre-push 现在会在 push 前跑 unit (碰 FunASR 路径再追加 parity)。"
echo "  紧急放行: FUNASR_SKIP_PREPUSH=1 git push"
echo "  卸载:     git config --unset core.hooksPath"
