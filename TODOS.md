# TODOS

记录 PR1 阶段明确不做、留待后续处理的事项。每条都来源于 `/plan-ceo-review` + `codex` 双重审稿沉淀，方案 v2（`docs/开发/重构计划-ASR引擎抽象.md`）第 12 节有完整背景。

## 状态约定
- **P1**：阻塞性，下一轮必做
- **P2**：重要但可延后
- **P3**：质量改进，机会做

---

## P1 — 阻塞 PR2 触发的前置条件

### 1. 跑通 Qwen3-ASR 1.7B spike
- 来源：codex review T7
- 入口：`spikes/qwen3_spike.py`
- 验证清单见 `spikes/README.md`
- **失败 = PR2 不触发**，重构停在 PR1（codex 的核心担忧）

### 2. 跑 FunASR parity 测试，确认 PR1 改动语义无回归
- 来源：codex review T4 + PR1 自身稳定性要求
- 入口：`FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`
- 首次跑会写 golden baseline 到 `tests/fixtures/golden/`；提交 baseline 后续每次重构都跑

---

## P2 — PR2 触发后落地（来自 codex review）

### 3. 异常分类纪律建立（codex T5）
- 现状：`FunASRTranscriber` 末端把所有错误重新包装成普通 `Exception("转录失败...")`；`TaskManager` 重试用字符串匹配判定 retryable
- 目标：替换为明确异常类层级
  - `EngineInitError`
  - `EngineTimeoutError`
  - `EngineExecutionError(original_exception)`
- 范围：`src/core/funasr_transcriber.py:354+`、`src/core/task_manager.py:296+`、`src/core/qwen3_transcriber.py`

### 4. Engine-level 资源配额（codex T10）
- 现状：FunASR pool 按 `max_concurrent_tasks` 启 worker；Qwen3 启用后无资源协调
- 目标：`config.engines.{name}.max_concurrent` + 调度器
- PR1 临时约束：同一时刻只有一个引擎驻留（lazy load + 切换时 shutdown 旧的）；正式落地必须强制执行

### 5. WebSocketHandler 拆分（codex T8）
- 现状：670 行单文件包揽 auth / upload / chunked upload / dispatch / cache 短路 / 通知
- 目标：拆为 3 个职责子模块：`AuthGuard` + `UploadSession` + `MessageRouter`
- 不入 PR1：扩大回归面无必要

### 6. `engines: dict` 嵌套配置结构（codex T9）
- 现状：PR1 用 `transcription.default_engine: str` + 独立 `qwen3` 块
- 目标：统一为 `config.engines: dict[str, EngineConfig]` 嵌套结构 + 配套 env override

---

## P2 — 长期偿债

### 7. 把 `file_based_process_pool.py` 替换为成熟库
- 现状：585 行自建 IPC + 大量 `.task/.ready` 文件舞蹈
- git log 显示至少 10 次相关补丁（commits `e1482d0`、`0f5a0f6`、`605c31d` 等）
- 候选：`multiprocessing.Pool` / `billiard` / `joblib` / `ray`
- 何时合适：PR2 落地后再做，避免在引擎抽象未稳定前动核心 IPC

### 8. 给 `FunASR` pool 进程僵死加 regression test
- 现状：靠 30s 巡检 + worker auto-restart 兜底，无测试
- 是 codex T6 的延伸

---

## P3 — 质量改进

### 9. `concurrency_mode` 死代码删除
- `lock` / `thread_pool` 实质废弃（生产唯一路径是 `pool`）
- 删之前确认无 env override 在用

### 9b. 清理 src/ 内 Windows 平台分支
- 根目录第三轮整理已删 Windows wrapper + Docker 文件，但 src/ 内还有跨平台代码：
  - `src/main.py:180-182` Windows event loop policy
  - `src/core/file_based_process_pool.py:114` Windows 特殊处理
  - `src/utils/platform_utils.py` 整个文件（161 行，含 Windows / Linux 分支）
  - `requirements.txt` `uvloop` 的 `sys_platform != 'win32'` 条件
- 删之前确认：删除后是否影响 mac/macOS pool 模式的工作（platform_utils 可能被其他地方依赖）

### 10. `tests/manual/` 中诊断脚本逐步转写为 pytest regression
- 重点：`tests/manual/diagnostics/test_mps_*.py`（MPS 历史 bug 的复现脚本）
- 转写后 retire 旧脚本

### 11. cam++ 剥离为独立 SpeakerDiarizer Stage
- 当前 D2 决策：两引擎都视为「ASR+说话人」黑盒
- 何时合适：用户的说话人方案选型完成 + Qwen3 集成稳定后

### 12. Pydantic 弃用 API 清理
- 多处用 `.json()` / `.dict()`，Pydantic V2 已弃用
- 替换为 `.model_dump_json()` / `.model_dump()`
- 范围：`src/core/database.py`、`src/api/websocket_handler.py`、`src/core/task_manager.py`、`src/core/funasr_transcriber.py`

### 13. venv 路径漂移导致 pytest binary shebang 失效
- 现状：`venv/bin/pytest` 的 shebang 写死了旧路径 `funasr_spk_server_dev`
- 当前 workaround：用 `venv/bin/python -m pytest`
- 修法：重建 venv 或 `pip install --force-reinstall pytest`（前提 venv 内有 pip）

---

## 已完成（PR1）

- [x] schemas 加 engine 字段（向后兼容默认 funasr）
- [x] database engine-aware：cache 表加 engine 列 + 旧 schema 自动迁移
- [x] `transcriber_dispatch.resolve_transcriber()`（30 行薄函数，非 ABC）
- [x] Qwen3 占位 transcriber（PR1 阶段 NotImplementedError）
- [x] task_manager 端到端 engine 流转
- [x] websocket_handler 三个 cache lookup 路径 + chunked session 透传
- [x] config `default_engine` 字段 + env override
- [x] pytest 基础设施 + conftest + 3 条测试音频
- [x] semantic parity 测试脚手架（默认 skip，需 env 显式启用）
- [x] Qwen3 spike 脚本 + report 模板
- [x] 旧 19 个手工脚本搬到 `tests/manual/`
- [x] .gitignore 修复（`models` 裸规则误匹配 `src/models`、音频 fixture 白名单）
