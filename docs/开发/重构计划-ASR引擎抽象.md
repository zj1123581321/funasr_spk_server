# 重构计划：ASR 引擎并存（PR1 最小闭环 + PR2 可选抽象）

> **版本**：v2（基于 codex 独立 review 反馈大幅修订）
> **生成于**：2026-05-15
> **基于**：`/plan-ceo-review` + `codex exec` 双重审稿
> **当前分支**：dev → 已合并 main（`40d1a9a`）
> **核心修订**：把原「窄版 B」进一步收窄为 PR1 最小闭环；PR2 是否做 ASREngine 完整抽象，由 PR1 跑通 + Qwen3 spike 结果决定。
>
> ## 🟢 状态:PR1 + PR2 + PR3 均已完成 (2026-05-15)
>
> **PR1** (`40d1a9a` 已合并 main):
> - 13 commits, 38 个 unit test, 3 个 FunASR parity test 全绿
> - golden baseline 落档 `tests/fixtures/golden/`
>
> **PR2** (`spike/qwen3-diarize-poc` 分支):
> - Qwen3 spike 通过 (RTF 0.108-0.118, 2/4 人未知场景验证通过, 见 `spikes/qwen3_diarize/spike_report.md`)
> - 13 commits 落地 Qwen3-Diarize 引擎集成: vendor + 三模块 + Transcriber + 全局唯一引擎 dispatch + config + 集成测试 + 文档
> - 58 个 unit test 全绿(含 Qwen3 11/ dispatch 10/ task_manager 10/ database 12/ websocket 3/ schemas 4/ config 8)
> - 5 个 integration test 全绿(FunASR parity 3 + Qwen3 e2e 2)
> - Qwen3 e2e 实测: 60s 双人音频 RTF 0.118 (与 PoC 报告一致)
> - 生产切换方式: `FUNASR_DEFAULT_ENGINE=qwen3` + `bash scripts/download_qwen3_models.sh` 即生效
>
> **PR3** (`spike/qwen3-diarize-poc` 分支, 2026-05-15):
> - **Qwen3 从同进程单 instance 升级到 multi-process worker pool** (解决 libllama 并发 unsafe)
> - 主进程不再持有 libllama context, 每个 worker subprocess 独立加载 GGUF + Metal + sherpa
> - 池大小独立: FunASR 用 `max_concurrent_tasks=2`, Qwen3 用 `qwen3_pool_size=3`
> - 任务目录物理隔离: FunASR 用 `temp/tasks/`, Qwen3 用 `temp/tasks_qwen3/`(避免 PM2 daemon 抢任务)
> - 7 commits + 30 个新增 test:
>   - unit (23): `test_file_based_pool_engine_aware.py` (6), `test_qwen3_worker_process.py` (6), `test_qwen3_pool_transcriber.py` (12, 含 task_dir 隔离断言), `test_config_qwen3_pool_size.py` (6 → 实际 6), `test_transcriber_dispatch.py` (+3 PR3 路由测试)
>   - integration (4 hybrid + 4 real): `test_qwen3_pool_concurrent_dispatching.py` (hybrid 真 pool + mock worker, 4 测试, 默认 enabled), `test_qwen3_pool_real_concurrency.py` (FUNASR_RUN_INTEGRATION=1, 4 测试, N=2 真 Qwen3 模型)
> - 真 e2e 实测: 串行 1 task ~15s, 并发 2 task ~22s (ratio 1.5x serial, 短音频 overhead 占比大), N=2 worker 总 RSS ~6.9GB (GGUF mmap 共享)
> - 关键架构决策(都写死, 见 `docs/开发/集成-Qwen3-多Worker池-新session-prompt.md`):
>   1. 复用 `FileBasedProcessPool`(加 `worker_entry_script` + `task_dir` 参数), 不另起炉灶
>   2. 不引入 ABC 抽象, 注册表式扩展
>   3. 复用 health monitor + auto restart, 不重写
>   4. 删除单 instance 路径(主进程 dispatch 只走 pool), 但 `Qwen3DiarizeTranscriber` 类保留供 worker 内部使用
>   5. 池大小独立字段 (FunASR=2, Qwen3=3) + 任务目录独立 (避免抢任务)
>   6. Qwen3 worker entry 独立 (`src/core/qwen3_worker_process.py`), FunASR `worker_process.py` 零侵入

---

## 0. 修订说明：为什么是两个 PR

v1 草案曾设计「完整 `ASREngine` 接口 + `EngineRouter` + `factory` + 三层 contract test 体系」。codex 独立 review 指出 10 条实质问题，核心批评是：

> 「**Qwen3 是否真能等价输出 speaker，是最大的未验证假设。在验证前建抽象体系，是过早优化。**」

因此 v2 拆成两个 PR：
- **PR1（必做）**：解决「让 FunASR 和 Qwen3 共存」所需的**最小代码改动** + 一个 Qwen3 可行性 spike 脚本
- **PR2（条件触发）**：等 PR1 + spike 跑通后，**如果**长期共存确定要做且要加第三个引擎，再补 `ASREngine` 抽象 + contract test 体系

这样所有架构决策都推迟到「Qwen3 真跑起来」之后再做。

---

## 1. 目标 & 非目标

### 1.1 PR1 目标
1. **现网零行为变化**：default engine 仍是 FunASR，不带 `engine` 字段的上传请求行为不变
2. **engine 字段端到端流转**：`upload_request.engine` → `TranscriptionTask.engine` → 选择 transcriber 实现
3. **缓存 engine-aware**：缓存 key 包含 `engine`，避免灰度时命中错引擎缓存
4. **FunASR 薄 adapter**：把现有 `FunASRTranscriber` 调用包一层，**保留现有返回结构** `(TranscriptionResult, raw_result)`，**不引入 ABC 接口**
5. **Qwen3 spike**：独立 spike 脚本验证 Qwen3-ASR 1.7B 能否（在外部打包的 speaker 方案下）输出 `text + timestamps + speaker`
6. **pytest 基础设施**：引入 `pytest.ini` + `conftest.py`，能跑起来；旧 19 个手工脚本搬到 `tests/manual/`
7. **Parity 测试 1 个**：semantic-level（不是 byte-equal）验证 PR1 改完 FunASR 路径输出和原版语义一致

### 1.2 PR2 目标（条件触发，**不在 PR1 范围**）
- ASREngine 抽象基类 + capabilities
- EngineRouter / factory
- Engine contract test 体系
- 异常分类重构（替换现有 `except Exception` + 字符串匹配）
- engine-level 资源隔离机制（max_concurrent per engine）
- WebSocketHandler 670 行拆分

### 1.3 非目标（v1/v2 都明确不做）
- ❌ 重写 task_manager.py / database.py / file_based_process_pool.py / mps_patch.py 主体
- ❌ FastAPI / SQLAlchemy / HTTP REST / Web 前端 / 多用户 / 流式 ASR
- ❌ cam++ 剥离为独立 Stage
- ❌ 引入新的执行框架依赖

---

## 2. Codex 10 条意见处理表

| # | Codex 发现 | PR1 是否处理 | 如何处理 |
|---|---|---|---|
| T1 | task_manager 不动 + engine 字段 → 字段传不到 | ✅ PR1 | TaskManager / FileUploadRequest / TranscriptionTask 都加 engine 字段 |
| T2 | cache key 只用 file_hash → 灰度命中错缓存 | ✅ PR1 | `transcription_cache` 表加 `engine` 列；UNIQUE 改为 `(file_hash, engine, output_format)` |
| T3 | transcribe 返回 `(TranscriptionResult, raw_result)` / dict，原方案接口收窄会丢 raw_result | ✅ PR1 | 薄 adapter 保留现有返回结构，不引入 ABC，**不收窄签名** |
| T4 | byte-equal parity 不现实 | ✅ PR1 | 改 semantic parity：忽略 created_at / processing_time，文本完全相等，时间窗容差 ±50ms，speaker 数量一致 |
| T5 | 现有 `except Exception` + 字符串匹配重试，新异常类不会自然生效 | ⏸ PR2 | PR1 保持现有错误处理模式，不改异常体系 |
| T6 | 方案文档把超时画在 task_manager，实际在 pool 内部 | ✅ PR1（仅修文档） | 第 4 节数据流图已修正 |
| T7 | Qwen3「ASR+说话人黑盒」是未验证假设 | ✅ PR1 | spike 脚本验证可行性，输出 spike report |
| T8 | WebSocket 拆分应另开 PR | ✅ PR1（移除） | 从 PR1 范围移除，进 TODOS |
| T9 | Pydantic Config 加 engines dict 涉及 env override 等被低估 | ✅ PR1 | 第 5.5 节详述 Pydantic 迁移方案 |
| T10 | FunASR + Qwen3 并存时内存/MPS/CPU 谁分配 | ⚠ PR1 仅声明，PR2 落地 | PR1 文档声明：Qwen3 默认禁用 + per-request 切换时全局只跑一个引擎；engine-level 并发预算放 PR2 |

---

## 3. 架构变化（PR1）

### 3.1 修订后的现状理解（基于 codex 提示重新读源码）
```
  WebSocketHandler.handle_upload_request
      ↓
  TaskManager.create_task(FileUploadRequest)
      ├─ DBManager.get_cached_result(file_hash) ◀── cache key 只有 file_hash ← T2
      └─ enqueue TranscriptionTask
      ↓
  TaskManager._worker → _process_task
      ↓
  get_transcriber()  ← 全局单例
      ↓
  FunASRTranscriber.transcribe(audio_path, task_id, callback, output_format)
      ↓ 内部分支
      ├─ if output_format == "json"  → return (TranscriptionResult, raw_result_dict)
      └─ if output_format == "srt"   → return srt_dict
      ↓
  TaskManager._process_task 把 raw_result 也存进 DB（用于格式转换）

  超时实际发生在：
  FunASRTranscriber.transcribe → model_pool.generate_with_pool → 内部 asyncio.wait_for(..., timeout=task_timeout_minutes*60)
                                                                  ▲
                                                                  这里抛 TimeoutError，不在 TaskManager
```

### 3.2 PR1 目标流（diff 视角）
```
  WebSocketHandler.handle_upload_request
      ↓
  TaskManager.create_task(FileUploadRequest)            ◀── FileUploadRequest 加 engine: Optional[str]
      ├─ DBManager.get_cached_result(file_hash, engine) ◀── (T2 修复)
      └─ enqueue TranscriptionTask                       ◀── TranscriptionTask 加 engine
      ↓
  TaskManager._worker → _process_task
      ↓
  resolve_transcriber(task.engine)                      ◀── 薄 dispatch 函数（不是 EngineRouter 类）
      ├─ engine in (None, "funasr")  → get_funasr_transcriber()  (现有 get_transcriber)
      └─ engine == "qwen3"            → get_qwen3_transcriber()   (PR1 阶段不启用，spike 后再决定)
      ↓
  transcriber.transcribe(audio_path, task_id, callback, output_format)
      ↓
  返回结构不变：(TranscriptionResult, raw_result) | srt_dict
      ↓
  TaskManager._process_task 保存到 DB（保存逻辑加 engine 字段）
```

**新增的代码量预估**：
- `schemas.py`：FileUploadRequest / TranscriptionTask 各加 1 个 optional 字段
- `task_manager.py`：`create_task()` 签名加 engine、`_process_task()` 调 `resolve_transcriber()`
- `database.py`：表 schema 加 1 列 + UNIQUE 改 + `get_cached_result` / `save_result` 签名加 engine
- `funasr_transcriber.py`：不变（仍是 `get_transcriber()` 返回单例）
- 新增 `src/core/transcriber_dispatch.py`：~30 行薄函数 `resolve_transcriber(engine_name)`
- 新增 `src/core/qwen3_transcriber.py`：占位骨架（PR1 阶段只有 `NotImplementedError`，spike 通过后再实现）
- `websocket_handler.py`：消息解析加 `engine` 字段透传到 FileUploadRequest

总改动约 200-300 行。

---

## 4. PR1 数据流（修正 T6 后的准确版本）

```
upload_request(file, engine?)
      ↓
WebSocketHandler 验证 + 解析（engine 字段透传）
      ↓
TaskManager.create_task
      ├─ DBManager.get_cached_result(file_hash, engine, output_format)  ◀── cache key 三联
      │     命中 → 返回缓存，结束
      └─ enqueue TranscriptionTask(engine=...)
      ↓
TaskManager._worker 派发到 _process_task
      ↓
resolve_transcriber(task.engine) → transcriber 实例
      ↓
transcriber.transcribe(...)
      │
      ├── happy: 返回 (TranscriptionResult, raw_result) → 存 DB（含 engine）→ 推 client
      │
      ├── pool 内部 TimeoutError ──▶ task_manager 既有 retry 逻辑（最多 2 次，基于字符串匹配）
      │     重试耗尽 → 任务 FAILED → 通知 client + 企微
      │
      ├── 转录任意 Exception ──▶ task_manager 既有 except Exception
      │     → 重试或失败
      │
      └── worker 进程僵死 ──▶ FileBasedProcessPool 30s 巡检兜底（既有逻辑）
            → 重启 worker → 任务继续等待 → 超时后归入上面 TimeoutError 路径
```

**关键修正**（vs v1）：超时不在 TaskManager 层，而在 `FileBasedProcessPool.generate_with_pool()` 内部用 `asyncio.wait_for` 包裹。`cancel_task()` 只改任务状态，不会中断正在跑的 worker 进程。这两点 PR1 不动。

---

## 5. PR1 具体改动详解

### 5.1 schemas.py（修 T1）
```python
class FileUploadRequest(BaseModel):
    file_name: str
    file_size: int
    file_hash: str
    force_refresh: bool = False
    output_format: str = "json"
    engine: Optional[str] = None  # 新增：None → 走 config.default_engine


class TranscriptionTask(BaseModel):
    # ... 既有字段 ...
    engine: str  # 新增：在 create_task 时根据 request.engine 或 default_engine 解析后填入
```

### 5.2 task_manager.py（修 T1）
**改动点**：
1. `create_task(request)` 增加：`engine = request.engine or config.transcription.default_engine`，存进 TranscriptionTask
2. `create_task` 调 `DBManager.get_cached_result` 时传 `engine`（修 T2）
3. `_process_task` 把 `get_transcriber()` 改为 `resolve_transcriber(task.engine)`
4. 保存结果到 DB 时传 `engine`

**不动**：异常处理、重试逻辑、超时机制都保持原样（T5 留给 PR2）

### 5.3 database.py（修 T2）
```sql
-- 迁移脚本（一次性）
ALTER TABLE transcription_cache ADD COLUMN engine TEXT NOT NULL DEFAULT 'funasr';
DROP INDEX IF EXISTS ux_cache_file_hash;
CREATE UNIQUE INDEX ux_cache_file_hash_engine_format
  ON transcription_cache(file_hash, engine, output_format);
```

**Python 改动**：
- `init_db()` 加 schema 检测 + 迁移（检查 `engine` 列是否存在，不在则 ALTER）
- `get_cached_result(file_hash, engine, output_format)` 签名扩展
- `save_result(..., engine)` 签名扩展
- 现有调用方全部传 `engine="funasr"` 作为默认值（向后兼容）

**风险**：现有 SQLite 文件需要在线迁移。迁移在 `init_db` 启动时执行，加事务保护。提供 `--no-migrate` 启动参数兜底（如果迁移失败可降级）。

### 5.4 transcriber_dispatch.py（新增，~30 行）
```python
# src/core/transcriber_dispatch.py
"""
轻量 dispatch 函数（不是 EngineRouter 类）
PR1 阶段意图：让 task.engine 能选到正确的 transcriber 单例
不引入 ABC 抽象，不引入 factory 注册表，等 PR2 再说
"""
from typing import Protocol
from src.core.config import config


def resolve_transcriber(engine_name: str):
    """根据 engine 名返回对应 transcriber 单例"""
    name = engine_name or config.transcription.default_engine
    if name == "funasr":
        from src.core.funasr_transcriber import get_transcriber
        return get_transcriber()
    if name == "qwen3":
        from src.core.qwen3_transcriber import get_qwen3_transcriber
        return get_qwen3_transcriber()
    raise ValueError(f"未知的 ASR 引擎: {name}")
```

### 5.5 config.py 修订（修 T9）
**v1 漏掉的**：
- `Config` 是 Pydantic 模型，加字段要走 env override、`load_config()` 打印、worker 进程读取一致性
- `engines` 嵌套结构涉及子 Pydantic 模型

**PR1 做法（最小改动）**：
```python
class TranscriptionConfig(BaseModel):
    # ... 既有字段 ...
    default_engine: str = "funasr"  # 新增
    # env override: FUNASR_DEFAULT_ENGINE
```

**暂不引入 `engines: dict` 嵌套结构**。Qwen3 自己的配置（model path、device 等）放 `qwen3` 顶层块或直接 env 变量，等 PR2 落地完整抽象时再统一进 `engines` 子结构。

worker 进程读取：`config.transcription.default_engine` 已通过现有 config singleton 加载，worker 启动时会重新构建 config，自动读到新字段。

### 5.6 spike：Qwen3 可行性验证（修 T7，**最重要的 PR1 产物**）
**新增文件**：`spikes/qwen3_spike.py`（不在 src/ 下，不进生产）

**验证清单**：
- [ ] 能加载 Qwen3-ASR 1.7B 模型（确认 modelscope / HuggingFace 名称、下载、加载耗时、显存/内存占用）
- [ ] 能对一条 5s 测试音频跑推理
- [ ] 输出格式：是否带 timestamps？是否带 speaker？如果不带，"外部打包说话人方案"是什么？
- [ ] 单进程并发安全性：能否在同一 Python 进程内同时跑两个 transcribe（asyncio）？
- [ ] 资源占用：单次推理峰值内存、是否能跑 MPS、CPU 占用
- [ ] 集成复杂度估算：从 spike 到 production-ready Qwen3Transcriber 还要多少工作？

**Deliverable**：`spikes/qwen3_spike_report.md`，含上述每项的实测数据和结论

**spike 失败的处理**：如果 spike 发现 Qwen3 不可行或代价远超预期，PR1 仍保持完成（最小闭环已建立），但 PR2 永远不触发，整个重构停在 PR1。这是 codex 担心的「过早抽象」的反面 — 不抽象的成本极低。

### 5.7 测试基础设施（pytest 真正启用）
**新增**：
- `pytest.ini`：`testpaths = tests/unit tests/integration`（manual 排除）
- `tests/conftest.py`：基础 fixtures（mock config, sample audio path）
- `tests/fixtures/audio/`：3 条小测试音频（用户提供）
- `tests/fixtures/golden/`：跑当前 FunASR 输出的 baseline 快照
- `tests/unit/test_transcriber_dispatch.py`：dispatch 路由测试（mock get_transcriber）
- `tests/unit/test_database_cache_engine_aware.py`：cache key 三联唯一性测试
- `tests/integration/test_parity_funasr_semantic.py`：semantic parity 测试（修 T4）

**搬迁**：现有 `tests/` 下所有 19 个手工脚本 → `tests/manual/`，加 README 说明这些不在 pytest 收集范围

**Semantic parity 算法**（修 T4）：
```python
def assert_semantic_equal(actual: TranscriptionResult, golden: dict):
    # 文本：严格相等
    assert [s.text for s in actual.segments] == [s["text"] for s in golden["segments"]]
    # 时间窗：容差 ±50ms（防止模型推理浮点波动）
    for a, g in zip(actual.segments, golden["segments"]):
        assert abs(a.start_time - g["start_time"]) <= 0.05
        assert abs(a.end_time - g["end_time"]) <= 0.05
    # speaker：数量一致，标签可重映射（speaker1/2 vs A/B 都接受）
    assert len(actual.speakers) == len(golden["speakers"])
    # 忽略：created_at, processing_time, task_id
```

### 5.8 资源隔离（PR1 仅声明，T10）
PR1 阶段，Qwen3 的 transcriber 加载是 lazy 的（`get_qwen3_transcriber()` 首次调用才加载）。

**约束声明**：PR1 阶段同一时刻**只允许一个引擎被加载**。即：
- 启动时根据 `default_engine` 加载对应 transcriber
- 如果有 per-request `engine` 切换，**触发一次完整 shutdown 旧引擎 + load 新引擎**（代价：可能 10-30s）
- 不支持两引擎同时驻留

**这个约束是临时的**，PR2 落地 engine-level 资源配额后取消。文档明示，避免误用。

---

## 6. PR1 数据迁移 & 回滚

### 6.1 SQLite schema 迁移
启动时 `init_db()`：
```python
async def init_db(self):
    # ... 现有初始化 ...
    # 检测并迁移 engine 列
    cursor = await db.execute("PRAGMA table_info(transcription_cache)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "engine" not in cols:
        async with db.transaction():
            await db.execute(
                "ALTER TABLE transcription_cache ADD COLUMN engine TEXT NOT NULL DEFAULT 'funasr'"
            )
            await db.execute("DROP INDEX IF EXISTS ux_cache_file_hash")
            await db.execute(
                "CREATE UNIQUE INDEX ux_cache_file_hash_engine_format "
                "ON transcription_cache(file_hash, engine, output_format)"
            )
            logger.info("数据库迁移：transcription_cache 增加 engine 列")
```

### 6.2 回滚预案
- PR1 整个回滚：`git revert` 整个 PR；现有 SQLite 数据库的 `engine` 列保留（不影响旧代码读，因为 SQLite ALTER ADD COLUMN 是非破坏性的）
- 启动失败：`config.transcription.default_engine` 改回 `"funasr"`（其实默认就是）
- 引擎切换异常：client 不传 `engine` 字段就回退默认

---

## 7. PR1 错误与救援表（仅新增/改动部分）

| 路径 | 失败模式 | 异常类（现有） | 救援动作 | 用户可见 | 已计划测试 |
|------|----------|----------------|----------|----------|------------|
| `resolve_transcriber()` | 未知 engine 名 | `ValueError` | TaskManager 现有 except 捕获 → FAILED | "未知引擎: X" | unit |
| `DBManager.get_cached_result(engine=...)` | engine 列不存在（旧库未迁移） | `sqlite3.OperationalError` | `init_db()` 启动时已迁移，运行时不会触发 | "服务初始化失败" | integration |
| `init_db()` 迁移 | ALTER TABLE 失败（极少见，磁盘满等） | `sqlite3.OperationalError` | 启动失败，记录错误，进程退出 | 服务不可用 | manual |
| Qwen3 spike 脚本 | 模型加载失败 | 任意 Exception | 写 report.md 标注失败 | N/A（开发期） | manual |

**关键不变量**：PR1 不引入新的异常类。`EngineInitError` / `EngineTimeoutError` 等留给 PR2。

---

## 8. PR2 触发条件 & 内容(不在 PR1 范围) — 已实施复盘

### 8.1 触发条件 (实际触发情况, 2026-05-15)
- ✅ PR1 已合并并稳定运行 (`40d1a9a` on main)
- ✅ Qwen3 spike report 显示可行 (`spikes/qwen3_diarize/spike_report.md`, RTF 0.108-0.118)
- ✅ Qwen3 在 PR1 框架下能跑通端到端 (单任务 + N 并发 + powermetrics 验证)
- ✅ 长期共存确定(FunASR 高准确率 vs Qwen3 自适应聚类, 各有适用场景)
- ⏸ 第三个引擎需求 — 暂无, 但 PR2 仍触发(用户实际选了直接落 Qwen3 集成)

### 8.2 PR2 实际实施 (与 v2 计划的对照)
原计划 PR2 内容(v2):
1. ❌ ABC 抽象 + EngineCapabilities — **跳过**, 注册表式 `_ENGINE_REGISTRY` dict 已足够
2. ❌ EngineRouter 类替代 dispatch 函数 — **跳过**, dispatch 函数加 strict validate 即可
3. ❌ contract test 体系 — **跳过**, 各引擎用独立 unit test 覆盖更直接
4. ❌ 异常分类重构(EngineInitError 等) — **跳过**, ValueError 已足够
5. ❌ engine-level 资源配额 + WebSocketHandler 拆分 — **跳过**, 全局唯一引擎下无并发引擎切换, 无意义
6. ✅ config.engines 嵌套结构 — **替代为 `config.qwen3` 顶级块**, 与 funasr 平级, 简洁

实际 PR2 落地:
1. ✅ **全局唯一引擎模式** — 服务器启动时由 `default_engine` 锁定, upload_request.engine strict reject
2. ✅ **Qwen3-Diarize 引擎** — vendor + 三模块(ASR/Diarize/Merge) + `Qwen3DiarizeTranscriber`(参数注入式)
3. ✅ **模型权重落地脚本** — `scripts/download_qwen3_models.sh` 双模式(prod 镜像 / URL)
4. ✅ **配置项** — `Qwen3Config` 类 + FUNASR_QWEN3_* env override + `config.json` qwen3 顶级块
5. ✅ **测试覆盖** — 11 unit (qwen3_transcriber) + 10 unit (dispatch strict) + 2 e2e integration
6. ✅ **文档** — README / 部署 / 重构计划同步更新

### 8.3 PR2 实际工作量
- 13 commits, 单人(CC)集中工作约 1 个工作日完成
- 关键加速: **不引入抽象**, 直接落地 + TDD 红/绿/commit 闭环
- 验证总计: 58 unit + 5 integration test 全绿

### 8.4 v2 计划的"过度设计"教训
PR2 实际落地证明 v2 计划的"完整 ABC 抽象 + contract test 体系"对 2 个引擎场景是过度设计.
**注册表式 dict + 鸭子类型 + strict validate** 三件套已经完整覆盖了"全局唯一引擎"的所有边界, 且代码总行数不到 60 行.
未来如真出现第三个引擎需求, 届时再触发抽象重构, 而不是现在.

---

## 9. 工作量

| PR | 内容 | CC 主导 | 人工编写 |
|----|------|---------|----------|
| **PR1** | engine 字段流转 + cache key + dispatch + spike + 测试基础 + parity | **~1.5 天** | **~5-7 天** |
| PR2 | （条件触发）完整抽象 + 异常体系 + 资源隔离 + WebSocket 拆分 | ~2-3 天 | ~2 周 |

---

## 10. 失败模式登记表（PR1）

| 路径 | 失败模式 | 已救援？ | 已测试？ | 用户感知 | 日志 |
|------|----------|----------|----------|----------|------|
| resolve_transcriber | 未知 engine 名 | Y (raise → TaskManager FAILED) | Y (unit) | 明确错误 | Y |
| init_db 迁移 | ALTER 失败 | N（启动失败退出） | N (manual only) | 服务不可用 | Y |
| FunASR 路径（不动） | 既有所有失败模式 | 同现状 | 同现状 | 同现状 | 同现状 |
| Cache 查询（加 engine 后） | 索引未迁移 | Y（init_db 阻塞） | Y (integration) | N/A | Y |
| Qwen3 spike | 任意失败 | N/A（开发期） | N | N/A | Y (spike report) |
| Engine 切换时旧引擎未 shutdown | 资源泄漏 | 部分（lazy load 已减轻） | N ⚠️ | 内存压力 | 部分 |

⚠️ 最后一项是 T10 的 PR1 残留风险。约束声明（第 5.8 节）要求同一时刻只有一个引擎驻留，但缺强制检查。PR1 接受此风险。

---

## 11. NOT in scope（v2 更新）

| 项目 | 理由 |
|------|------|
| **ASREngine ABC 抽象 + factory** | 推迟到 PR2，避免在 Qwen3 未验证前过早抽象 |
| **EngineRouter 类** | 用薄 `resolve_transcriber()` 函数替代 |
| **Engine contract test 体系** | 推迟到 PR2 |
| **异常分类重构（EngineInitError 等）** | 推迟到 PR2，PR1 保持现有错误处理模式 |
| **Engine-level 资源配额** | PR1 仅约束声明，强制落地推迟到 PR2 |
| **WebSocketHandler 拆分** | 移除，进 TODOS |
| **`engines: dict` 嵌套配置** | PR1 用 `default_engine` 字符串 + 独立 `qwen3` 块，等 PR2 统一 |
| 重写 task_manager / database / file_based_process_pool 主体 | 不在范围 |
| FastAPI / Web 前端 / 多用户 / 流式 ASR | 不在范围 |
| cam++ 剥离为独立 Stage | 不在范围 |

---

## 12. TODOS（建议入 `TODOS.md`）

1. **[P2/M] WebSocketHandler 670 行拆分**
   - 单文件糅合 auth/upload/chunked upload/dispatch；建议拆 3 个职责子模块
   - 不入 PR1：扩大回归面无必要，等 PR2 一起做或独立 PR

2. **[P2/M] 异常分类纪律建立**
   - 当前 `except Exception` 通配 + 字符串匹配重试；TaskManager 字符串匹配 retryable 错误
   - 替换为明确异常类（EngineInitError / EngineTimeoutError / EngineExecutionError）
   - 入 PR2 或独立 PR

3. **[P2/M] Engine-level 资源配额**
   - 多引擎共存时内存/MPS/CPU 谁分配
   - 入 PR2

4. **[P2/M] `file_based_process_pool.py` 替换为 multiprocessing.Pool / joblib**
   - 585 行自建 IPC + 大量 `.task/.ready` 文件舞蹈是 bug 温床
   - 何时合适：本次重构稳定后下一轮

5. **[P3/M] cam++ 剥离为独立 SpeakerDiarizer Stage**
   - 何时合适：「自己实现说话人」方案选型完成后

6. **[P3/S] 把 `tests/manual/` 中有价值的诊断脚本逐步转为 pytest regression test**
   - 特别是 `tests/diagnostics/test_mps_*.py`

7. **[P3/S] `concurrency_mode` 死代码删除**
   - `lock` / `thread_pool` 已实质废弃，可下次重构时清理

---

## 13. PR1 完成定义（Definition of Done）

- [ ] `pytest tests/unit tests/integration -v` 全绿
- [ ] `tests/integration/test_parity_funasr_semantic.py` 在 3 条 golden 音频上 100% 通过
- [ ] `config.json` 默认 `default_engine: "funasr"`，现网行为零变化
- [ ] 数据库迁移在本地 + 生产 SQLite 副本上验证通过
- [ ] `spikes/qwen3_spike_report.md` 写完，含可行性结论 + 资源占用 + 集成估算
- [ ] 旧 19 个测试脚本搬迁完毕，`tests/manual/README.md` 写明现状
- [ ] WebSocket 协议向后兼容（不带 `engine` 字段的旧 client 行为零变化）

---

## 14. 待 user 确认事项汇总（PR1 完成时回顾）

- [x] **D3** ✅ 已确认：C 方案（分两 PR）
- [x] **D4** ✅ 已确认：用户提供播客音频（60s 双人）+ macOS say 合成单人 + ffmpeg 合成静音
- [x] **D5** ✅ 已确认：Qwen3 spike 脚本同 PR1 落档，实跑由用户主动触发
- [x] **Q1-Q7 实施细节**：在 PR1 落地过程中按需决策，全部体现在 commit 历史

---

## 附：本次 review 历史

- **v1**（已废弃）：完整 ASREngine 抽象 + factory + 三层 contract test 体系
- **codex 独立 review**：10 条实质问题，核心批评「Qwen3 未验证前建抽象是过早优化」
- **v2**（当前）：拆 PR1（最小闭环 + spike）+ PR2（条件触发，看 PR1 结果决定）
