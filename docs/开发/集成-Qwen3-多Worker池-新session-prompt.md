# 新 session 启动 prompt — Qwen3 引擎切到 multi-process worker pool

> **生成于**: 2026-05-15
> **当前分支**: `spike/qwen3-diarize-poc`(Qwen3 集成 PR 已落地)
> **目标**: 把 Qwen3 引擎从「单 instance + asyncio 协程并发(unsafe)」升级到「multi-process worker pool」, 与 FunASR 同一套架构, **且测试必须真正覆盖并发场景**

新 session 直接复制下面内容到对话开头使用.

---

## Prompt 正文(复制以下内容)

我需要把当前 Qwen3-Diarize 引擎的并发模型从「同进程单 instance」升级到「multi-process worker pool」, 跟 FunASR 现有部署形态对齐. 上一个 session 留下了**真实并发风险但没测出来**的隐患, 这次必须用 TDD + 真并发测试覆盖.

### 背景与动机(必须先理解, 不要绕过)

**当前 Qwen3 的并发是 unsafe 的**:
- `Qwen3DiarizeTranscriber` 是进程级单例, 持有 1 个 libllama context (`_asr_engine`)
- `task_manager` 走 `asyncio.Queue + ThreadPoolExecutor`, 启 `max_concurrent_tasks=2` 个协程 worker, 共享同一个 transcriber 单例
- 两个任务同时调进同一个 libllama context → KV cache / Metal compute buffer 串台 → 崩溃 / 输出错乱
- **libllama.cpp 单 context 设计上就不支持并发**, 这不是 bug 是约束

**上一个 session 的测试盲区**(必须吸取):
- 单元测试用 mock 引擎, 不触发 libllama → 测不到
- e2e integration 是单任务串行 → 测不到并发
- smoke wiring 测试也用 mock → 测不到
- 真 server 冒烟只发了 1 个请求 → 测不到
- **没有任何测试构造过"2 个真请求同时调 Qwen3 真引擎"的场景**, 风险被遮蔽

**FunASR 现状是已验证的正确架构**:
- PM2 fork 1 个主进程 (`funasr-server`)
- 主进程内: `WebSocketServer` + `TaskManager` + `FileBasedProcessPool`
- pool 启动 N=2 个 subprocess workers (`venv/bin/python src/core/worker_process.py`)
- 每个 worker subprocess **独立**加载一份 FunASR 模型, 各自一个进程级 PyTorch + MPS context
- 任务通过**文件队列**分配给 worker, 结果写回文件, 主进程读取后通过 WebSocket 回客户端

**PoC v5 已验证 Qwen3 multi-process 是正确路线**:
- N=3 时 GPU 97% 满载, Throughput 2.12 task/min (5min 双人音频)
- GGUF mmap 共享, 总 RSS ~5-7GB (不是 N × 3.8GB)
- N≥4 GPU 100% 饱和, 纯排队不会更快
- 见 `spikes/qwen3_diarize/spike_report.md` 第 5 节 + `spikes/qwen3_diarize/benchmark/run_concurrency_test.sh`

### 你需要先读的(按重要度)

1. **`src/core/file_based_process_pool.py`** — FunASR 现有 pool 架构(直接复用对象, 不要另起炉灶)
2. **`src/core/worker_process.py`** — FunASR worker entry(要扩展为支持 engine 参数)
3. **`src/core/qwen3_transcriber.py`** — Qwen3 当前单 instance 实现
4. **`src/core/qwen3/{asr,diarize,merge}.py`** — Qwen3 三模块(worker 内部要调它们)
5. **`spikes/qwen3_diarize/spike_report.md`** 第 5 节 + **`benchmark/run_concurrency_test.sh`** — N=3 并发数据 + GPU 物理上限
6. **`src/core/task_manager.py`** — 看 `_process_task` 当前怎么调 `resolve_transcriber` + pool
7. **`src/core/config.py`** — `Qwen3Config` + `TranscriptionConfig`
8. **`CLAUDE.md`** — 测试 + 部署约定(macOS only, prod 在 ~/Production/)
9. **`docs/开发/重构计划-ASR引擎抽象.md`** — 架构演进背景

### 已知事实(避免重复探索)

- **PM2 fork mode 单实例**:`ecosystem.config.cjs` 是 fork (不是 cluster), pid 1 个, ~410MB(不装模型)
- **`concurrency_mode = "pool"`** 是 prod / dev 默认, FunASR 已在用
- **`max_concurrent_tasks = 2`** 控制 pool 大小(FunASR 是 2, Qwen3 sweet spot 是 3)
- **`FileBasedProcessPool._launch_worker_process`** 用 `subprocess.Popen` 起 worker
- **worker entry**:`cmd = [sys.executable, "src/core/worker_process.py", "--worker-id", N, "--task-dir", ...]`
- **任务分配**:主进程写任务 JSON 到 task_dir 下文件, worker 轮询读取, 结果回写
- **健康检查**:`_start_health_monitor` + `_ensure_workers_alive`, 死进程自动重启
- **模型路径**:`./models/qwen3_diarize/Qwen3-ASR-1.7B/` + `./models/qwen3_diarize/sherpa/`
- **config.qwen3 已就绪**:asr_model_dir / segmentation_model / embedding_model / preset 参数全部能从 config 读
- **vendor 引擎**:`src/core/vendor/qwen_asr_gguf/`, import 即触发 libllama Metal init(一次性, 进程级)

### 已经拍板的决策(共 9 项, 不要再问)

**架构层**:
1. 跟 FunASR 同一套, 复用 `FileBasedProcessPool` 基础设施, **不另起炉灶**
2. **不引入 ABC 抽象**: 注册表式扩展, 跟当前 `_ENGINE_REGISTRY` 思路一致
3. **错误恢复**: 复用 `FileBasedProcessPool` 现有 health monitor + auto restart, 不重写
4. **删除单 instance 路径**: Qwen3 之后只走 pool, 不保留两套模式

**实现细节**:
5. **池大小配置**: 新增 `config.transcription.qwen3_pool_size = 3` 独立字段(FunASR `max_concurrent_tasks=2` 不变).
   - 理由: 两引擎物理最优值真的不同(FunASR=2, Qwen3=3 PoC sweet spot), 独立字段表达清晰; 共用一个字段反而迫使运维切引擎时改 env.
   - env override: `FUNASR_QWEN3_POOL_SIZE`

6. **worker entry 组织**: 新增独立 `src/core/qwen3_worker_process.py`, **不改造现有 `worker_process.py`**.
   - 理由: CLAUDE.md "低耦合高内聚" 原则; FunASR 路径零侵入(`worker_process.py` 不动); FunASR/Qwen3 模型加载逻辑完全不同(modelscope vs vendor+libllama+sherpa), 夹在一起易混
   - `FileBasedProcessPool` 加一个构造参数 `worker_entry_script` (默认 `src/core/worker_process.py`), Qwen3 池传 `src/core/qwen3_worker_process.py`

7. **Qwen3DiarizeTranscriber 类去留**: **保留**, 但只在 worker subprocess 进程内实例化使用.
   - 主进程不再 import `Qwen3DiarizeTranscriber`, 所有实例化都在 worker 进程内 → libllama context 自然 per-worker 隔离
   - Worker 内单线程跑任务, 不需要 asyncio.Lock
   - `get_qwen3_transcriber()` 入口保留, 内部 lazy 加载, worker 主循环调它的 `transcribe()`

**测试层**:
8. **并发测试粒度**: hybrid + 真 e2e 两个都做.
   - hybrid(真 pool + mock worker entry): 默认 enabled, 秒级, 验证主进程派发不串台
   - 真 e2e (`FUNASR_RUN_INTEGRATION=1`): 真 N=2 Qwen3 模型并发, 验证 libllama 真能并行
   - 任选其一都有盲区, 必须两者结合堵死

9. **真服务器手工冒烟必做**: 部署前最后一道防线, 见下方"工作产出要求".

### 工作产出要求

一个 PR, 包含:

**代码改造**:
- `src/core/worker_process.py` 加 `--engine` 支持 qwen3 模型加载分支(或新增 `qwen3_worker_process.py`)
- `src/core/file_based_process_pool.py` 改造: pool 大小按 engine 决定, 命令行透传 engine, 任务分发 JSON 加 engine 字段(无影响 funasr)
- `src/core/transcriber_dispatch.py` Qwen3 分支改成"通过 pool 调度", 不再返回 transcriber 单例直接调
- `src/core/task_manager.py` 适配: 不管 engine 走哪个, 都通过 pool 提交任务
- `src/core/config.py` 加 `qwen3_pool_size`(如选 A)
- `config.json` 默认值同步

**测试新增(必须有)**:
- `tests/unit/test_worker_process_engine.py` — worker entry 接受 engine 参数, 不同 engine 加载不同模型(mock 模型加载)
- `tests/unit/test_file_based_pool_engine_aware.py` — pool 启动按 engine 选 worker 命令行, 透传 engine 字段
- `tests/integration/test_qwen3_pool_concurrent_dispatching.py`(默认 enabled) — 真 pool + mock worker, 构造 2 个并发请求, 验证:
  - 两个请求分到两个不同 worker_id
  - 两个 worker 各自的输出不串台(各自的 task_id / 结果独立)
  - 一个 worker mock 抛错, 另一个不受影响
- `tests/integration/test_qwen3_pool_real_concurrency.py`(`FUNASR_RUN_INTEGRATION=1`) — 真 N=2 并发, 真 Qwen3 模型:
  - 同时上传 2 条音频(同 hash 不同)→ 各自完成 → 文本互不串台
  - 验证总耗时 < 单任务耗时 × 1.5(并行至少有些收益)
  - 验证 worker 进程数 == pool_size
  - 验证 RSS 总量 < 8GB(GGUF mmap 共享 sanity check)

**文档**:
- `docs/部署.md`: Qwen3 模式下 pool 大小, 内存预算说明
- `README.md`: 引擎对照表更新, 标注 Qwen3 是 multi-process pool
- `docs/开发/重构计划-ASR引擎抽象.md`: 在 PR2 复盘节点补充"Qwen3 pool 化"作为 PR3 完成项
- `.env.example`: 加 `FUNASR_QWEN3_POOL_SIZE` 注释段(如选 A)

**回归**:
- FunASR parity 测试零回归(`FUNASR_RUN_INTEGRATION=1 pytest tests/integration/test_parity_funasr_semantic.py`)
- 现有 Qwen3 e2e 仍能跑通(改成走 pool 之后, 仍能完成单任务)
- smoke wiring 全绿(`tests/integration/test_smoke_engine_wiring.py`)

**真服务器手工冒烟**(部署前最后一道防线):
- 起 dev server `FUNASR_DEFAULT_ENGINE=qwen3 FUNASR_QWEN3_POOL_SIZE=2`(2 是为了减少模型加载时间, 测并发足够)
- 用 asyncio + websockets 写一段小 client, **同时**发 2 个 upload_request(不同 file_hash)
- 验证: 两个 task 各自走完整流程, 文本互不串台, server 不崩
- 关 server, 报数据(总耗时 / 两 task RTF / RSS 峰值)

### 工作方式约束(非常重要, 不可省略)

**1. 严格 TDD — 红/绿/commit 三步, 不积累改动**

每加一个功能都按以下顺序:

- **红**: 先写测试 → 跑 → 确认 fail
- **绿**: 写最小实现让测试过 → 跑 → 确认 pass
- **commit**: 用 git commit 沉淀这个"红 → 绿"闭环, 一个 commit 一件事
- 然后下一个 cycle, 不要积累多个改动一次性提交

**对应到本任务的红/绿/commit 闭环**(推荐顺序):
1. worker entry 加 --engine 参数(unit 测试 mock 加载) — 红/绿/commit
2. worker 内部 qwen3 模型加载 + transcribe 路径(unit 测试 mock vendor) — 红/绿/commit
3. FileBasedProcessPool engine-aware 派发(unit 测试 subprocess.Popen mock) — 红/绿/commit
4. 主进程 dispatch 改走 pool(unit 测试) — 红/绿/commit
5. **hybrid 并发测试: 真 pool + mock worker, 2 并发不串台** — 红/绿/commit
6. **真 e2e 并发: N=2 真模型, 2 并发不串台** — 红/绿/commit
7. 文档 + 真 server 手工冒烟 — commit

**2. 测试覆盖约束 — 这次必须有真并发场景**

上次失败教训: 全套测试都是单任务, 没有任何一个用例并发调引擎. 这次约束:
- ❌ **不允许只跑单任务测试**, 必须至少 1 个测试构造 N=2 并发
- ❌ **不允许全 mock**, 必须至少 1 个 integration 测试用真 Qwen3 模型跑 N=2 并发
- ✅ **测试做不出来不算完成**: 如果发现 N=2 并发不稳定, 必须先修代码, 不能 skip 测试

**3. 自主推进, 不打断用户**

- 仅在 4 个明确列出的决策点用 AskUserQuestion 一次性问完
- 遇到任何技术问题(库选择 / 命名 / 错误处理 / log 格式 / 等等)**自己决定推进**
- 真正需要用户介入的只有: 凭证 / 用户私人数据 / 部署环境特定细节
- 任务完成前不要主动停下等"用户确认"
- **任务定义为完成**: PR 内所有代码 + 所有测试绿 + 真服务器并发冒烟通过 + 文档更新, 这时候才给最终汇报

**4. 其他**

- macOS only(CLAUDE.md 约定)
- 不修改 `spikes/` 下的代码(历史记录)
- 复用 `FileBasedProcessPool` 现有 health monitor / auto-restart, 不重写
- 保留 FunASR 路径零侵入(只加分支, 不动既有逻辑)

### 开始

1. Read 上面列的 9 个文件熟悉上下文
2. **不需要问用户任何决策点** — 9 项决策都已经在上面"已经拍板的决策"段写死
3. **全程不向用户提问**(除非凭证 / 用户私人数据 / 部署环境特定细节), 按 TDD 红/绿/commit 循环把活干完
4. 真服务器并发冒烟通过后, 给用户最终汇报(含: commits 列表 / 测试矩阵 / 并发实测数据 / 内存峰值 / 后续 TODO)

### 已知 TODO(用户已经拒收, 不要做)

- **模型能力评测**(WER / DER / 标注集): 独立工程, 不在本 PR 范围, 提一句留 TODO 即可
- ASR 引擎抽象重构(ABC + EngineRouter): 之前讨论过, 不做; pool 化够了
- WebSocket protocol 重构: 不在本范围
