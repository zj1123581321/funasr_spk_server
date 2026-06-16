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

### 14. 词级时间戳 Phase 2 — 替换式 merge（词重算段边界）
- 来源：`/plan-eng-review`（2026-06-09）+ codex 跨模型挑战
- **What**：用词级时间戳重算 speaker 段边界，替换 `merge.py:216` 的字符比例近似（`_split_text_by_weights`）
- **Why**：若下游需要段边界达到词级精度（现状 char-ratio + silence_align 已到 ~200-260ms）
- **触发条件**：eval 证明现状 ~200ms 段边界不够用时才做
- **当前状态**：Phase 1 走增量版（保留现有段，只把 MMS 词挂进 `segment.words`），段边界不变。Phase 2 才动 merge
- **背景**：CEO review 判定段边界替换是"高风险边际改善"，词级真正价值在 `segment.words`（token 时间戳）。详见 `spikes/qwen3_word_timestamp/SUMMARY.md` + `docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md`
- 优先级：P3（条件触发）

### 15. num_speakers per-request 化
- 来源：`/plan-ceo-review`（2026-06-10，diarize 开关设计评审 D7 决策）
- **What**：把 `num_speakers` 加进 TranscribeOptions，per-request 可传
- **Why**：会议等已知人数场景可提聚类精度；TranscribeOptions 口袋落地后穿透成本极低
- **Cons / 为何延期**：缓存 key 要再折一维 + E1 投影兼容规则要重定（不同 spk 数的 diarized 行不能互投）+ 测试矩阵翻倍，当前无客户端需求方
- **触发条件**：有真实客户端提出指定人数需求
- 优先级：P3（条件触发）
- 依赖：diarize 开关 PR（TranscribeOptions 结构）落地

### 16. FunASR server 级「不加载 cam++」部署形态
- 来源：`/plan-ceo-review`（2026-06-10，D4 决策附带）
- **What**：config 开关控制 AutoModel 初始化时不传 `spk_model`，纯文字稿部署形态
- **Why**：funasr 的 cam++ embedding 提取只 gate 在模型加载（`auto_model.py:488`），per-request 关不掉；纯转录场景这是白烧的算力
- **Cons**：要定义「不加载时收到 diarize=true 请求」的冲突规则（拒绝或降级）
- **触发条件**：出现纯文字稿的 funasr 部署需求
- 优先级：P3（条件触发）

### 17. VRAM preflight 显存探针
- 来源：`/plan-eng-review` + `codex` 外部声音（2026-06-16，word_align 显存落地评审）
- **What**：跑 CUDA word_align 前量显卡剩余显存，不够则不加载 CUDA aligner、直接走 CPU；记录每次请求前后 VRAM delta
- **Why**：word_align 落地 PR 靠「catch OOM 后转 CPU」事后补救，preflight 是事前预防
- **🔒 评审定案（2026-06-16 /plan-eng-review + codex）**：见 `docs/开发/gpu加速/2026-06-16-word-align显存安全-评审定案与落地计划.md`。锁定：探针 `free_vram_mib()` **只用 nvidia-smi，不引 NVML**（冷路径不值得新依赖，**偏离本条原写的「NVML 优先」**）；**尊重 `CUDA_VISIBLE_DEVICES`** 不读死第一行（codex #12）；阈值进 `Qwen3Config` config 字段（env 可覆盖）+ 保守默认 + 3060 标定；探针放独立模块 `src/core/gpu_mem.py` 给 Lane 1 + Lane 2 共用。**preflight 不替代 OOM fallback**（TOCTOU，codex #11）。
- **Cons / 为何延期**：阈值要二次标定（保守起点：未加载要求 free ≥4.5GB，已加载要求 ≥1.5-2GB，非最终标准）
- **⚠️ codex 推迟风险提醒**：pool_size=1 的序列化只挡内部并发 CUDA 会话，挡不住同卡 CapsWriter 占用波动 / ORT BFCArena 残留 / 显存碎片 / 用户调高 pool_size。第一个请求仍是「试错探针」，OOM 可能在 fallback 前就把 CUDA session 弄坏
- **触发条件**：`qwen3_pool_size` 调到 >1，或同卡显存竞争变频繁，或观测到 OOM thrash
- **依赖**：word_align per-request + CPU fallback + poison PR（2026-06-16）落地
- **课题关系**：#17 与 #18 同属「word_align 显存安全」课题，**一次设计、分两次发**——本条是 **Lane 1**（小、低风险、先发），探针 `free_vram_mib()` 是 #18 sidecar 路由器复用的**共享原语**（非一次性前置）。设计/落地交接见 `docs/开发/gpu加速/2026-06-16-TODOS17-18-word-align-显存安全-交接.md`
- 优先级：P2（Lane 1 先行）

### 18. 独立 word_align sidecar 进程
- 来源：`/plan-eng-review` + `codex` 外部声音（2026-06-16，word_align 显存落地评审）
- **What**：把 word_align 从主 ASR 服务拆成独立进程，按需启动 CUDA session、空闲 TTL 自动退出释放 VRAM；主服务调用，不可用时 CPU fallback 或返回无词；加生命周期日志 + 健康检查
- **Why**：落地 PR 的 poison + dispose 只能「尝试」要回显存，ORT BFCArena 高水位未必真还（2026-06-16 3060 实测确认：同尺寸任务显存阶梯爬升封顶 ~7844MiB，只有重启进程才回落）。**唯一可靠的显存释放是进程退出**——sidecar 让偶发的词级时间戳请求不再污染主 ASR 服务的长期显存基线
- **🔒 评审定案（2026-06-16 /plan-eng-review + codex 15 findings）**：见 `docs/开发/gpu加速/2026-06-16-word-align显存安全-评审定案与落地计划.md`。锁定：**长驻进程 + idle TTL 自杀 + Unix domain socket（stdlib，零新依赖）request/response**，传 audio_path + chunks JSON；**不复用 `FileBasedProcessPool` 类**（每任务即退 + auto-respawn + 启动前清文件 + 固定目录语义与长驻 idle-TTL 拧着，codex #1/#2/#3/#5/#10/#15）；sidecar **只做 CUDA**（CPU 兜底留主进程）；**进程全局单例**（pool>1 也单 session，codex #6）；超时**先杀 sidecar 再 CPU**杜绝双跑（codex #4）；OOM **杀/退休 sidecar 不永久 poison 主进程**（codex #8）；**瘦入口**只 import `qwen3/word_align.py`（codex #9）；Lane 2 后主进程 **硬切 CUDA aligner 路径**测试钉死（codex #7）；缓存 +wa 看 **has_words** 非 sidecar 成功（codex #14）；**仅 cuda runtime 启用**。
- **Cons / 为何延期**：跨进程协议 + 生命周期管理 + 健康检查，工程量最大；主服务多一个故障面
- **触发条件**：Lane 1（#17）上线后用真机数据复评——若 preflight + CPU 降级后 OOM 基本绝迹可继续延后；若仍常撞顶 / word_align 高频常驻则做
- **依赖**：word_align per-request PR（2026-06-16）+ **同课题 Lane 1（#17）的探针落地后复用**
- **课题关系**：同「word_align 显存安全」课题的 **Lane 2**（大、结构性、后发），复用 #17 探针。交接文档同上
- 优先级：P3（Lane 2，#17 之后）

### 19. word_align preflight duration/chunk-aware 动态阈值
- 来源：`codex` 外部声音 #13（2026-06-16，#17/#18 显存安全评审）
- **What**：preflight 阈值从「平的固定值」升级为「按本次请求音频时长/chunk 数动态算所需显存余量」再放行 CUDA word_align
- **Why**：MMS CTC-FA session 显存随输入 shape 往上爬，同样「已加载」跑 60s 与 60min 峰值显存差很多。平阈值（已加载 free ≥1.5-2GB）挡得住短音频，挡不住长音频——长音频仍可能运行时 OOM（靠 fallback 接住）
- **Cons / 为何延期**：要先标定「时长→显存峰值」曲线，复杂度不小；平阈值 + OOM fallback 已能兜底
- **触发条件**：#17 上线后 3060 真机数据证明长音频确实常在 preflight 放行后 OOM
- **依赖**：#17（VRAM preflight 探针 + 阈值 config 字段）落地
- 优先级：P3（#17 之后，数据驱动）

### 20. 批量上传路径改异步轮询契约（根治 300s 超时）— ✅ 端到端完成（2026-06-16）
- **✅ 客户端切换完成（2026-06-16）**：两个客户端（独立代码库）已从同步死等切到 `task_status_batch` 轮询，300s 超时端到端闭环。T5 完成。
- **✅ 服务端落地（2026-06-16，已 push main）**：T1-T4 完成，746 unit + 3 FunASR parity 全绿。
  - `task_status_batch` 批量查询：`schemas.py:TaskStatusBatchResponse/TaskStatusBatchItem` + `websocket_handler.py:_handle_task_status_batch` / `_build_task_status_batch_item`（同步组装不 await 钉 COMPLETED 原子性；JSON 走 result / SRT 走 srt_content；终态全集带 error；上限 50 截断+warn；poll-miss `status=null`+`task_expired`/`task_not_found`）。
  - 上传协议**零改动**（无 wait 开关 / 无 task_accepted），入队照旧回 `task_queued`/`upload_complete`(带 position)。
  - 文档同步：`Server-Client 交互协议.md`（服务端规格）+ `docs/使用/客户端交互指南.md`（客户端迁移节）+ `CLAUDE.md`。
- **✅ T5 客户端×2 已切（2026-06-16）**：停 300s 堵等 → 收 ack 后批量 `task_status_batch` 轮询；终态全集停轮；poll-miss 凭 file_hash 重投。
- 来源：`/plan-eng-review`（2026-06-16，高负载队列机制审查；本轮选 C「先止血，异步契约下一步」）
- **What**：把批量/分片上传的「finalize 即同步等 `task_complete`」改成「finalize 立刻返回 `task_id` → 客户端轮询 `task_status` 拿进度/结果」。服务端 `task_status` 查询接口**已存在**（`websocket_handler.py:131/316` → `_send_task_status` → `task_manager.get_task`），主要是客户端契约 + 部署协调
- **Why**：`接收消息超时(300秒)` 的**真正根因**是客户端单 ws 同步死等。qwen3 pool=1，真实并发≈1，队列深处任务等待 ≈ 位置×单任务时长 >> 300s，**必然超时**。本轮止血（内存/资源泄漏修复 + 队列满可重试信号）只让「队列已满」变优雅、让内存不爆，**修不掉这个超时**——客户端只要还站着死等，深队列就还超时。轮询在本项目长音频路径**已被证明可行**（CUDA box 上 ws 长音频断连后改的就是轮询），不是空想
- **现状/起点**：①服务端轮询接口现成可用；②本轮加的 self.tasks TTL（终态任务保留 ~1h）正是为轮询窗口设计的——异步化后客户端在 TTL 窗口内轮询命中内存对象，超窗后靠 DB 缓存（`db_manager.save_result` 已双写）按 file_hash 命中兜底；③队列满的 `queue_full` 可重试信号天然适配「提交被拒就退避重投」
- **耦合点（务必同步看）**：异步轮询的 TTL 必须 ≥ 客户端最长轮询窗口，否则慢客户端轮询「任务不存在」。本轮 TTL 默认值若调小，需复核此约束
- **Cons / 为何延期**：要改客户端 + 协调部署顺序（**先服务端、后客户端**，与 diarize 上线同款假设，需单测钉死老客户端同步等待路径仍兼容）；本轮止血与契约改动解耦后可分次安全上线
- **触发条件**：止血 PR 上线后，若高负载批量场景仍频繁 300s 超时（预期会）
- **依赖**：本轮止血 PR（self.tasks TTL + queue_full 信号 + session TTL）落地
- 优先级：P1（300s 超时是用户可见的核心症状，止血后应尽快接手）
- **🔒 评审定案（2026-06-16 /plan-ceo-review + 对抗 spec 子代理 + codex 外部声音）**：见 `docs/开发/2026-06-16-异步轮询契约-设计定案与落地计划.md`。锁定：**无 `wait` 开关、无 `task_accepted` 新消息**（codex 战略简化——服务端入队后今天就回 `task_queued{position}`，客户端在此 ack 后转轮询即可，上传协议零改动 ⇒ 无 flag day）；服务端真新代码只有 `task_status_batch`（**内联 result**，含终态 failed/timed_out/cancelled + SRT 走 srt_content，上限 50/批，无 live position）+ **正经 in-flight 去重 index**（非扫 self.tasks，只含已入队任务避占位死等，锁内原子查-插 + 连接迁移 + ack 回 canonical task_id，key 含 output_format）；跳过确定性 task_id；poll-miss 凭 file_hash 重投（重启分片需重传）。

### 22. in-flight file_hash 去重（独立 PR，从异步轮询契约拆出）
- 来源：`/plan-eng-review`（2026-06-16，异步轮询契约评审，问题1 定案拆出）
- **What**：`submit_task` 入队前查同 `(file_hash, 折维tag, output_format)` 的在途任务，命中则折叠到已有 task_id 不二次入队。
- **Why**：双客户端撞同一文件时，pool=1 上白跑一个槽（第二次命中第一次缓存或幂等覆写，无正确性 bug，仅浪费算力）。
- **落地（避坑版，codex #2-6）**：**自愈 index**（`(file_hash,折维tag,output_format)→task_id`，命中后校验 `self.tasks[tid]` 仍在且非终态，脏条目当 miss + 删——杀掉「6+ 终态点漏删」整类 bug）；锁内原子查-插（防 race）；**只含已入队任务**（不含 upload_request 阶段未上传占位，避占位死等）；命中折叠要**迁移连接映射**到 canonical + ack 回 canonical task_id（client 改轮询它）。
- **Cons / 为何独立**：真实耦合跨 task_manager/websocket_handler 两层 + 6+ 终态落地点，blast radius 比「超时根治」大；去重 bug（轮询死 task_id）比它解决的「白跑一个槽」更糟，不与超时根治绑发。
- **依赖**：本轮薄轮询 + batch 落地 → ✅ 已满足（#20 服务端 2026-06-16 落地），可随时接手。
- 优先级：P2

### 21. per-连接/客户端并发限流（防单客户端霸占 pool=1 队列）
- 来源：`/plan-ceo-review`（2026-06-16，异步轮询契约评审，SELECTIVE EXPANSION 推迟项）
- **What**：给 ws 连接/客户端加在队并发上限，防一个客户端甩满队列饿死另一个。
- **Why**：真实并发=pool=1，队列是准入控制。现 `queue_full` 已提供背压（被拒退避重投），但无 per-连接配额——理论上单客户端可长期占满队列。
- **Cons / 为何延期**：当前仅 2 个可信自控客户端，恶意/失控占用非现实威胁；加准入配额 + 公平调度有复杂度。
- **触发条件**：接入不可信客户端 / 客户端数增多 / 实测出现互饿。
- 优先级：P3

### 23. 冷启动 /health 可观测性（启动序改造）
- 来源：`codex` 外部声音 #5/#6（2026-06-16，可观测性仪表盘评审）
- **What**：让 /health 在服务冷启动期（模型加载中/加载卡死/失败）也能应答，报 `starting`/`loading`/`degraded`。
- **Why**：模型是 eager 加载（`src/main.py:47-49` `transcriber.initialize()` 在 `:56 websockets.serve()` 之前）。socket 在模型加载完之后才绑，所以 process_request 健康端点观测不到启动卡死/失败——最危险的冷启动期 /health 连答都答不了。
- **怎么做**：在 `transcriber.initialize()` 之前先绑一个极简早期 HTTP health listener（独立于后面 late-bind 的 `websockets.serve`），启动完成后交接/关闭。
- **Cons / 为何延期**：要动 `main.py` 启动序 + 多一个早期独立 listener，部分偏离"同端口 process_request"设计（可观测性仪表盘 D3）；冷启动卡死罕见，日志 + PM2 autorestart 已兜底进程级失败。
- **触发条件**：观测到冷启动卡死/失败频繁，或需要外部探针在启动期就拿到状态。
- **依赖**：可观测性仪表盘 PR（/health liveness）落地。
- 优先级：P3

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
