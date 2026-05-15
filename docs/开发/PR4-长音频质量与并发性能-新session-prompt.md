# 新 session 启动 prompt — Qwen3 长音频质量 & 并发性能问题

> **生成于**: 2026-05-15
> **当前分支**: `spike/qwen3-diarize-poc` (PR3 已落地, 含心跳 + audio format 转换修复)
> **目标**: 解决 PR3 真实长音频测试暴露的 3 个残留问题 — 长音频质量退化 / 资源挤压 / RTF 性能下降 2x

新 session 直接复制下面内容到对话开头使用.

---

## Prompt 正文(复制以下内容)

我需要解决 PR3 上线 Qwen3 多 Worker 池后, 真实生产音频测试暴露的 3 类残留问题. 上一个 session (PR3) 已经修复 m4a 格式支持 + 长任务 client recv timeout, 但跑了真实 83min m4a + 149min mp3 并发后, 还有 3 个本质性问题没解.

### 背景: PR3 已落地的 11 个 commits

PR3 完成情况, 不要重做:

```
4be6a6c feat(qwen3): Qwen3PoolTranscriber 加 progress 心跳, 修复长音频 client recv timeout
cd578a8 fix(qwen3): worker 加 ffmpeg audio format 转换 + 长音频并发冒烟脚本
0acb2c7 fix(main): server 启动按 default_engine 选 transcriber + 加 Qwen3 并发冒烟脚本
b770d60 docs(qwen3): PR3 文档同步 — 部署 / README / 重构计划
e493b6b test(qwen3): 真 e2e N=2 并发 + 加固 pool task_dir 隔离 + worker file_name 改写
afedd99 test(qwen3): hybrid 并发派发测试 — 真 pool + mock worker, 2 并发不串台
6c12ec2 feat(qwen3): dispatch 切到 Qwen3PoolTranscriber, 主进程不再持有 libllama context
87d304f feat(qwen3): config 加 qwen3_pool_size 字段 + FUNASR_QWEN3_POOL_SIZE env override
5713d43 feat(qwen3): 新增 Qwen3PoolTranscriber 主进程 wrapper
e6e5971 feat(qwen3): 新增 qwen3_worker_process.py worker entry
169f899 feat(pool): FileBasedProcessPool 加 worker_entry_script 构造参数
```

PR3 已经做了:
- Qwen3 引擎走 multi-process worker pool (N=2 / N=3), libllama context per-worker 隔离
- 主进程 dispatch 切到 pool wrapper, 不再持有 libllama
- task_dir 物理隔离 (FunASR `temp/tasks/` vs Qwen3 `temp/tasks_qwen3/`)
- worker 内 ffmpeg 转 wav (m4a / mp3 等容器支持)
- progress 心跳 30s (防 client recv timeout)
- 95 unit + 4 hybrid + 4 real e2e 测试全绿

### 真实测试暴露的 3 类残留问题(按 P0 → P3 优先级)

#### 🔴 P0-1: 长音频(>80 min)LLM 退化 / 重复 token 循环

**实测**: 149min mp3 转录, 服务端 32.8 min wall, 输出 411 segs / 2 speakers (识别正确), **但中后段质量崩溃**:

| 时间段 | 内容 | 状态 |
|---|---|---|
| 0–80 min | 真实文本 "欢迎收听晚点聊..." | ✅ 质量好 |
| 80 min (5179s) | "我们这个团队的这个能力。那这个能力,我们就是去关注这个东西..." 反复 | ❌ 间歇退化 |
| 146 min (8796s) 末段 138s | "我这个AI,我这个AI,我这个AI..." 死循环 | ❌ 完全崩坏 |

**根因**: worker log 出现大量 `[llama.cpp] decode: failed to find a memory slot for batch of size 1` — Qwen3-ASR-1.7B GGUF 的 LLM 在长上下文下 **KV cache 满**, decode 失败, 退化成 token 重复循环.

**解决方向(待 session 内确认)**:
- A. **自动分段**: 主进程 / wrapper 层把 audio > 60 min 用 ffmpeg 切成 30-50 min 片段, 各自跑, 合并结果. Speaker 跨片段对齐需要重新聚类 (用 embedding 模型把片段间 cluster id 映射统一).
- B. **vendor 修复**: `src/core/vendor/qwen_asr_gguf/` 内 KV cache reset / 强制 cap context window. 但 vendor 是 CapsWriter upstream, 改动需要 fork.
- C. **质量预警**: 检测重复 token 循环 (连续 N 段文本相似度 >95%), flag 输出. 不修复但避免静默错误.
- D. **混合**: 短音频(<60 min) 直接跑, 长音频(>60 min) 强制分段 + 跨段对齐.

建议: **A + C 组合**, B 留观察(超出本 PR 范围).

#### 🔴 P0-2: 并发时 CPU/GPU 打满, 挤压其他进程

**实测**: N=2 并发跑长音频时, 整机 CPU/GPU 资源全被吃满, 同机器其他服务(PM2 funasr-server 8767 / capswriter / etc) 被挤压.

**根因(推测)**:
- Qwen3 worker 子进程: libllama Metal (GPU) + sherpa diarize (8 CPU threads, OMP) + onnxruntime (encoder) → 单 worker 已经吃 8+ CPU 核 + 大部分 GPU
- N=2 时 2 × 8 = 16 个 OMP threads + 共享 1 个 Apple M1 Max GPU
- 没有 process priority / CPU affinity / GPU 限流

**解决方向**:
- A. **降低单 worker 线程数**: `config.qwen3.num_threads` 从 8 → 4, 减少 OMP 抢占
- B. **process nice 优先级**: worker subprocess 启动时 `nice -n 10` 让出 CPU 给其他进程
- C. **GPU 排队管理**: vendor 层加 Metal command queue 优先级? (可能超出范围)
- D. **资源监控 + 自动降级**: 检测 system load > X 时降到 N=1

建议: **A + B 优先**(改动小), C/D 留观察.

#### 🟡 P1-1: RTF 0.23 vs PoC 0.118 慢 2x

**对比数据**:

| 测试 | RTF | 说明 |
|---|---|---|
| PoC v3 单 worker (5min 双人) | **0.118** | spike_report.md baseline |
| PoC v5 N=3 (5min 双人) | ~0.142 | benchmark, GPU 97% 满载 |
| PR3 N=2 60s podcast 并发 | 0.282 | task 7 e2e 实测 |
| PR3 N=2 83min m4a 并发 | 0.282 | PR3 长音频测试 |
| PR3 N=2 149min mp3 并发 | 0.220 | PR3 长音频测试 |

**慢了大约 2x**. 用户记忆 PoC 是 0.15, 实际数据 0.118 ~ 0.142, 反正 **PR3 后明显变慢**.

**可能根因(按推测概率排序)**:
1. **Worker 单任务后退出 + 重启策略**: 每次 task 都要重新加载 libllama (~30s) + sherpa (~5s) + onnxruntime encoder (~3s) ≈ **38s overhead per task**. PoC 是常驻 worker, 没这部分.
2. **ffmpeg 转 wav 落盘开销**: 长音频 m4a/mp3 → wav 用 ffmpeg 全量解码 + 落临时文件, 149min mp3 转换 ~8s. PoC 直接读 wav 没这步.
3. **N=2 共享 GPU 排队**: 单 worker 时 GPU 独占, N=2 时 Metal command queue 排队, 单任务实际 GPU 时间 ≈ 50%.
4. **pool 派发 IO overhead**: task 文件读写, audio 复制到 task_dir, pickle 序列化 — 长音频时 audio 复制成本不小(149min mp3 102MB).

**解决方向**:
- A. **worker 常驻**: 移除 "单任务后退出" 策略, worker 持续接 task. 但需要解决 MPS 内存累积问题 (FunASR worker 退出策略就是为了这个).
- B. **直接管道转换**: ffmpeg stdout pipe → sherpa stdin, 不落磁盘. 但 vendor 代码要改.
- C. **audio 软链接**: 不真复制 audio 到 task_dir, 用 symlink. 风险: worker 内修改/删除 source 影响主进程.
- D. **benchmark 单 worker** 排除并发因素, 看 PR3 单 worker 真实 RTF, 然后跟 PoC 对比, 锁定回归位置.

建议: **D 先排查**(写个 bench script 跑单 worker), 再针对性 A/B/C.

#### 🟡 P1-2: Worker 单任务后退出策略可能不适合 Qwen3

继 P1-1 的 worker 重启 overhead. FunASR worker 退出策略是因为 MPS 长跑内存碎片化 / OOM, 但 Qwen3:
- libllama 用 Metal 直接管 GPU 内存, 不走 PyTorch MPS allocator
- 单 worker 跑完 149min RSS 7.8GB, 但**完成后不会自动 release**
- 即便 worker 退出, libllama 也是 OS-level 清理, 重启时 mmap GGUF 又一次

**待确认**: Qwen3 worker 不退出是否会有累积内存问题? 跑几个连续任务看 RSS 趋势.

#### 🟢 P2: 真实进度通知(优化项, 不阻塞)

当前心跳进度值是模拟的 (5 → 95 线性递增), 不是真实 ASR / Diarize 阶段. 用户体验上 "5%, 8%, 11%..." 跟实际无关.

**解决方向**: worker 内每完成一个阶段(ASR 25% / Diarize 50% / Merge 75%)写一个 progress 文件, 主进程读取. 但跨进程 progress 落实需要协议设计.

#### 🟢 P3: task_timeout_minutes 配置误导

server 启动日志 print "任务超时: 30 分钟", 但实际 timeout 由 `pool._calculate_timeout` 算 (300s + duration × 0.3, max 60 min). 这个 30 min 是死代码, 删了或者改成动态.

### 你需要先读的(按重要度)

1. **`docs/开发/集成-Qwen3-多Worker池-新session-prompt.md`** — PR3 完整背景
2. **`src/core/qwen3_pool_transcriber.py`** — 主进程 wrapper + 心跳
3. **`src/core/qwen3_worker_process.py`** — worker entry + ffmpeg 转换 + 单任务退出
4. **`src/core/file_based_process_pool.py`** — pool 派发 / health monitor / `_calculate_timeout`
5. **`src/core/qwen3/{asr,diarize,merge}.py`** — Qwen3 三模块
6. **`src/core/vendor/qwen_asr_gguf/inference/`** — vendor 引擎(P0-1 LLM 退化的根因区域)
7. **`spikes/qwen3_diarize/spike_report.md`** 第 5 节 — PoC RTF baseline
8. **`tests/manual/server/long_audio_concurrent.py`** — 长音频真服务器测试脚本
9. **`data/transcription_cache.db`** — 149min 测试结果在 cache 里(file_hash a7d476bd...)
10. **`logs/workers/worker_*.log`** — worker 跑长音频的日志, 含 `decode: failed to find a memory slot` warning

### 实测数据(诊断根据, 不要重测除非必要)

**149min mp3 N=2 并发**:
- 服务端 wall: 32.8 min (1970s server_time)
- RTF: 0.220
- segments: 411
- speakers: 2 (Speaker1 嘉宾 85.2% / Speaker3 主持人 14.8%) ✅ 识别正确
- 中后段质量: ❌ 80 min 后间歇重复, 146 min 后 138s 完全死循环 "我这个AI..."

**83min m4a N=2 并发**:
- 服务端 wall: 23 min (1396s server_time)
- RTF: 0.282
- segments: 551
- speakers: 8 (Speaker1, 2, 8, 11, 12, 14, 16, 42) — speaker 过分散, 可能聚类阈值需调

**60s podcast 单 worker (历史 PoC v3)**:
- RTF: 0.118
- 这是 baseline, 现在 PR3 慢了 2x

**Worker RSS 峰值**:
- 60s podcast: 3.45 GB
- 83min m4a: 5.0 GB
- 149min mp3: **7.8 GB** (8GB 警戒)

### 工作产出要求

PR4 一个 PR, 包含:

**代码改造(按问题对应)**:
- **P0-1 长音频分段**: 新增 `src/core/qwen3_segment.py` (audio > N min 切片) + `Qwen3PoolTranscriber` 分段调度 + speaker 跨段对齐合并
- **P0-1 质量预警**: `src/core/qwen3/merge.py` 加重复 token 检测, raw_result 加 `quality_warning` 字段
- **P0-2 资源限流**: `qwen3_worker_process.py` 启动时设 `os.nice(10)`, `config.qwen3.num_threads` 默认值 8 → 4
- **P1-1/P1-2 worker 常驻**: 移除 worker 单任务后退出策略, 加 N 任务后或 M 分钟后退出兜底(防内存累积). 先 benchmark 单 worker 真实 RTF 验证回归点
- **P3 配置清理**: 删 `task_timeout_minutes` print 或改成动态展示

**测试新增(必须有)**:
- `tests/unit/test_qwen3_segment.py` — 分段逻辑 unit
- `tests/integration/test_qwen3_long_audio_segmented.py` — 长音频分段 e2e
- 单 worker bench script: `tests/manual/server/qwen3_single_worker_bench.py`
- 真服务器长音频 + 资源占用监控冒烟(扩展 long_audio_concurrent.py 加 CPU/GPU/RSS 采样)

**文档**:
- `docs/部署.md` 加长音频 / 资源占用相关章节
- 复盘 PR3 → PR4 链路

**手工冒烟**(部署前最后一道防线):
- 跑同样 83min m4a + 149min mp3 并发
- 验证 149min 末段不再 "我这个AI" 死循环
- 监控并发期间整机 CPU/GPU/load, 不应挤死其他 PM2 进程
- 单任务 RTF 对比 PoC 0.118 baseline, 接近即可

### 工作方式约束(同 PR3, 不可省略)

**严格 TDD**: 红 → 绿 → commit 三步, 不积累.

**测试覆盖约束**: 
- ❌ 不允许只跑短音频测试, 必须有长音频(>60 min)的 e2e
- ❌ 不允许只跑单 worker, 必须有 N=2 并发资源监控
- ✅ 测试做不出来不算完成

**自主推进**: 决策点用 AskUserQuestion 一次性问完(预计 3-4 个: 分段长度阈值 / speaker 对齐策略 / worker 常驻策略 / 资源限制方式). 技术细节自己定.

### 已经拍板的决策(继承 PR3, 不再讨论)

1. 跟 FunASR 同一套 pool 架构, 复用 FileBasedProcessPool
2. 不引入 ABC 抽象
3. macOS only
4. PoC 路线不修改 vendor 代码 (本 PR 不动 vendor)
5. 保留 FunASR 路径零侵入

### 已知 TODO(用户已经拒收, 不要做)

- 模型能力评测 (WER / DER 标注集) — 独立工程
- ASR 引擎抽象重构 — 不做
- WebSocket protocol 重构 — 不做
- Qwen3 vendor 内 KV cache 修复 — 本 PR 不做(超出范围)

### 开始

1. Read 上面列的 10 个文件熟悉上下文
2. 跑 `tests/manual/server/qwen3_single_worker_bench.py`(还没写, 先写) 拿到 PR3 单 worker 真实 RTF, 验证 vs PoC 0.118 的差距是不是只来自并发 overhead
3. 用 AskUserQuestion 把 3-4 个关键决策点一次性问完
4. TDD 红/绿/commit 循环把活干完
5. 真服务器长音频 + 资源监控冒烟通过后, 给用户最终汇报(含: 各问题解决前/后 RTF / 资源占用对比 / 149min 末段重复修复验证)
