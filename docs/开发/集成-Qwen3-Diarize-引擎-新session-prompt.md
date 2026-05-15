# 新 session 启动 prompt — 集成 Qwen3-Diarize 引擎进主项目

> **生成于**: 2026-05-15
> **PoC 分支**: `spike/qwen3-diarize-poc` (已 push origin, 5 commit)
> **集成目标**: 把 PoC 落地为正式引擎,与 FunASR 引擎并存,全局只允许一个运行

新 session 直接复制下面内容到对话开头使用.

---

## Prompt 正文(复制以下内容)

我需要把 PoC 集成到 funasr_spk_server 主项目, 作为可选的第二个 ASR 引擎,与现有 FunASR 引擎并存, **全局只允许一个引擎运行, 由配置文件指定**.

### 你需要先读的(按重要度排序)

1. **`spikes/qwen3_diarize/spike_report.md`** — PoC 完整报告 (权威 PoC 上下文)
2. **`docs/开发/重构计划-ASR引擎抽象.md`** — PR1/PR2 路线图, 看第 8 节"PR2 触发条件"
3. **`CLAUDE.md`** — 项目代码 / 测试 / 部署约定 (macOS only, 不 Docker)
4. **`src/core/transcriber_dispatch.py`** — PR1 留下的 dispatch 接口
5. **`src/core/funasr_transcriber.py`** — 现有 FunASR 引擎的实现 + 调用契约
6. **`src/core/schemas.py`** + **`src/core/database.py`** — TranscriptionResult schema 和 cache key 三联

### PoC 关键事实(避免重复探索)

- **分支**: `spike/qwen3-diarize-poc` (已 push origin), 5 commit
- **PoC 位置**: `spikes/qwen3_diarize/`
  - `src/asr.py` / `src/diarize.py` / `src/merge.py` — 三模块 wrapper
  - `src/vendor/qwen_asr_gguf` — symlink 自 `~/Production/qwen_asr_server/core/server/engines/qwen_asr_gguf/` (CapsWriter-Offline upstream)
  - `benchmark/parallel_e2e_bench.py` — 真并行 e2e + 资源监控
- **PoC venv 是 Python 3.12** (CapsWriter 引擎要求), 主项目 venv 是 3.11.9
- **模型**: GGUF Q5_K_M 1.47GB + encoder ONNX 606MB + sherpa pyannote 5.7MB + NeMo TitaNet 38MB
- **生产推荐 preset**: `auto` (NeMo + threshold 0.9 + 自动聚类, 2/4 人未知场景都验证过)
- **实测性能**:
  - 单任务 wall RTF 0.108-0.118 (5min 双人 / 8min 4 人)
  - N=3 并发 throughput 顶峰 2.12 任务/min
  - 内存 RSS ~4.5GB/instance (mmap 共享后实际增量 1-2GB)
  - powermetrics 直接验证 ANE 全程 0W, GPU N=3 时 97% 满载

### 已经拍板的决策(不要再问)

1. **全局唯一引擎**: 由 `config.transcription.engine = "funasr" | "qwen3_diarize"` 决定, 启动时只 instance 一个
2. **`upload_request.engine` 字段保留但严格 validate**:
   - 不传 → 默认用 config
   - 传了相同的 → 通过
   - 传了不同的 → reject + 明确 error "Server configured with X, cannot accept Y"
3. **`database` 表的 engine 列保留** (缓存 key 三联仍要按 engine 区分, 跨时期跨部署有意义)

### 必须先和我对齐再动手的剩余决策点

**用 AskUserQuestion 问我**:

1. **PoC 的 qwen_asr_gguf 引擎代码归属**:
   - (A) 拷进 `src/core/vendor/qwen_asr_gguf/` (自洽但占体积, ~50MB Python 源码 + 2.5MB dylib)
   - (B) Git submodule 指向 HaujetZhao/CapsWriter-Offline 子目录 (纠结于 upstream license + 跟踪同步)
   - (C) 部署时手动 symlink (跟 PoC 一致, 简单但部署文档复杂)

2. **Python 版本**:
   - (A) 主项目升 3.12 (影响 FunASR 引擎, 要全量 parity 回归)
   - (B) Qwen3 引擎跑独立 venv + subprocess 调用 (隔离但复杂)
   - (C) 试 CapsWriter 引擎在 3.11 跑得起来吗 (可能可行, 需测)

3. **模型权重部署**:
   - (A) 写 `scripts/download_models.sh`, 部署时手动跑
   - (B) 启动时按需下载 (首次启动慢 5-10min, 需要网)
   - (C) 文档说明, 运维手动放置

4. **PR1 留的 `qwen3` 占位名 vs 新引擎名**:
   - 接管 PR1 的 `qwen3` 标识 (推荐, 简洁)
   - 用新名 `qwen3_diarize` 区分

### 工作产出要求

一个 PR, 包含:

- 新增 `src/core/qwen3_diarize_transcriber.py` (或同等命名), 实现与 FunASR 对齐的契约:
  `transcribe(audio_path, task_id, progress_callback, output_format) → (TranscriptionResult, raw_result)`
- 改 `src/core/transcriber_dispatch.py` (改成全局唯一引擎 loader, 启动时根据 config 决定)
- 改 `src/api/websocket_handler.py` (engine 字段 validation)
- 改 `config.json` + `.env.example` (新配置项)
- 新增 `tests/unit/test_qwen3_diarize_transcriber.py` + `tests/integration/test_qwen3_diarize_e2e.py`
- 跑 FunASR parity 测试零回归:
  `FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/test_parity_funasr_semantic.py`
- 更新 `README.md`、`docs/部署.md`、`docs/开发/重构计划-ASR引擎抽象.md` (PR2 完成复盘)

### 工作方式约束(非常重要)

**1. 严格 TDD — 红/绿/commit 三步循环, 不可省略**

每加一个功能都按以下顺序:

- **红**: 先写测试, 跑 → 确认 fail (`-x` 选项让 pytest 停在第一个错误处加快迭代)
- **绿**: 写最小实现让测试过, 跑 → 确认 pass
- **commit**: 用 git commit 沉淀这个"红 → 绿"闭环, 一个 commit 一件事
- 然后下一个 cycle, 不要积累多个改动一次性提交

**对应到本任务的具体场景**:
- `Qwen3DiarizeTranscriber.transcribe()` 的契约测试 (mock 内部 ASR/diarize 调用, 验证返回 `(TranscriptionResult, raw_result)` 结构) — 红/绿/commit
- `transcriber_dispatch.resolve_transcriber()` 改成全局唯一 loader 的测试 (mock config, 验证启动时只 init 一个) — 红/绿/commit
- engine 字段 validate 的测试 (3 个 case: 不传 / 传相同 / 传不同) — 红/绿/commit
- 端到端集成测试 (`tests/integration/test_qwen3_diarize_e2e.py`, 类似现有 funasr parity test) — 红/绿/commit

**禁止**:
- ❌ 一次性写完所有代码再写测试 (这不是 TDD)
- ❌ 跳过红的阶段直接写"绿色"实现 (没人能验证测试本身正确)
- ❌ 一个 commit 塞 N 个功能

**2. 自主决策, 不打断用户**

- 仅在 4 个明确列出的剩余决策点上, 用 AskUserQuestion 一次性问完, 之后**全程不再向用户提问**
- 遇到任何非用户独有的技术问题(包括但不限于: 包版本冲突、API 接口选择、文件命名、目录结构、import 路径、错误处理策略、log 格式、测试覆盖范围、commit message 写法), **自己决定并推进**
- 真正需要用户介入的只有: 凭证 / 密码 / 外部账号 / 用户私人数据 / 部署环境特定的细节
- 任务完成前不要主动 ExitPlanMode 或 stop 等待"用户确认"
- **任务定义为完成**的标志: PR 内所有代码写完、所有测试绿、`FUNASR_RUN_INTEGRATION=1 pytest` 跑通无回归、相关文档更新, 这时候才给用户最终汇报

**3. 其他**

- macOS only (CLAUDE.md 已约定)
- 不修改 `spikes/` 下的代码 (历史记录)
- TaskCreate 拆解工作, 完成一项 mark 一项

### 开始

1. Read 上面列的 6 个文件熟悉上下文
2. 用 AskUserQuestion 问 4 个剩余决策点 (一次性问完)
3. 拿到决策后, **全程不再向用户提问**, 按 TDD 红/绿/commit 循环把活干完
4. 任务完成后给用户最终汇报
