Always respond in 中文

# 代码设计上的要求
- 各部分功能尽量低耦合，高内聚。
- 各个函数代码做好注释。
- 有完备的日志系统，方便后期调试确认问题。
- 避免写冗余代码，提高项目的可复用性。
- 尽力遵循工程上的最佳实践。适当使用文件夹来增加整个项目的可读性，但不要添加过多无关文件。
- 及时完善 .gitignore 文件。

# 测试约定（PR1 之后启用）

## pytest 套件
- 测试目录三层：
  - `tests/unit/` — 单元测试（mock 外部依赖，毫秒级）
  - `tests/integration/` — 端到端集成（含真实 FunASR 模型）
  - `tests/manual/` — 历史手工脚本（**不在 pytest 收集范围**），仅作复现参考
- 推荐命令：`venv/bin/python -m pytest`（venv 的 pytest binary shebang 漂移，用 `-m pytest` 走 venv python）
- integration 默认 skip，需 `FUNASR_RUN_INTEGRATION=1` 才跑

## TDD 流程
- 新功能 / 改 bug 都先写测试再改代码
- 红 → 绿 → commit 是最小单位，**不要积累多个改动一次性提交**
- 每个 commit message 写清楚改了什么 + 解决什么问题

## Parity 测试
- 改动 FunASR 路径（schemas / database / task_manager / websocket_handler / funasr_transcriber）后必跑：
  ```
  FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/
  ```
- 通过 = 改动安全；失败 = 你改坏了 FunASR 路径，回去查
- golden baseline 在 `tests/fixtures/golden/`，由首次运行自动生成

# ASR 引擎架构

## 当前现状（2026-05-23）

两个引擎并行接入生产，dispatch 走轻量函数路由（**不是 ABC 抽象**）：

- **FunASR**（生产稳定）: `src/core/funasr_transcriber.py`，MPS GPU 加速
- **Qwen3**: `src/core/qwen3_pool_transcriber.py`（**runtime-aware 池 dispatch**, 见下文）+ `src/core/qwen3_transcriber.py`（单例）+ `src/core/qwen3/asr.py`（引擎构造）。Mac 上 frontend ONNX 走 CoreML ANE（`onnx_provider="COREML_ANE_FE"`），见 `spikes/qwen3_mac_hw_accel/SUMMARY.md`
- **Dispatch**: `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 按 engine 名分支
- **引擎选择优先级**: `upload_request.engine` > `config.transcription.default_engine`（env `FUNASR_DEFAULT_ENGINE`）> `funasr`
- **缓存隔离**: 缓存 key 按 `(file_hash, engine)` 区分，跨引擎不命中

## Runtime + Diarize backend 抽象

`src/core/runtime.py` 的 `detect_runtime()` 返回 `MacRuntime / CudaRuntime / CpuRuntime`，按 `sys.platform` + `onnxruntime CUDAExecutionProvider` 探测自动选；`FUNASR_RUNTIME=cpu/mac_ane/cuda` env 强制 override。每个 runtime 暴露：
- `validate()` — `CudaRuntime` 显式 assert CUDA EP 在 ORT providers 列表，缺则 fail-fast（替代 ORT silent CPU fallback）
- `recommend_diarize_backend()` — Mac/Cpu → `sherpa`, Cuda → `ort_cuda`
- `recommend_num_threads()` — Mac 固定 4, Linux 按 `cpu_count()` 给 2/4

Qwen3 diarize 有 **两个 backend 实现**，通过 `src/core/qwen3/diarize.py:run_diarization_dispatched` 路由：

- **`sherpa`**（默认，Mac/Cpu）: `src/core/qwen3/diarize.py:run_diarization`，sherpa-onnx `OfflineSpeakerDiarization` + sherpa C++ FastClustering
- **`ort_cuda`**（CUDA 平台默认）: `src/core/qwen3/diarize_ort.py:run_diarization_ort_cuda`，Python `onnxruntime` 直 wrap pyannote-segmentation-3.0 + TitaNet + scipy 复刻 FastClustering（cosine + complete linkage）。**8 vCPU + RTX 3060 上 30min wall RTF 0.047 vs sherpa CPU 0.080**，详见 `docs/开发/gpu加速/2026-05-22-ORT-CUDA-diarize-backend.md`
- **优先级**: 显式 `backend` 参数 > `FUNASR_QWEN3_DIARIZE_BACKEND` env > `runtime.recommend_diarize_backend()`
- **为什么自建 ort_cuda 而不是用 sherpa CUDA build**: sherpa-onnx CUDA build 的 C++ wrapper 跟 llama.cpp CUDA 撞 segfault，ORT Python API 不撞（`scripts/_remote_ort_cuda_clash_check.py` 验证过）

## Qwen3 池 dispatch (runtime-aware)

`src/core/qwen3_pool_transcriber.py:get_qwen3_pool_transcriber()` 按 `detect_runtime()` 分发到两套池：

- **`cuda` runtime** → **`Qwen3InProcPool`** (`src/core/qwen3_inproc_pool.py`)
  - 单进程内 N 个 `Qwen3DiarizeTranscriber` 实例，`asyncio.Queue` 调度 acquire/release
  - 共享同一 cuda context，**race-free** — 避开 multi-process 跨进程 CUDNN/cuda buffer race
  - RTX 3060 + 8 vCPU 实测 `pool_size=2` 跑 1800s × 2 并发: TOTAL_WALL 142s, 每 task RTF 0.079
  - 详见 `docs/开发/gpu加速/2026-05-23-CUDA并发突破.md`
- **其他 runtime (Mac/CPU)** → **`Qwen3PoolTranscriber`** (file-based multi-process pool, 历史路径)
  - `FileBasedProcessPool` 派发到 `qwen3_worker_process.py` subprocess
  - Mac MPS 上行为 100% 不变

**为什么不统一一套池**: CUDA 多进程下 cuDNN handle 跨进程 race 撞死 worker (实测 MPS 任何 thread% 设置都不解); Mac 多进程下 sherpa CPU 没此问题, multi-process 隔离反而更稳. **rule-of-two backends, runtime 自动选**.

`pool_size` 两套共用 `config.transcription.qwen3_pool_size` (env `FUNASR_QWEN3_POOL_SIZE` 覆盖). 单例缓存在 `_qwen3_pool_singleton`, 测试用 `reset_qwen3_pool_singleton()` 清.

## Qwen3 后处理 pipeline

`Qwen3DiarizeTranscriber.transcribe` 在 ASR + diarize 后串联多层后处理（顺序固定，每层都有 config flag + env override 可关）：

1. **`filter_spurious_speakers`** — 丢掉总时长太小的"假说话人"，把碎片归到时间最近的有效 speaker
2. **`apply_cluster_centroid_merge`**（PR3，`cluster_merge_enabled`）— 多人场景把过聚的 cluster 合并；用 sherpa embedding extractor 算 centroid。dominant share ≥ 0.6 时还会用更宽松的 `cluster_merge_dominant_minor_threshold`（默认 0.5）把跟 dominant 接近的 minor cluster 也合到 dominant（兜底拦截解码器漂移引入的中长噪声 cluster，见 `docs/开发/archive/spk-over-detect-归因调研结果.md`）
3. **`merge_asr_chunks_and_diarize`** — 按 Qwen3 内部 40s chunk 时间窗切文本到 diarize turn
4. **`apply_short_segment_guard`**（PR4，`short_segment_guard_enabled`）— drop 微短段 / ABA 抖动平滑 / 合并连续同 speaker
5. **`apply_silence_align_to_segments`**（spike 405abf6，`silence_align_enabled`）— ffmpeg silencedetect + snap-to-silence 把段切点吸附到最近静音中点，60s podcast +19pp / 60min long +33pp 对齐率，RTF 影响 <1%，见 `spikes/qwen3_silence_align/SUMMARY.md`

加新后处理层：照 `apply_silence_align_to_segments` 的 helper 形状（`(enabled / 空 / 异常)→fallback to input`），挂到 transcribe 流程内同时给 stats 日志，配 5 个 config 字段就行。

### ⚠️ Qwen3 worker audio 转换的边界

`src/core/qwen3_worker_process.py` 对 sherpa-supported 格式（`{.wav, .flac, .ogg, .mp3, .opus}`）**跳过** ffmpeg `convert_to_wav`，仅 m4a/aac/mp4/mov/webm 走转码（sherpa libsndfile/librosa fallback 读不了）。**不要为了"统一"把所有格式都 ffmpeg 转 wav** — 调研（`docs/开发/archive/spk-over-detect-归因调研结果.md`）证明 ffmpeg→wav 即使保持 16kHz mono 也会改变 audio 字节，通过 pyannote+TitaNet embedding 放大，触发 sherpa diarize FastClustering over-detect（60min-2spk 真实场景 → 4 spk）。

## 关键架构决策

**为什么不用 ABC / factory / contract test 抽象**（决策档案: `docs/开发/archive/重构计划-ASR引擎抽象.md`）:

PR1 设计阶段曾计划 ABC 抽象 + factory + contract test 体系（PR2 触发），实际工程化（PR2-4）后判定**过度设计**。当前"全局唯一引擎实例 + 薄 dispatch 路由 + per-engine config 隔离"模式已支撑 2 个引擎 + 后处理 pipeline + runtime-aware 池 dispatch（in-proc / multi-process 两套）等复杂需求，工程复杂度更低，第三个引擎接入再触发抽象不迟。

## 加新引擎的步骤
1. 在 `src/core/` 加 `<engine>_transcriber.py`，提供 `get_<engine>_transcriber()` 单例工厂（或 pool wrapper，参考 `qwen3_pool_transcriber.py`）
2. 在 `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 加 `if name == "<engine>": ...` 分支
3. 加 unit test 到 `tests/unit/test_transcriber_dispatch.py`
4. 跑 parity 确认 FunASR + Qwen3 既有路径无回归

## 加新 diarize backend 的步骤
1. 在 `src/core/qwen3/diarize_*.py` 加新文件，暴露 `run_diarization_<backend>(audio_path, ..., num_speakers, cluster_threshold, ...) -> list[dict]`，输出 schema 跟 sherpa `run_diarization` 一致（`[{"start", "end", "speaker"}, ...]`）
2. 在 `src/core/qwen3/diarize.py:run_diarization_dispatched` 加 `if backend == "<name>": from ... import run_diarization_<backend>; return run_diarization_<backend>(...)` 分支
3. 在 `src/core/runtime.py` 的 `RuntimeEnvironment.recommend_diarize_backend()` 实现里返回新 backend 名（如果新 backend 是某 runtime 默认）
4. 单测 mock ORT session 验证 backend 行为（参考 `tests/unit/test_diarize_ort_backend.py`），integration 加 parity 测试（`tests/integration/test_diarize_ort_parity.py`）

# Config 体系（2026-05-22 治理后）

`src/core/config.py` 是单一 source of truth, Pydantic schema + 4 层优先级 + runtime-aware sentinel.

## 优先级链

```
defaults < config.json < FUNASR_PROFILE < FUNASR_* env
```

每层只填上层留空的字段; 显式覆盖永远胜出. 启动日志会列出"FUNASR_PROFILE=X applied. 覆盖字段(N): ..."防止"我明明 config.json 写了 X 怎么变 Y"的惊讶感.

## FUNASR_PROFILE 套餐 env

一行 env 切平台 / 切环境, 不再手工拼 5+ `FUNASR_*` env:

| profile | port | engine | qwen3_pool | encoder | log |
|---|---|---|---|---|---|
| `mac_prod` | 8767 | qwen3 | 3 | coreml_ane_full | INFO |
| `mac_dev` | 8867 | qwen3 | 2 | coreml_ane_full | DEBUG |
| `cuda_prod` | (默认 8767) | qwen3 | 2 | cuda | INFO |
| `cuda_dev` | 8867 | qwen3 | 2 | cuda | DEBUG |

用法: `FUNASR_PROFILE=cuda_dev venv/bin/python run_server.py`. 未知 profile name → warn + ignore, 不挂. 加新 profile 改 `src/core/config.py` 的 `PROFILES` dict.

## "auto" sentinel 字段 (Pydantic model_validator 解析)

`Qwen3Config.num_threads` / `provider` 默认 `"auto"`, 在 model load 时一次性解析:

```python
num_threads="auto" → detect_runtime().recommend_num_threads()
  · MacRuntime  → 4 (PoC: t=4 比 t=8 wall -11.5%)
  · CudaRuntime → 2/4 按 vCPU (≤4=2, ≥5=4)

provider="auto" → "cpu"  # sherpa embedding extractor 跨 runtime 都 cpu
```

字段类型 `int | str`, 解析后保证 int, 下游 5 个消费点 (transcriber:58/237/399/575, inproc_pool:117) 永远拿到具体 int. 显式 int / 显式 provider 字符串不被覆盖.

## vendor 字段进 Pydantic

vendor 不再 `os.environ.get(...)`, 全走 Qwen3Config:

- `backend_mlpackage_units: Literal["CPU_AND_NE", "CPU_AND_GPU", "ALL"] = "CPU_AND_NE"` — Phase 3 backend mlpackage compute_units
- `encoder_timing_enabled: bool = False` — encoder 打印 mel/fe/be 耗时 (排查性能用)

env override 仍可用 (`FUNASR_QWEN3_BACKEND_MLPACKAGE_UNITS` / `FUNASR_QWEN3_ENCODER_TIMING`), 但走 `_override_if_set` 标准路径, Pydantic 看到, `print_config` 显示.

## startup engine-runtime fail-fast

`Config._validate_engine_runtime`: `default_engine=qwen3` + cuda runtime + ORT CUDA EP 缺 → `sys.exit(1)`, 报 "用 FUNASR_RUNTIME=cpu 降级或修依赖". 不再 lazy 等第一个 task 才挂.

per-request `engine ≠ default_engine` 已被 `transcriber_dispatch.py:57` 拒, 所以 startup 仅查 default_engine 充分.

## 切换设备 / 切引擎操作手册

不同部署目标 × 不同引擎的常见组合, 推荐姿势:

### A. Mac + Qwen3 (主路径)

```bash
# prod (PM2 守护)
FUNASR_PROFILE=mac_prod pm2 start ecosystem.config.cjs

# dev (前台直跑, 看日志)
FUNASR_PROFILE=mac_dev venv/bin/python run_server.py
```

### B. Linux CUDA + Qwen3 (远端 dev box / 未来 prod)

```bash
export FUNASR_PROFILE=cuda_dev   # 或 cuda_prod
# LD_LIBRARY_PATH 配 CUDA libs (远端启动脚本里设, 详见 scripts/_remote_*.sh)
venv/bin/python run_server.py
```

### C. Mac/Linux + FunASR (历史路径 / 回归对比)

所有 profile 默认都是 qwen3 引擎, 想用 FunASR 走 env 覆盖:

```bash
# 干净法 (走 config.json 默认 port/log, 推荐)
FUNASR_DEFAULT_ENGINE=funasr venv/bin/python run_server.py

# 套餐 + 覆盖法 (沿用 profile 的 port/log, 只改引擎)
FUNASR_PROFILE=mac_dev FUNASR_DEFAULT_ENGINE=funasr venv/bin/python run_server.py
```

注: FunASR 不支持 CUDA, Linux 上自动走 CPU (MPS 仅 Mac).

### "改哪里" 决策树

```
要长期保留这个配置吗?
├── 是 → 改文件
│       ├── 单机日常用法         → 本机 .env 写 FUNASR_PROFILE=xxx
│       ├── 多机同环境共用       → config.json (但 profile 通常已够用)
│       └── 加一个新部署目标     → src/core/config.py 的 PROFILES dict
└── 否 (一次性 / 调试) → 命令行 export FUNASR_xxx=yyy
```

### 最常见 3 种用法

1. **dev 机切环境** (最常用) — 本机 .env 一行:
   ```
   FUNASR_PROFILE=mac_dev   # 或 cuda_dev
   ```
2. **临时换引擎调试** — 命令行 cover:
   ```bash
   FUNASR_PROFILE=mac_dev FUNASR_DEFAULT_ENGINE=funasr venv/bin/python run_server.py
   ```
3. **加新部署目标** (新机器 / 新硬件配置) — 改 `src/core/config.py:PROFILES`:
   ```python
   PROFILES = {
       ...
       "cuda_l40_prod": {  # 例: L40 GPU, pool 4
           "transcription": {"default_engine": "qwen3", "qwen3_pool_size": 4},
           "qwen3": {"asr_encoder_provider": "cuda"},
       },
   }
   ```

### 启动后怎么验证配置生效

启动日志会打印 (`_apply_profile_defaults` 输出):
```
FUNASR_PROFILE=cuda_dev applied. 覆盖字段 (5):
  server.port('(默认)'→8867),
  transcription.default_engine('(默认)'→'qwen3'),
  ...
```

如果发现某字段没被 profile 覆盖, 100% 是 `.env` 或 shell env 里设了更高优先级的 `FUNASR_*`. grep `.env` 找污染源 (清完再起服务).

## 加 config 字段的步骤

1. `Qwen3Config` (或对应 BaseModel) 加字段 + 默认值 + docstring
2. `_apply_env_overrides` 加一行 `_override_if_set(...)` 注入 env
3. 如有"运行时感知"需求 → 默认值用 `"auto"` sentinel + 在 `_resolve_auto_sentinels` model_validator 加分支
4. 如有平台/环境 profile 差异 → 加到 `PROFILES[<profile_name>]`
5. unit test 覆盖默认值 / env override / profile / "auto" 解析 (mock detect_runtime)

# 部署约定（macOS only）

本项目仅在 macOS Apple Silicon 上运行（依赖 MPS GPU 加速）。**不要调用全局 `docker-deploy` skill**。

prod/dev 物理隔离：

| 环境 | 目录 | 端口 | 守护 |
|---|---|---|---|
| prod | `~/Production/funasr_spk_server/` | 8767 | **PM2** (`funasr-server`) |
| dev | `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server/` | 8867 | **不挂 PM2**（前台直跑） |

## prod 部署
```bash
cd ~/Production/funasr_spk_server
git pull origin main
venv/bin/pip install -r requirements.txt   # 仅 requirements 有变化时
pm2 restart funasr-server                  # 启动数据库自动迁移
```

## dev 运行
默认前台直跑，方便看日志和调试：
```bash
venv/bin/python run_server.py
```
仅在需要长跑调试时才用 PM2：`pm2 start ecosystem.config.cjs`，用完 `pm2 delete funasr-server-dev && pm2 save`。

详细部署文档：`docs/部署.md`
