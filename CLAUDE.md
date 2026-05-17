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

## 当前现状（2026-05-17）

两个引擎并行接入生产，dispatch 走轻量函数路由（**不是 ABC 抽象**）：

- **FunASR**（生产稳定）: `src/core/funasr_transcriber.py`，MPS GPU 加速
- **Qwen3**: `src/core/qwen3_pool_transcriber.py`（多 worker pool）+ `src/core/qwen3_transcriber.py`（单例）+ `src/core/qwen3/asr.py`（引擎构造）。Mac 上 frontend ONNX 走 CoreML ANE（`onnx_provider="COREML_ANE_FE"`），见 `spikes/qwen3_mac_hw_accel/SUMMARY.md`
- **Dispatch**: `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 按 engine 名分支
- **引擎选择优先级**: `upload_request.engine` > `config.transcription.default_engine`（env `FUNASR_DEFAULT_ENGINE`）> `funasr`
- **缓存隔离**: 缓存 key 按 `(file_hash, engine)` 区分，跨引擎不命中

## Qwen3 后处理 pipeline

`Qwen3DiarizeTranscriber.transcribe` 在 ASR + diarize 后串联多层后处理（顺序固定，每层都有 config flag + env override 可关）：

1. **`filter_spurious_speakers`** — 丢掉总时长太小的"假说话人"，把碎片归到时间最近的有效 speaker
2. **`apply_cluster_centroid_merge`**（PR3，`cluster_merge_enabled`）— 多人场景把过聚的 cluster 合并；用 sherpa embedding extractor 算 centroid
3. **`merge_asr_chunks_and_diarize`** — 按 Qwen3 内部 40s chunk 时间窗切文本到 diarize turn
4. **`apply_short_segment_guard`**（PR4，`short_segment_guard_enabled`）— drop 微短段 / ABA 抖动平滑 / 合并连续同 speaker
5. **`apply_silence_align_to_segments`**（spike 405abf6，`silence_align_enabled`）— ffmpeg silencedetect + snap-to-silence 把段切点吸附到最近静音中点，60s podcast +19pp / 60min long +33pp 对齐率，RTF 影响 <1%，见 `spikes/qwen3_silence_align/SUMMARY.md`

加新后处理层：照 `apply_silence_align_to_segments` 的 helper 形状（`(enabled / 空 / 异常)→fallback to input`），挂到 transcribe 流程内同时给 stats 日志，配 5 个 config 字段就行。

## 关键架构决策

**为什么不用 ABC / factory / contract test 抽象**（决策档案: `docs/开发/archive/重构计划-ASR引擎抽象.md`）:

PR1 设计阶段曾计划 ABC 抽象 + factory + contract test 体系（PR2 触发），实际工程化（PR2-4）后判定**过度设计**。当前"全局唯一引擎实例 + 薄 dispatch 路由 + per-engine config 隔离"模式已支撑 2 个引擎 + 后处理 pipeline + 多 worker pool 等复杂需求，工程复杂度更低，第三个引擎接入再触发抽象不迟。

## 加新引擎的步骤
1. 在 `src/core/` 加 `<engine>_transcriber.py`，提供 `get_<engine>_transcriber()` 单例工厂（或 pool wrapper，参考 `qwen3_pool_transcriber.py`）
2. 在 `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 加 `if name == "<engine>": ...` 分支
3. 加 unit test 到 `tests/unit/test_transcriber_dispatch.py`
4. 跑 parity 确认 FunASR + Qwen3 既有路径无回归

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
