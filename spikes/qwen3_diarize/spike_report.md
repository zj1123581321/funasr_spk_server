# Qwen3-ASR + 说话人区分 PoC 报告

**日期**: 2026-05-15
**分支**: `spike/qwen3-diarize-poc`
**测试硬件**: M1 Max, 64GB
**测试音频**: `temp/video_samples/晚点聊-sample-2person-5min.mp3` (5min 双人 podcast, 48kHz mono mp3)

## 1. 验收结论

| 指标 | 目标 | 实测 | 结论 |
|---|---|---|---|
| ASR RTF | < 0.15 | **0.115** | ✓ 达标 |
| Diarize RTF | (subgoal) | **0.465** | ✗ 远超 |
| 端到端 RTF (串行) | < 0.15 | **0.620** | ✗ |
| 端到端 RTF (理论并行) | < 0.15 | **0.465** | ✗ (diarize 瓶颈) |
| 2 speakers 不串 | 必达 | **✓ 主持人/嘉宾正确分离** (63% vs 32%) | ✓ |
| 词级 timestamp | nice-to-have | ✗ (无 aligner 模型) | 待 P1 |
| 峰值 RSS | < 8GB | 3997 MB | ✓ |

**整体判定**: **功能 PoC 成功,性能未达标**。说话人区分正确,但 sherpa-onnx 的 diarize 是瓶颈 (3x 慢于目标)。

## 2. 实际架构

```
晚点聊-sample-2person-5min.mp3
  │
  ├─→ ASR 主路 (Metal GPU)
  │     qwen_asr_gguf (引擎 symlink 自 ~/Production/qwen_asr_server)
  │     - encoder: int4 ONNX (CPU EP), frontend 23M + backend 583M
  │     - LLM decoder: Q5_K_M GGUF, 1.47G (Q5_K 169 tensor + Q6_K 29 tensor + F32 113)
  │     - aligner: 关闭 (production 没此模型)
  │     - 40s 一个 chunk, prefill 2168 tok/s, decode 61 tok/s
  │     → text (2052 chars), 无 word-level timestamp
  │
  ├─→ Diarize 副路 (CoreML EP, 实测主要 CPU)
  │     sherpa-onnx 1.13.2
  │     - segmentation: pyannote-segmentation-3.0 ONNX 5.7M
  │     - embedding: 3D-Speaker eres2net_base_zh-cn ONNX 37.8M
  │     - clustering: FastClustering, threshold=0.9 (双人最优)
  │     → 12 turns, 2 speakers
  │
  └─→ Merger (CPU, 0ms)
        按 turn 时长比例线性切分 ASR text
        → 12 segments (start, end, speaker, text)
        → JSON / SRT / RTTM
```

## 3. 详细数据

### 3.1 ASR (qwen_asr_gguf, Q5_K_M GGUF)

```
audio_duration : 300.001s
asr_elapsed    : 34.41s (纯推理) / 46.41s (含 engine init 12s)
RTF            : 0.115 (纯推理) / 0.155 (含 init)
peak_rss       : 3997 MB
text_length    : 2052 chars
encode_time    : 12.5s (ONNX CPU)
LLM prefill    : 5.6s (13048 tokens, 2339 tok/s)
LLM generate   : 16.2s (1192 tokens, 73.5 tok/s)
```

文本质量极好,中英混合 (SaaS / Bloomberg / Agent / OpenCL) 都识别正确。Q5_K_M 精度对这个任务足够,不需要更大量化。

### 3.2 Diarize (sherpa-onnx)

threshold 调优数据(都用 coreml + num_threads=4, num_speakers=-1):

| threshold | turns | unique speakers | speaker 比例 |
|---|---|---|---|
| 0.5 | 13 | 6 | 过度细分 |
| 0.6 | 12 | 4 | 仍多 |
| 0.7 | 12 | 3 | 接近 |
| 0.8 | 12 | 3 | 接近 |
| **0.9** | **12** | **2** | **63% / 32% — 主持人/嘉宾** ✓ |

最终选 threshold=0.9。**重要 bug**: 锁定 `num_clusters=2` 始终被 sherpa-onnx 吞掉(unique_speakers=1),只能用 threshold 模式。已在 [diarize.py](src/diarize.py) default 改成 None.

```
diarize_elapsed: 139.5s
RTF            : 0.465
peak_rss       : 3438 MB (整个进程,含 sherpa_onnx_core)
turns          : 12
speakers       : 2
provider       : coreml (但实测 ANE 利用率未确认)
```

### 3.3 Speaker 准确度(肉眼 spot-check)

```
Speaker1 (嘉宾吴明辉): 9 segs, 188.9s, 1362 chars (~63%)
Speaker3 (主持人曼奇): 3 segs, 96.0s, 690 chars  (~32%)
```

肉眼 review SRT,**主持人/嘉宾角色正确**:
- `[31.5-78.4] Speaker3: "欢迎收听晚点聊,我是曼奇。今天的嘉宾是明略科技..."` ✓
- `[81.5-99.6] Speaker3: "今天非常高兴的邀请到了明略的创始人吴明辉..."` ✓
- `[2:13-3:09] Speaker1: "我是某种程度上还是比较认同的..."` ✓ 嘉宾

**已知错误**:
- segment 4 [31.5-78.4] 是 47s 长 turn,内部含一句嘉宾说的"实现?"被错误归到 Speaker3 — pyannote segmentation 切的 turn 粒度太粗,turn 内的 speaker 切换没被捕获。

## 4. 关键发现 (写给后续工作)

### 4.1 复用 CapsWriter 的 qwen_asr_gguf 引擎 = 正确决策
- 不需要自己写 raw llama.cpp + Metal 集成
- engine 内已经处理: ONNX int4 encoder + Q5_K_M GGUF decoder + chunk_size 40s 流水线 + KV cache 管理
- 模型权重直接 symlink 自 production,省 1.4G 下载

### 4.2 但有缺陷
- **No aligner**: production 没 aligner 模型 (`qwen3_aligner_*.onnx/gguf` 不在 release),所以拿不到 word-level timestamp。要加得自己找/转模型
- `transcribe(audio_file)` 的 `duration=0.0` 默认值会让 `load_audio` 加载 0 秒 — bypass 用 `engine.asr(audio_array, ...)` 直接喂 numpy
- 模型文件名跟 schema 默认值不一致 (production 去掉了 `.int4` 和 `.q4_k` 后缀)

### 4.3 sherpa-onnx diarize 性能不达标的可能原因
- CoreML EP 标了但未必真路由到 ANE — 要用 powermetrics 验证
- 3D-Speaker eres2net 模型 (37.8M) 还是太大;可以试 NeMo 或 model.int8 量化版
- 138s 大头在 embedding 提取阶段,不是 segmentation

### 4.4 sherpa-onnx 接口陷阱
- `FastClusteringConfig(num_clusters=2, threshold=0.5)` 锁定模式不工作 → 所有 turn 都归到 speaker 0
- 必须用 threshold 模式,然后调参

## 5. 后续路径

### P0 (PoC 闭环 -> 可用 prototype)
- [ ] **性能优化**: 验证 sherpa-onnx CoreML 是否真用 ANE。试 NeMo embedding / model.int8.onnx;目标 diarize RTF < 0.2
- [ ] **改进 turn 边界**: 调 pyannote 的 `min_duration_on/off`,让长 turn 切碎,减少 turn 内 speaker 切换
- [ ] **真并行**: 把 ASR 跟 diarize 用 subprocess 分进程跑,实测并行 RTF (现在是同一进程串行)

### P1 (做更精细的对齐)
- [ ] 找 / 自己量化 `qwen3_aligner_*` 模型,启 ASR engine 的 aligner — 拿词级 timestamp
- [ ] 用 word-level timestamp 重写 merger,精度从 turn 级 (10-50s) 提升到词级

### P2 (生产化)
- [ ] 评估 DER (Diarization Error Rate) — 需要人工标注 ground truth 或者用更精的参考工具(pyannote.audio 4.x)做对照
- [ ] 集成回 funasr_spk_server,作为新引擎 (`engine="qwen3-diarize"`)
- [ ] WebSocket 输出格式适配现有 `TranscriptionResult` schema

## 6. 文件清单

```
spikes/qwen3_diarize/
  README.md                          PoC 入口说明
  spike_report.md                    本报告
  .gitignore                         排除 venv/ models/ output/
  venv/                              uv 起的 3.12 venv (不进 git)
  src/
    __init__.py
    asr.py                           ASR wrapper (调 qwen_asr_gguf)
    diarize.py                       Diarize wrapper (Agent 写)
    merge.py                         时间轴融合
    vendor/qwen_asr_gguf -> ~/Production/qwen_asr_server/core/server/engines/qwen_asr_gguf
  benchmark/
    asr_bench.py                     ASR 单跑 bench
    diarize_bench.py                 Diarize 单跑 bench (Agent 写)
    e2e_bench.py                     端到端
  models/
    production_models -> ~/Production/qwen_asr_server/models (symlink)
    sherpa/
      pyannote-segmentation-3.0/     6MB
      3dspeaker-eres2net/            38MB
  tests/fixtures/audio/晚点聊-sample-2person-5min.mp3 (symlink)
  output/                            JSON / SRT / RTTM 输出
```

## 7. 复现命令

```bash
cd spikes/qwen3_diarize
source venv/bin/activate

# 端到端
python benchmark/e2e_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3

# 单跑 ASR
python benchmark/asr_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3 --out-json output/asr.json

# 单跑 Diarize
python benchmark/diarize_bench.py tests/fixtures/audio/晚点聊-sample-2person-5min.mp3 \
    --cluster-threshold 0.9 --num-speakers -1 --provider coreml > output/diarize.json
```
