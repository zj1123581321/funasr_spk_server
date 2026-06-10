# Qwen3 词级时间戳 PoC — SUMMARY

> 配套计划:`docs/开发/2026-06-09-qwen3-词级时间戳-PoC计划.md`
> 指标口径:AAS(平均绝对偏移 ms)、RTF(Mac 实测)、长音频是否崩。

> **✅ 已落地生产实现(2026-06-09)**:选型 MMS-300M CTC-FA,增量挂 `segment.words`(`word_align_enabled` 默认关 opt-in)。实现见 `src/core/qwen3/word_align.py` + `CLAUDE.md`「Qwen3 后处理 pipeline」5.5 层;落地清单见 PoC 计划文档顶部「已落地实现」。集成测试 `tests/integration/test_qwen3_word_align_e2e.py` 真 MMS 跑通(podcast 60s 出词、词时间落段窗内、覆盖率≥0.5)。

## 3060 CUDA 实测(2026-06-10,probe `scripts/_remote_word_align_rtf_probe.py`)

RTX 3060 上 word_align 走 `CUDAExecutionProvider`,podcast 60s 端到端(ASR cuda + diarize ort_cuda + MMS cuda):

| | RTF | 备注 |
|---|---|---|
| OFF(纯 ASR+diarize) | **0.0461** | |
| ON(加词级时间戳) | **0.0570** | |
| **增量 Δ** | **+0.0109(~1.1%)** | 对比 Mac CPU 的 +0.166,快一个数量级 |

- 出词 368(挂段 349),`failed_windows=0`;MMS ONNX session 首次 build ~2.3s(一次性),稳态每次 +0.65s wall(60s 音频)。
- **✅ 共存安全**:MMS CUDA + llama.cpp CUDA(Qwen3 ASR)+ ORT CUDA diarize 三者进程内共存,跑完未 segfault(验证了 sherpa CUDA build 曾撞 llama.cpp 的同类风险在 ORT Python API 路径下不复现)。
- **⚠️ 部署陷阱**:装 `ctc-forced-aligner` 会拉 CPU `onnxruntime` 覆盖 `onnxruntime-gpu` 的 capi → CUDA provider 消失;CUDA 机器装后须 `pip uninstall onnxruntime + force-reinstall --no-deps onnxruntime-gpu`(见 requirements.txt 注释)。
- 结论:**CUDA 上词级时间戳几乎免费,可放心常开;Mac CPU 上 +17% 需权衡。** diarize 开关议题可塌缩成纯布尔(时间戳恒有基线)。

## PoC-B — FunASR fa-zh 轻量 forced aligner ✅(2026-06-09)

脚本:`scripts/poc_b_fa_zh.py`(ASR 全文当参考,fa-zh 吃 `(audio, text)` → 字级时间戳)。
模型:`fa-zh` = `iic/speech_timestamp_prediction-v1-16k-offline`,model.pt 151MB,modelscope 自动下。

### 短样(podcast_2speakers_60s.wav,60s,360字)
| 指标 | 值 |
|---|---|
| fa-zh 首次加载 | 18.4s(冷),复用后 4.8s |
| **对齐耗时** | **0.76s** |
| **RTF** | **0.013**(Mac CPU) |
| 输出 | 329 字 char 级 `[start_ms,end_ms]` |
| 目测准度 | 好 —— 「法」130-370ms 对上 golden 首段 0.11s,后续连续递增落在段窗内 |
| 输出 | `outputs/poc_b_60s.json` |

**关键结论:RTF 0.013 → 轻量对齐器常开只加 ~1.3% 开销,跟 diarize ~40% 不是一个量级。
"所有模式恒开词级时间戳"几乎白嫖,不影响 diarize 开关经济账。** 直接回答交接文档的核心顾虑。

### 长音频(audio_149min.mp3,8938s,63445字 monolith)
- **一次喂 monolith → 静默死(无结果 JSON,leaked-semaphore 清理告警)= OOM 级。**
- 性质:**内存/规模限制,非 Qwen 那种 320s 对齐质量硬墙。**
- 解法:生产本就按 diarize turn / 40s ASR chunk 分段喂,不会喂 monolith。短样已证分段 RTF 0.013、准。
  → **fa-zh 长音频路径 = 照现有分段喂,天然解决,非 blocker。**

## PoC-A — Qwen vendor aligner ✅(2026-06-09,在 3060 box 跑)

权重在 3060:`/data/projects/CapsWriter-Offline-with-AI/models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B/`。
脚本 `scripts/poc_a_qwen_aligner.py`,model_dir 直接指过去(不拷),override `llm_fn=q5_k`,encoder CPU + GGUF CUDA。

### 短样(podcast_2speakers_60s,60s)
| 指标 | 值 |
|---|---|
| aligner 加载 | 1.35s |
| **对齐耗时** | **1.42s**(encoder 1.10 + decoder 0.29) |
| **RTF** | **0.024**(3060 CUDA) |
| 输出 | 360 项(含标点),字级 start/end |
| 目测准度 | 好 —— 「法」0-0.24s,后续连续递增,跟 fa-zh 一致 |
| 输出 | `outputs/poc_a_60s.json` |
- libllama.so + libggml-cuda.so 在 box vendor bin/,GGUF 上 CUDA 正常,无 segfault。

## 头对头 A vs B(`scripts/compare_a_b.py`,302 个可比汉字)
两个**独立**对齐器逐字 start 一致度(互证,非 vs 真值):
- **AAS 183ms**,p50 150ms,p90 370ms
- 命中率 ≤100ms 36%,≤200ms 66%,**≤500ms 95%**
→ 不同声学前端,字级 ~150ms 分歧正常,两条路线互证都靠谱。

## PoC-C — 现状段级边界 vs 词级(`scripts/poc_c_segment_boundary.py`)
golden 段 start/end(= 现状 `字符比例+静音吸附` 产出)vs fa-zh 逐字:
- 段【起点】偏移:**AAS 263ms** p90 860ms max 860ms
- 段【终点】偏移:**AAS 200ms** p90 560ms max 560ms

**解读:现状段边界已 ~200-260ms 级别(silence-align 的功劳),没漂到秒级。词级的真正增量不是"纠段边界",而是"段内逐字/逐词时间戳"——这是现状完全没有的能力。**

## 三个未知数 → 全部回答
1. **feasibility**:A、B 都跑通、都准。✅
2. **常开代价**:RTF A=0.024(CUDA)/ B=0.013(Mac CPU),都 ~1-2%,**远小于 diarize ~40%。词级可恒开,diarize 开关议题可塌缩成纯布尔。** ✅
3. **值不值**:段级现状已 ~200ms 准;词级 ROI 取决于下游是否需要**字/词级粒度**(卡拉OK高亮 / 点词跳转 / 精确检索片段)。是产品取舍。✅

## PoC-MMS — MMS-300M CTC-FA(轻量多语种)✅(2026-06-09,Mac CPU)

`ctc-forced-aligner` 1.0.2(**deskpai ONNX fork**,非 MahmoudAshraf torch 版),走 onnxruntime CPU,uroman 罗马化,内置 30s+2s overlap 切块。脚本 `scripts/poc_mms_ctc.py`。依赖补装:`unidecode`。

### 中文 60s
| 指标 | 值 |
|---|---|
| 加载 | 0.76s(冷 52s 含下 ONNX) |
| 对齐耗时 | 9.96s |
| **RTF** | **0.166**(Mac CPU) |
| 输出 | 360 字 |
| 输出 | `outputs/poc_mms_60s.json` |

### 中英混排 45s(149min 开头,含 "SaaS")
- "SaaS" → S/a/a/S 对齐到 11.4s,跟前后中文单调连续 ✅ **英文 token 处理正常**
- RTF 0.204

## 三方头对头(中文 60s,逐字 start 一致度)
| 对 | AAS | p90 | ≤100ms |
|---|---|---|---|
| **MMS vs Qwen** | **58ms** | 100ms | 89% |
| Qwen vs fa-zh | 183ms | 370ms | 36% |
| MMS vs fa-zh | 203ms | 390ms | 29% |

**关键:MMS 跟 Qwen(最高精度)中文几乎完全一致(58ms);fa-zh 是偏松的离群者。→ MMS 中文精度 ≈ Qwen,远好于 fa-zh。**

## 最终选型对比(多语种需求下,fa-zh 已出局)
| | Qwen aligner | MMS-300M CTC-FA | ~~fa-zh~~ |
|---|---|---|---|
| 多语种 | 原生最稳 + 22 方言 | 1126 语言,中英实测 OK | ❌ 仅中文 |
| 中文精度 | 最高(基准) | ≈Qwen(差 58ms) | 偏松(差 200ms) |
| RTF | 0.024(CUDA) | 0.166(Mac CPU,可上 ORT-CUDA) | 0.013 |
| Mac 生产 | **第二个 llama.cpp GGUF** + 自建切块 | 纯 ONNX(契合现有 onnxruntime 栈),内置切块 | — |
| 长音频 | 320s 硬墙必切块 | trellis monolith OOM,分段喂即可 | 同 |
| 集成 | vendor 代码现成,权重已有(813MB,在 3060) | pip 一个包(~300MB ONNX)+ unidecode | — |
| 风险 | 第二 GGUF 进程内 / in-proc pool 内存 | deskpai fork 社区较小;RTF 高于 fa-zh | — |

**判断:多语种 + Mac 主生产下,MMS 是更顺的工程选择 —— 中文精度≈Qwen、多语原生、纯 ONNX 契合现有栈、内置切块免手写、不在进程内压第二个 GGUF。代价是 RTF 0.166(仍 <20%)。Qwen aligner 作为"追极致精度 / MMS 某语种不够"的兜底。**
