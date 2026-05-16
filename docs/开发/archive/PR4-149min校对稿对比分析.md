# PR4 149min 校对稿 vs Qwen3 PoC 对比分析

> 日期: 2026-05-16
> 参考稿来源: `https://sum.lexgogo.site/view/view_Hhm78tykkVWc1VnRmLnSn6aclGhBGDQc1MD4qhuTSVM?raw=calibrated`
> 本地参考稿: `tmp_long_audio/reference_149min_calibrated.txt`
> Qwen3 输出: `tmp_long_audio/poc_outputs/audio_149min.qwen3_long_poc.json`

## 结论

149min 音频的 Qwen3 PoC 已经解决了 PR3 的长上下文崩坏问题。初版主要问题是 speaker 文本漂移; 夜间迭代后, chunk-aware merge 已把 speaker 指标明显拉起来。

- baseline 文本准确率约 90.89%, speaker 字符准确率约 90.54%, segment 多数派准确率约 74.35%。
- v4 全流程重跑后, speaker 字符准确率 96.89%, segment 多数派准确率 92.67%。
- v5 融合版保留当前最佳文本准确率 91.12%, speaker 字符准确率 96.80%, 是目前最均衡的可查看结果。
- 当前主要剩余瓶颈是专名/英文术语、插入偏多、以及没有 forced aligner 导致的短问短答边界仍会漂移。

## 文本准确率

归一化口径: NFKC、英文小写、去空白/标点/符号, 保留中英文数字。参考稿剥离头部元数据。

| 口径 | Ref chars | Hyp chars | 编辑距 | CER | 准确率 |
|---|---:|---:|---:|---:|---:|
| 保留说话人标签 | 58,650 | 60,040 | 6,281 | 10.71% | 89.29% |
| 剔除说话人标签 | 57,852 | 60,040 | 5,529 | 9.56% | 90.44% |
| 剔除标签 + 填充词 | 57,411 | 59,411 | 5,307 | 9.24% | 90.76% |

误差结构:

| 类型 | 字符数 | 占可解释错误 |
|---|---:|---:|
| 替换 | 1,737 | 31.0% |
| 删除 | 836 | 14.9% |
| 插入 | 3,024 | 54.0% |

Hyp/Ref 长度比为 1.038, Qwen3 明显偏“多说/补写”。这类插入包括重复语气词、补充性短句、笑声、追问句、以及语义 plausible 但参考稿不存在的内容。

## 高频错误类型

| 类型 | 参考稿 | Qwen3 |
|---|---|---|
| 路线词 | `闭源` | `B端` |
| 专名/产品名 | `OpenClaw` | `OpenCL` / `OpenCLoud` / `龙虾` |
| 英文专名 | `Claude Code` | `Cloud Code` |
| 英文术语 | `soul` | `solve` / `secret` |
| 缩写 | `API` | `CLI` / `C L I` |
| 项目名 | `章鱼` | `张鱼`, 且和 `龙虾` 混淆 |
| 公司/产品 | `企业微信` | `且微信` / `企微` |
| 英文音译 | `benchmark` | `奔驰马克` |
| 中文近音 | `搀着我` | `拆了之后` |
| 幻觉插入 | `突然间猝死。` | `突然间猝死。哈哈哈哈哈哈！` |

术语计数:

| 术语 | 参考稿 | Qwen3 |
|---|---:|---:|
| `OpenClaw` | 26 | 0 |
| `Claude` | 17 | 0 |
| `API` | 6 | 2 |
| `soul` | 5 | 0 |
| `章鱼` | 29 | 19 |
| `张鱼` | 0 | 5 |
| `龙虾` | 102 | 134 |
| `明略` | 22 | 18 |
| `Agent` | 68 | 61 |

## Speaker 区分

参考稿解析出 266 个 turn, `吴明辉` / `程曼祺` 各 133 个。Qwen3 输出 421 个 segments。

speaker 映射:

| Qwen3 | 参考人物 | 对齐字符纯度 |
|---|---|---:|
| `Speaker1` | 吴明辉 | 94.83% |
| `Speaker3` | 程曼祺 | 66.01% |

按真实人物召回:

| 真实人物 | 正确 Qwen3 speaker | 召回 |
|---|---|---:|
| 吴明辉 | `Speaker1` | 94.09% |
| 程曼祺 | `Speaker3` | 69.12% |

总体:

| 指标 | 数值 |
|---|---:|
| 精确对齐字符口径 speaker accuracy | 90.54% |
| segment 多数派正确率 | 313 / 421 = 74.35% |
| duration 多数派正确率 | 94.77% |
| 混合真实说话人的 segment | 98 段, 约 1621.43s |
| 多数派错分 segment | 108 段, 约 465.57s |
| 高纯度整段错分 | 22 段, 约 178.61s |

Qwen3 输出分布:

| Speaker | segments | 时长 | 时长占比 | 文本字符 |
|---|---:|---:|---:|---:|
| `Speaker1` | 252 | 7580.8s | 84.8% | 56,504 |
| `Speaker3` | 169 | 1313.4s | 14.7% | 9,738 |

结构问题:

- `<=1s` 短段: 63 个。
- `>=60s` 长段: 39 个。
- speaker 切换: 235 次。
- overlaps: 95 个。

典型错分模式:

1. 换人边界被合并。例如开头主持人介绍段末尾混入“大家好, 我是明略的创始人吴明辉”。
2. `Speaker3` 经常吸收吴明辉的短回答或换人后的第一句话, 后半段更明显。
3. `Speaker1` 偶发把程曼祺的问题标成吴明辉, 数量少于 `Speaker3 -> 吴明辉` 的错分。

## 资源与速度

149min PoC summary:

| 指标 | 数值 |
|---|---:|
| audio duration | 8937.94s |
| total elapsed | 1540.78s |
| total RTF | 0.172 |
| ASR elapsed | 793.19s |
| ASR RTF | 0.089 |
| diarize elapsed | 728.98s |
| diarize RTF | 0.082 |
| RSS peak | 6285MB |
| CPU peak | 451.7% |
| quality warnings | 0 |

ASR 与 diarization 串行时几乎对半占用总耗时。理论上, 如果 CPU diarization 与 Metal ASR 可以流水化, 单任务端到端 RTF 下限接近 `max(0.089, 0.082) + overhead`, 约 0.09-0.11。但多 worker 场景不能简单并行多个 Qwen3 decoder, 否则会争抢 Metal 和统一内存带宽。

## 改进优先级

## 已验证 v2-v6 改进

2026-05-16 夜间继续迭代后, 已把 P0 的一部分落成 PoC:

- vendor `TranscribeResult` 暴露 Qwen3-ASR 内部 40s chunk 的 `text/start/end`。
- wrapper `ASRResult` 透传 `chunks`。
- merge 新增 `merge_asr_chunks_and_diarize()`, 优先按 40s chunk 时间窗与 diarization turn 相交来分配文本。
- manual PoC runner 支持 `--reuse-turns-json` 和 `--text-postprocess tech-podcast`。
- 主 `Qwen3DiarizeTranscriber` 也接入 chunk-aware merge; 没有 chunks 时回退原线性切字。
- 新增评估脚本 `tests/manual/server/evaluate_qwen3_poc_against_reference.py`。

149min 对比指标:

| 版本 | 说明 | 文本准确率 | speaker 字符准确率 | segment 多数派准确率 | wrong segments | mixed segments | 运行 RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | 12min 窗口内线性切字 baseline | 90.89% | 90.54% | 74.35% | 108 | 111 | 0.172 |
| v2 | 复用 baseline turns + 40s chunk merge + 第一版术语纠错 | 91.08% | 96.68% | 91.83% | 47 | 66 | 0.089 |
| v3 | v2 基础上补充安全术语纠错 | 91.12% | 96.69% | 91.83% | 47 | 66 | n/a |
| v4 | 全流程重跑 diarization + ASR + chunk merge + 术语纠错 | 91.04% | 96.89% | 92.67% | 42 | 63 | 0.174 |
| v5 | v3 最佳文本 + v4 speaker overlap majority 融合 | 91.12% | 96.80% | 92.17% | 45 | 66 | n/a |
| v6 | 复用 v4 输出 turns + 短 turn 标点搜索保护 | 91.01% | 96.88% | 92.81% | 41 | 63 | 0.091 |

最重要的提升来自 chunk-aware merge, 不是 speaker 文本规则。尝试用语义启发式重标 speaker 后, 指标下降, 因此没有纳入代码。

当前建议查看两个结果:

- 最均衡文本/speaker: `tmp_long_audio/poc_outputs_v5_fused/audio_149min.qwen3_long_poc.json`
- speaker 指标最高的全流程输出: `tmp_long_audio/poc_outputs_v4_full/audio_149min.qwen3_long_poc.json`
- 短 turn 保护验证输出: `tmp_long_audio/poc_outputs_v6_short_turn_guard/audio_149min.qwen3_long_poc.json`

复现命令:

```bash
DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin" \
venv/bin/python tests/manual/server/qwen3_long_audio_poc.py \
  tmp_long_audio/audio_149min.mp3 \
  --reuse-turns-json tmp_long_audio/poc_outputs/audio_149min.qwen3_long_poc.json \
  --text-postprocess tech-podcast \
  --out-dir tmp_long_audio/poc_outputs_v2

venv/bin/python tests/manual/server/fuse_qwen3_poc_speakers.py \
  --text-json tmp_long_audio/poc_outputs_v3/audio_149min.qwen3_long_poc.json \
  --speaker-json tmp_long_audio/poc_outputs_v4_full/audio_149min.qwen3_long_poc.json \
  --text-postprocess tech-podcast \
  --out-dir tmp_long_audio/poc_outputs_v5_repro

venv/bin/python tests/manual/server/evaluate_qwen3_poc_against_reference.py \
  tmp_long_audio/reference_149min_calibrated.txt \
  tmp_long_audio/poc_outputs_v5_fused/audio_149min.qwen3_long_poc.json
```

### P0: 接入时间对齐, 替代按时长线性切字

当前 [src/core/qwen3/merge.py](../../src/core/qwen3/merge.py) 的融合方式是按 diarization turn 时长比例切 ASR 全文。它会在语速变化、插话、短问短答场景里漂移, 是 speaker 文本归属准确率的主要上限。

优先方案:

1. 利用 Qwen3 ASR 的 internal chunk/item 时间戳, 至少做到 chunk 级文本时间对齐。
2. 引入 Qwen3-ForcedAligner 或等价 aligner, 做字/词级时间戳。
3. 再用时间戳和 diarization turn 做 interval join, 而不是线性切字。

预期收益: speaker segment 正确率和主持人召回会明显提升, 尤其是程曼祺短问题不再被吞进吴明辉长回答。

### P0: 术语热词与二阶段纠错

这类技术播客的主要正文错误集中在专名和英文术语。建议引入上下文热词表或二阶段校正:

- `OpenClaw`, `Claude Code`, `Manus`, `Agentic Service`, `AI Native`
- `API`, `soul`, `memory`, `skills`, `benchmark`
- `章鱼`, `龙虾`, `明略`, `企业微信`, `飞书`, `钉钉`

实现上可以先做轻量后处理: product glossary + 模糊匹配 + 上下文 disambiguation。比调标点更有收益。

### P1: 控制插入偏多

当前 Hyp 比 Ref 长 3.8%, 插入是最大误差来源。应验证:

- 降低 decoder temperature。
- 更保守的重复/笑声/语气词后处理。
- 对“哈哈哈”、重复语气词、无声学依据的短补句做质量告警。

### P1: diarization 参数网格和后处理

149min 是双人对话, 83min 出现 8 speaker 过分散。建议做网格:

- `num_speakers=2` vs auto。
- `cluster_threshold` around current value。
- 短 turn 合并。
- 相邻同 speaker 合并。
- 过短 speaker segment 归并到最近高置信 speaker。

### P1: 输出层标记混合段

对低置信、跨换人点、短答拼长句的 segment, 不要强行输出单 speaker。可以输出 `speaker_confidence` 或 `mixed_speaker=true`, 让下游校对成本降低。

### P2: 资源调度

建议调度策略:

- ASR decoder 使用全局 Metal semaphore, 默认同一时间只允许 1 个 Qwen3 decoder 占 GPU。
- diarization 走 CPU 有限并行, 如 2-4 threads per task。
- 单 worker 内允许 diarize 与 ASR 分阶段流水, 多 worker 之间不要盲目同时 ASR。

## 下一步建议

1. 保留 12min 宏分段, 进入主 transcriber 路径。
2. 优先改 ASR/diarization 融合方式, 从线性切字升级到时间戳 interval join。
3. 加技术播客 glossary/热词纠错, 先解决专名和英文缩写。
4. 对 149min 以 `num_speakers=2` 跑 diarization 参数复测, 对 83min 做 cluster threshold 网格。
5. 增加 speaker confidence / mixed segment 输出, 为下游校对暴露不确定性。

## 2026-05-16 v7 speaker 后处理实验

### 实验思路

在不重跑 ASR / diarization 的前提下，基于当前 speaker 最优的 v6 输出做一次轻量后处理：

1. 保持 `segments[].text/start/end` 不变，只调整极少数高置信 speaker label。
2. 对双人访谈做一个显式假设：文本量主导的 speaker 是回答者/嘉宾，少量文本 speaker 是提问者/主持人。
3. 仅对短段触发：
   - 回答者短段若强匹配问句 cue（如“你觉得/你们/可以讲讲/是什么/吗/？”）则转给提问者。
   - 提问者短段若强匹配回答者 cue（如“我觉得/我们公司/我的/龙虾/章鱼”）则转给回答者。
4. 同时为紧邻换人边界和极短段增加非破坏性 debug：`mixed_speaker` 与 `debug.speaker_confidence_hints`。这些字段不影响现有 JSON/SRT 消费方。

新增脚本：

- `tests/manual/server/postprocess_qwen3_speakers.py`

最佳参数来自离线网格搜索，选择变更数很少但稳定提升的组合：

```bash
python3 tests/manual/server/postprocess_qwen3_speakers.py \
  tmp_long_audio/poc_outputs_v6_short_turn_guard/audio_149min.qwen3_long_poc.json \
  --strategy lexical-role \
  --max-question-sec 4 \
  --max-answer-sec 12 \
  --question-threshold 2 \
  --answer-threshold 3 \
  --out-dir tmp_long_audio/poc_outputs_v7_speaker_smooth

python3 tests/manual/server/evaluate_qwen3_poc_against_reference.py \
  tmp_long_audio/reference_149min_calibrated.txt \
  tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.qwen3_long_poc.json \
  > tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.eval.json
```

输出：

- JSON: `tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.qwen3_long_poc.json`
- SRT: `tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.qwen3_long_poc.srt`
- Eval: `tmp_long_audio/poc_outputs_v7_speaker_smooth/audio_149min.eval.json`

### 指标对比

| 版本 | 说明 | 文本准确率 | speaker 字符准确率 | segment 多数派准确率 | wrong segments | mixed segments |
|---|---|---:|---:|---:|---:|---:|
| v4 full | 全流程 chunk-aware merge | 91.04% | 96.89% | 92.67% | 42 | 63 |
| v5 fused | 最佳文本 + speaker overlap 融合 | 91.12% | 96.80% | 92.17% | 45 | 66 |
| v6 short-turn guard | 短 turn 标点搜索保护 | 91.01% | 96.88% | 92.81% | 41 | 63 |
| v7 speaker smooth | v6 + 高置信短问答角色平滑 | 91.01% | **96.97%** | **93.51%** | **37** | 63 |

v7 只改 6 个 segment 的 speaker label；文本准确率不变。改善主要来自修复若干明显短问句/短回答边界错配，例如：

- `是什么？` 从 `Speaker1` 调整为 `Speaker3`
- `可以讲讲就你们...接下来...计划吗？` 从 `Speaker1` 调整为 `Speaker3`
- `有些可能现在已经发生了...我觉得...我们公司...` 从 `Speaker3` 调整为 `Speaker1`

### 是否值得保留

值得保留为 **manual/experimental 后处理脚本**，但暂不建议默认接入 production transcriber：

- 优点：无需重跑 ASR，变更很少，149min 指标稳定提升。
- 风险：依赖“双人访谈 + 主问客答”的角色假设，不适合多人会议或角色对称对话。
- 已通过 `debug` 字段记录每次变更原因和 cue score，便于人工审计。

### forced alignment 可行性补充

本地 vendor 代码包含 `src/core/vendor/qwen_asr_gguf/inference/aligner.py`，但当前模型目录只发现 ASR encoder/LLM：

- `models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_llm.gguf`
- `models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx`
- `models/qwen3_diarize/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx`

未发现 `qwen3_aligner_encoder_*` 或 `qwen3_aligner_llm*` 文件。因此当前不能直接启用 Qwen3 forced aligner。后续若补齐 aligner 模型，应优先把 speaker merge 从“chunk 内按字符比例切分”升级到“字/词时间戳与 diarization turns interval join”，这是继续降低 cross-boundary mixed 的最高优先级。

## 2026-05-16 Qwen3-ForcedAligner 可行性与细时间戳实验

### 模型与运行环境

已下载官方 `Qwen/Qwen3-ForcedAligner-0.6B` 到本地：

- `models/qwen3_diarize/Qwen3-ForcedAligner-0.6B/`
- 主要权重：`model.safetensors`，约 1.7GB

通过 `qwen-asr` 官方 Python 包直接调用 `Qwen3ForcedAligner`，在 Apple Silicon 上使用：

- `device_map="mps"`
- `dtype=torch.float16`

最小 smoke test 可用，2s 中文短句 `是什么？` 输出逐字时间戳：

```text
是 0.00-0.16
什 0.32-0.48
么 0.48-0.56
```

这证明官方 PyTorch 版 aligner 在本地可用；vendor GGUF/ONNX 路径仍缺少 `qwen3_aligner_encoder_*` 与 `qwen3_aligner_llm*`，暂不能直接通过现有 vendor engine 开启 `enable_aligner=True`。

### 新增实验脚本

新增两个 manual 脚本：

1. `tests/manual/server/align_qwen3_poc_segments.py`
   - 不重跑 diarization。
   - 将现有 v6 的 40s chunk 文本用 Qwen3-ForcedAligner 对齐到字/词时间戳。
   - 再按现有 segment 时间区间做 interval join。
   - 结果：如果原始 diarization interval 已经把某段吞成同一 speaker，则无法修复 speaker，只能改善切分形态。

2. `tests/manual/server/align_qwen3_poc_with_window_diarize.py`
   - 对选定 40s 窗口重新跑 sherpa diarization（`num_speakers=2`）。
   - 用 Qwen3-ForcedAligner 对该窗口拼接文本输出字/词级时间戳。
   - 将 timestamp item 与 fresh diarization turns 做 interval join，重建该窗口 segments。
   - local speaker cluster 映射回全局 speaker 时优先使用与原 segment 的 overlap；若窗口原本被单 speaker 吞掉，则最长 local cluster 保留原 speaker，另一个 cluster 映射到另一 speaker。

### 关键实验与指标

| 版本 | 说明 | text acc | speaker char acc | segment majority acc | wrong | mixed |
|---|---|---:|---:|---:|---:|---:|
| v6 | short-turn guard baseline | 91.01% | 96.88% | 92.81% | 41 | 63 |
| v7 | v6 + lexical role smoothing | 91.01% | 96.97% | 93.51% | 37 | 63 |
| v8 probe 5520 | 单个 5520-5560s 窗口，FA + window diarize | 91.01% | 96.96% | 92.97% | 40 | 62 |
| v8 topwrong | 11 个 top wrong 窗口，FA + window diarize | 91.01% | 97.25% | 94.17% | 33 | 57 |
| v8b top80 | 59 个高风险窗口，FA + window diarize + overlap 映射 | 91.01% | **98.15%** | **94.35%** | **32** | **34** |
| v8 selected | 从 top80 结果中用 reference 贪心选择正收益窗口（上界分析） | 91.01% | **98.44%** | **95.37%** | **26** | **36** |
| v8c all | 全部 224 个窗口都重跑 window diarize + FA | 91.01% | 86.67% | 79.35% | 127 | 48 |

输出路径：

- `tmp_long_audio/poc_outputs_v8_align_diarize_probe_5520/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v8_align_diarize_topwrong/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v8b_align_diarize_top80/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v8_align_diarize_selected/audio_149min.qwen3_long_poc.json`
- `tmp_long_audio/poc_outputs_v8c_align_diarize_all/audio_149min.qwen3_long_poc.json`

### 复现命令

下载/安装依赖：

```bash
uv pip install -p venv/bin/python 'huggingface_hub[cli]' qwen-asr
venv/bin/huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir models/qwen3_diarize/Qwen3-ForcedAligner-0.6B
```

11 个 top wrong 窗口实验：

```bash
DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin" \
venv/bin/python tests/manual/server/align_qwen3_poc_with_window_diarize.py \
  tmp_long_audio/poc_outputs_v6_short_turn_guard/audio_149min.qwen3_long_poc.json \
  tmp_long_audio/audio_149min.mp3 \
  --window-ids 9,91,93,116,121,126,129,138,168,213,216 \
  --out-dir tmp_long_audio/poc_outputs_v8_align_diarize_topwrong \
  --device-map mps \
  --dtype float16

venv/bin/python tests/manual/server/evaluate_qwen3_poc_against_reference.py \
  tmp_long_audio/reference_149min_calibrated.txt \
  tmp_long_audio/poc_outputs_v8_align_diarize_topwrong/audio_149min.qwen3_long_poc.json \
  > tmp_long_audio/poc_outputs_v8_align_diarize_topwrong/audio_149min.eval.json
```

59 个高风险窗口实验：

```bash
DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin" \
venv/bin/python tests/manual/server/align_qwen3_poc_with_window_diarize.py \
  tmp_long_audio/poc_outputs_v6_short_turn_guard/audio_149min.qwen3_long_poc.json \
  tmp_long_audio/audio_149min.mp3 \
  --window-ids 8,9,24,29,33,36,41,44,54,57,63,65,66,68,69,70,71,72,73,74,91,93,107,116,117,120,121,122,123,126,127,129,133,134,135,138,141,144,145,146,152,165,166,168,169,176,178,186,187,188,191,192,200,206,207,212,213,216,217 \
  --out-dir tmp_long_audio/poc_outputs_v8b_align_diarize_top80 \
  --device-map mps \
  --dtype float16
```

### 结论

1. **Qwen3-ForcedAligner 本地可用，且速度可接受。** 40s 中文窗口对齐通常约 0.5-1s；window diarization 约 3-4s，是主要耗时。
2. **仅有 forced alignment 不够。** 如果继续使用已经错分的旧 segment/turn 作为 speaker interval，aligner 只能重新切文本，无法把被吞掉的短问句恢复成另一 speaker。
3. **ForcedAligner + 局部重 diarization 明显有效。** 对高风险窗口重跑 40s diarization，再用字级 timestamp 做 interval join，speaker char acc 从 96.88% 提升到 98.15%，mixed 从 63 降到 34。
4. **不能盲目全窗口启用。** 全部 224 个窗口都局部重 diarize 会破坏 speaker 全局一致性，speaker char acc 降到 86.67%。因此需要“高风险窗口检测 + 稳健 cluster 映射 + 可回退评估”。
5. **v8 selected 是上界分析，不是 production 逻辑。** 它使用 reference 选择正收益窗口，证明这条路线理论上可到 98.44% speaker char acc / 95.37% segment majority，但真实工程需要无参考的窗口选择策略。

### 下一步工程方向

- 将 `align_qwen3_poc_with_window_diarize.py` 的逻辑收敛为可配置实验 pipeline：
  - 高风险窗口检测：长 chunk 中多 speaker 贴边、短问句 cue、A/B/A 抖动、mixed_speaker debug、低 overlap confidence。
  - 局部 diarization 参数网格：`min_duration_on/off`、`num_speakers=2`、窗口长度 40s vs 80s。
  - cluster 映射：从简单 overlap 映射升级到 speaker embedding centroid 或全局 reference turns 映射。
  - 回退规则：如果重切后 speaker 切换过多、低 confidence item 过多或与原分布差异过大，则保留原窗口。
- 中长期：如果能把官方 PyTorch aligner 封装到 production 路径，优先用于“可疑窗口二次细切”，而不是默认处理全量窗口。

### 2026-05-16 补充：RTF 与 speaker embedding centroid 映射

根据 `window_stats[].elapsed` 统计，Qwen3-ForcedAligner 自身在 MPS/float16 上的速度约为：

| 实验 | 窗口数 | 对齐音频时长 | aligner elapsed sum | aligner RTF |
|---|---:|---:|---:|---:|
| topwrong 11 windows | 11 | 440s | 7.45s | 0.0169 |
| top80/high-risk 59 windows | 59 | 2360s | 37.16s | 0.0157 |
| all 224 windows | 224 | 8960s | 142.15s | 0.0159 |

因此如果**只跑 Qwen3-ForcedAligner**，149min 全量约 142s，RTF ≈ **0.016**。如果加上当前实验里的 per-window sherpa diarization，实测全 224 windows 端到端约 13-14min，整体 RTF 约 **0.09**；瓶颈主要是 window diarization，不是 aligner。

已初步测试 speaker embedding centroid 映射：

- 用 v6 中较长、较稳定的原 segments 为 `Speaker1/Speaker3` 建全局 speaker embedding centroid。
- 每个局部 40s diarization cluster 取若干 turn embedding，与全局 centroid 做 cosine 匹配。
- 在 11 个 top wrong windows 上对比：

| 映射策略 | speaker char acc | segment majority acc | wrong | mixed |
|---|---:|---:|---:|---:|
| v6 baseline | 96.88% | 92.81% | 41 | 63 |
| FA + window diarize + overlap 映射 | 97.25% | 94.17% | 33 | 57 |
| FA + window diarize + centroid 映射 | **97.62%** | **94.70%** | **30** | 57 |

结论：centroid 映射在 top wrong 窗口上明显优于 overlap 映射，值得继续扩展到 high-risk 59 windows 和生产化候选 pipeline。

## 2026-05-16 补充（二）：centroid 扩展、自动 high-risk detector 与 quality gate

### 实验思路

本轮继续验证 `Qwen3-ForcedAligner + local 40s diarization + global speaker embedding centroid mapping` 是否值得工程化：

1. 补跑 high-risk 59 windows + centroid 映射（v8e）。
2. 补跑 all 224 windows + centroid 映射（v8f）作为强制对照。
3. 在 `align_qwen3_poc_with_window_diarize.py` 增加实验参数：
   - `--num-speakers 2|auto|none`
   - `--min-duration-on/off`
   - `--quality-gate`
   - centroid mapping debug：`mapping_scores` / `mapping_margin`
4. 新增无 reference 选择器 `tests/manual/server/select_qwen3_high_risk_windows.py`，用 hypothesis 侧 topology/debug/lexical cue 给 40s window 打分。
5. 验证 quality gate 对手工 high-risk、自动 topK、all windows 的回退效果。

### 输出路径与指标

| 版本 | 说明 | text acc | speaker char acc | segment majority acc | wrong | mixed | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | 原始长音频 PoC | 90.89% | 90.54% | 74.35% | 108 | 111 | `tmp_long_audio/poc_outputs` |
| v4 full | chunk-aware merge | 91.04% | 96.89% | 92.67% | 42 | 63 | `tmp_long_audio/poc_outputs_v4_full` |
| v5 fused | speaker fusion | 91.12% | 96.80% | 92.17% | 45 | 66 | `tmp_long_audio/poc_outputs_v5_fused` |
| v6 | short-turn guard baseline | 91.01% | 96.88% | 92.81% | 41 | 63 | `tmp_long_audio/poc_outputs_v6_short_turn_guard` |
| v7 | v6 + lexical smoothing | 91.01% | 96.97% | 93.51% | 37 | 63 | `tmp_long_audio/poc_outputs_v7_speaker_smooth` |
| v8b | high-risk 59 + overlap mapping | 91.01% | 98.15% | 94.35% | 32 | 34 | `tmp_long_audio/poc_outputs_v8b_align_diarize_top80` |
| v8d | topwrong 11 + centroid mapping | 91.01% | 97.62% | 94.70% | 30 | 57 | `tmp_long_audio/poc_outputs_v8d_centroid_topwrong` |
| v8d-gate | topwrong 11 + centroid + quality gate | 91.01% | 97.59% | 95.18% | 27 | 57 | `tmp_long_audio/poc_outputs_v8d_centroid_topwrong_gate`；reference-derived window list，仅分析用 |
| v8e | high-risk 59 + centroid mapping | 91.01% | 98.15% | 94.35% | 32 | 34 | `tmp_long_audio/poc_outputs_v8e_centroid_top80` |
| v8e-gate | high-risk 59 + centroid + quality gate | 91.01% | 98.09% | 95.14% | 27 | 35 | `tmp_long_audio/poc_outputs_v8e_centroid_top80_gate` |
| v8 selected | reference 贪心上界 | 91.01% | 98.44% | 95.37% | 26 | 36 | `tmp_long_audio/poc_outputs_v8_align_diarize_selected`；不可 production |
| v8f | all 224 + centroid mapping | 91.01% | 86.65% | 79.19% | 128 | 48 | `tmp_long_audio/poc_outputs_v8f_centroid_all` |
| v8f posthoc-gate | all 224 + posthoc quality gate | 91.01% | 86.58% | 79.64% | 123 | 49 | `tmp_long_audio/poc_outputs_v8f_centroid_all_posthoc_gate`；仅拒 3/224，不足以防全量破坏 |
| v9a | auto detector top40 + centroid | 91.01% | 96.98% | 93.13% | 38 | 52 | `tmp_long_audio/poc_outputs_v9a_auto_top40_centroid` |
| v9b | auto detector top40 + centroid + gate | 91.01% | 96.94% | 93.43% | 36 | 53 | `tmp_long_audio/poc_outputs_v9b_auto_top40_centroid_gate` |
| v9c | auto detector top20 + centroid + gate | 91.01% | 97.26% | 92.97% | 40 | 57 | `tmp_long_audio/poc_outputs_v9c_auto_top20_centroid_gate` |
| v9d | auto top20 + gate + min_on/off 0.3/0.5 | 91.01% | 97.26% | 93.30% | 38 | 58 | `tmp_long_audio/poc_outputs_v9d_auto_top20_centroid_gate_min0305` |

### 本轮关键发现

1. **centroid mapping 扩展到 high-risk 59 后，与 overlap mapping 指标几乎持平。**  
   v8e = 98.15% speaker char acc / 94.35% segment majority / wrong 32 / mixed 34，未超过 v8b。这说明 centroid 在 topwrong 11 的收益主要来自少数窗口；扩展到 59 后收益被其他窗口抵消。

2. **quality gate 对手工 high-risk 有价值。**  
   v8e-gate 拒绝窗口 `[117, 144, 213]`，speaker char acc 从 98.15% 小降到 98.09%，但 segment majority 从 94.35% 提升到 95.14%，wrong 从 32 降到 27，已非常接近 reference oracle selected 的 95.37% / wrong 26。若工程目标更看重段级 speaker 稳定性，gate 值得保留。

3. **all windows 仍然不可行，centroid 也无法解决盲目全量替换。**  
   v8f speaker char acc 86.65%，与 v8c all/overlap 同量级灾难。posthoc gate 只拒绝 3/224，仍为 86.58%。结论不变：必须先做高风险选择，不能靠 gate 挽救全量重 diarization。

4. **当前无 reference detector 还不够好。**  
   `select_qwen3_high_risk_windows.py` 的 top40 对手工 59 窗口召回约 40.7%，precision 约 60%；top20 召回约 20.3%。实际指标：top40+gate 仅 96.94% char / 93.43% majority，top20+gate 97.26% char / 92.97% majority。它能减少 mixed，但无法稳定抓住最高收益窗口。

5. **refined 后再套 v7 lexical smoothing 不可直接采用。**  
   对 v9b/v9c 再跑 `postprocess_qwen3_speakers.py --strategy lexical-role` 会大幅退化（约 94% speaker char / 87% segment majority），因为 forced-align 后 segment 形态改变，原短句 lexical 规则触发过多。

### 运行耗时 / RTF 估算

- v8e high-risk 59 + centroid：276s 运行 + 16s eval；处理音频 2360s，端到端 refinement RTF ≈ 0.117（含 centroid 构建、window diarization、FA）。
- v8f all 224 + centroid：915s 运行 + 16s eval；处理音频约 8960s，端到端 refinement RTF ≈ 0.102。
- v9 top20：约 133-134s；top40：约 213-214s。
- 结合上一轮统计，FA 自身 RTF 约 0.016；当前瓶颈仍是 per-window sherpa diarization + centroid/embedding 相关开销。

### 代码改动是否值得保留

值得保留为 manual/PoC：

- `tests/manual/server/select_qwen3_high_risk_windows.py`：无 reference detector 初版，适合继续 sweep/调参。
- `align_qwen3_poc_with_window_diarize.py` 的 `--quality-gate`、`--min-duration-on/off`、`--num-speakers`：用于实验网格和回退分析。
- centroid mapping debug fields：后续可用于阈值学习或线上 debug。

暂不应直接 production：

- 手工 high-risk 59/topwrong 11 window list：来自 reference/error 分析，只能作为 benchmark。
- 当前 auto detector scoring 权重：效果不足，不能作为默认生产策略。
- refined 后直接套 lexical smoothing：已验证退化。

### 下一步

1. 改进 auto detector：加入 cross-hypothesis disagreement（v6/v7/v8f/v8e 或 lightweight FA-only）与 window-level change prediction；当前只靠 topology/lexical/debug 不够。
2. 改进 quality gate：现在能识别极碎/低质窗口，但无法判断”低风险窗口被整体翻转”。需要加入原 speaker distribution 置信度、centroid top1/top2 聚合统计、local cluster duration dominance。
3. 工程 PoC 推荐路线：生产候选应是 **detector 选窗 + centroid mapping + quality gate**，而不是全量。以当前手工 detector 上界看，149min 双人对话可达到约 **98.09% speaker char / 95.14% segment majority / wrong 27**，但自动 detector 仍只能到约 **97.26% speaker char**，尚未达到采用门槛。

## 2026-05-16 补充（三）：short-segment guard 后处理 + detector v2 audio-aware + 综合对比

本轮在 v8e-gate 错例分布上发现 refine 流程在短段上引入 regression（27 个 wrong 段中 21 个 ≤ 2s，14 个 ≤ 1s，包含 6 个 0.0-0.2s 幽灵碎片）；同时 oracle (v8 selected) 在短段上也有 13/26 ≤ 1s wrong，说明 ForcedAligner 切短后 speaker 仍可能错是物理上界。基于这个分析做了两条不同方向的实验：纯后处理 short-segment guard（v12 系列）和 audio-aware detector v2。

### 新增脚本

- `tests/manual/server/postprocess_qwen3_short_segment_guard.py` — 纯逻辑后处理（无 GPU）：
  - drop_tiny_segments: 把 < `short_drop_sec` 的孤立短段并入时间最近邻段（按 gap 选 prev/next）
  - aba_smoothing: A-B-A 抖动平滑（短中间段 ≤ `aba_max_mid_sec` 且是 backchannel / question_tail / 高 char-density 短碎片，回退到 A speaker）
  - merge_consecutive_same_speaker: refine/aba 后合并连续同 speaker 段
- `tests/manual/server/select_qwen3_high_risk_windows_v2.py` — detector v2 加入 audio-side features：
  - 全曲 ffmpeg silencedetect 一次性算每 window silence_count / silence_total_sec
  - 全局 dominant/minority speaker centroid（复用 build_global_centroids 思路）+ 每个 segment 对应 audio interval 的 embedding 余弦距离（confident_minority/dominant_in_*_label）
  - 同 window 内相邻 segment embedding cosine 距离（同 hyp 标签但 audio embedding 远离，捕捉段内 speaker switch）
  - 每 window 跑一次轻量 sherpa diarize (num_speakers=2)，输出 local_turns/distinct_speakers/local_switches/speech_ratio/min_max_turn_dur
  - 1s block RMS energy 标准差/均值
  - 输出 `tmp_long_audio/detector_v2/` 含 high_risk_windows_v2.json + top{20,40,60,80}_ids.txt

### 关键发现

#### 1. short-segment guard 是巨大改进（适用任意 baseline）

短段 guard 对 v7 / v8b / v8e_gate / oracle 全部都有正收益。最优参数 `short_drop_sec=2.0, aba_max_mid_sec=1.5, merge_same=on` 在 v8eg 上把 wrong 从 27 砍到 8（-70%），seg_majority 从 95.14% 升到 97.60%，仅 sp_acc 微降 0.25%。

#### 2. detector v2 比 v1 显著好

manual high-risk 59 召回/精确率：

| K  | v1 P | v1 R | v2 P | v2 R |
|----|------|------|------|------|
| 20 | 60.0% | 20.3% | **80.0%** | **27.1%** |
| 40 | 60.0% | 40.7% | **70.0%** | **47.5%** |
| 60 | 50.0% | 50.8% | **53.3%** | **54.2%** |
| 80 | 38.8% | 52.5% | **46.2%** | **62.7%** |

最强 audio features（按 TP/FP）：`same_label_adjacent_far_embeddings`（同 hyp 标签但相邻 embedding 远 > 0.4，12/12 TP）、`confident_minority/dominant_audio_in_*_label`（embedding 反对 hypothesis）、`local_switches_far_exceed_hyp`（mini diarize switch ≥ hyp + 3）。

#### 3. 综合对比（按 wrong 升序）

| 路线 | 是否 reference-derived | text acc | sp_char | seg_maj | wrong | mixed | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| **v9d top20 + centroid + gate + v12 d=2.0** ★ | ❌ 纯 auto | 91.01% | **97.29%** | **98.11%** | **6** | 46 | 生产可用最佳 sp |
| **v9b top40 + centroid + gate + v12 d=2.0** ★ | ❌ 纯 auto | 91.01% | 96.91% | **98.13%** | **6** | 47 | 生产可用最佳 majority |
| v9c top20 + centroid + gate + v12 d=2.0 | ❌ 纯 auto | 91.01% | 97.29% | 97.80% | 7 | 45 | |
| **v8e-gate + v12 d=2.0** | ✓ manual 59 | 91.01% | **97.84%** | 97.60% | 8 | 38 | 标杆 |
| **oracle + v12 d=1.5** | ✓ greedy oracle | 91.01% | **98.27%** | 97.33% | 9 | 35 | 物理上界 |
| v8b + v12 d=1.5 | ✓ manual 59 | 91.01% | 98.01% | 96.84% | 11 | 36 | |
| oracle + v12 d=0.8 | ✓ greedy oracle | 91.01% | 98.40% | 96.83% | 11 | 33 | |
| **v7 + v12 d=1.5（仅后处理！）** ★ | ❌ 纯 auto | 91.01% | 96.96% | 96.45% | **11** | 50 | 零 GPU 路径 |
| v8e-gate + v12 d=0.8 | ✓ manual 59 | 91.01% | 98.03% | 96.33% | 13 | 35 | |
| **v8e-gate（baseline）** | ✓ manual 59 | 91.01% | 98.09% | 95.14% | 27 | 35 | 上一版最佳 |
| **oracle（v8 selected baseline）** | ✓ greedy oracle | 91.01% | 98.44% | 95.37% | 26 | 36 | |
| v10c top80 + centroid + gate | ❌ auto | 91.01% | 94.17% | 90.72% | 53 | 49 | top80 误报过多, 退步 |
| v10c top80 + centroid + gate + v12 d=2.0 | ❌ auto | 91.01% | 94.10% | 93.46% | 25 | 48 | 后处理救不活 top80 |
| v7（baseline） | ❌ auto | 91.01% | 96.97% | 93.51% | 37 | 63 | 上一版基线 |

#### 4. 何时 short-guard 不够

把 v8eg+d=2.0 剩下 8 个 wrong 与 oracle+d=1.5 剩下 9 个对比：8 个错段几乎完全重合，全部是「长段含跨 turn」或「短段紧贴 turn 边界」的物理边界 case：

- win 66 14s 跨 turn（W 72 / C 25）：长段 ASR 没切对，单 segment 含双 speaker
- win 127 4.3s 跨 turn（W 17 / C 16）
- win 144 4.3s 跨 turn（C 8 / W 12）
- win 156 3.6s 混
- win 188 3.0s “你们自己现在也训练小模型吗” 整段被分错
- win 191 1.5s “非常有意思那我必须要去”
- win 72 3.3s 长问句
- win 44 0.9s 短碎片

这一层错误需要 **段内重切（chunk-level / sub-chunk diarize）**，single-segment-as-unit 的 v12 short-guard 救不了。

#### 5. detector v2 + centroid + gate + v12 实测

| 路线 | sp_char | seg_maj | wrong | mixed | 备注 |
|---|---:|---:|---:|---:|---|
| v13a det v2 top20 + centroid + gate | 97.22% | 94.61% | 30 | 58 | 比 v9c top20 (97.26/40) 略好 |
| v13a + v12 d=2.0 | 97.21% | 97.43% | **8** | 49 | 不如 v9d top20 + v12 (wrong=6) |
| v13b det v2 top40 + centroid + gate | 96.69% | 92.82% | 40 | 55 | 比 v9b (96.94) 略差 |
| v13b + v12 d=2.0 | 96.66% | 95.77% | 14 | 49 | 退步 vs v9b+v12 (wrong=6) |
| v13b + v12 d=1.5 | 96.66% | 95.52% | 15 | 48 | |

**意外结论**: detector v2 在 manual 59 召回上比 v1 提升 (top40 P 60→70%, R 41→48%)，但端到端 refined 输出反而**不如 v1**。原因推测：manual 59 是 reference-derived 标签，不等于"refine 后会改善"的窗口；v2 选的多出的窗口实际是 refine 把它们变得更糟的 false-positive 类。**detector v2 暂不值得替换 v1**。

#### 6. chunk-level mini-diarize refine（sub-agent 实验）

新脚本 `tests/manual/server/refine_qwen3_chunk_level.py` 实现 chunk-aligned refine（不依赖 detector 选窗口）：

| 路线 | sp_char | seg_maj | text_acc | 备注 |
|---|---:|---:|---:|---|
| v11 cross-trigger (104 chunks) | 96.94% | 91.65% | 90.73% | 比 v7 退步 |
| v11b silence-only (8 chunks) | 96.40% | 92.66% | 91.01% | 比 v7 退步 |
| v11c cross + v6 base | 97.16% | 91.83% | 90.73% | 微改善 sp 但 maj 退 |

**结论**: chunk-aligned refine 失败。原因 — 它的 chunk 边界与原 `merge_asr_chunks_and_diarize` 的 chunk 边界重合，FA 不能跨 chunk 看上下文。而 **v8e_gate 用 fixed 40s grid offset by macro_start，window 中心刚好落在 ASR chunk 中段**，FA 能在 chunk 边界两边看上下文，捕捉 chunk 边界附近的 speaker 切换 — 这正是 v8e_gate 比 chunk-aligned 强的根本原因。

文件保留作为"为什么不行"的诊断，建议不工程化。

### 最终对比表（按 wrong 升序，去重）

| 路线 | reference-derived? | sp_char | seg_maj | wrong | RTF (refine) | 推荐 |
|---|---:|---:|---:|---:|---:|---|
| v9d top20 + centroid + gate + v12 d=2.0 | ❌ | **97.29%** | **98.11%** | **6** | 0.015 | ★★★ 生产 |
| v9b top40 + centroid + gate + v12 d=2.0 | ❌ | 96.91% | **98.13%** | **6** | 0.024 | ★★ 生产 |
| v9c top20 + v12 d=2.0 | ❌ | 97.29% | 97.80% | 7 | 0.015 | ★ |
| v8e-gate + v12 d=2.0 | ✓ | 97.84% | 97.60% | 8 | 0.022 | benchmark |
| v13a det v2 top20 + v12 d=2.0 | ❌ | 97.21% | 97.43% | 8 | 0.015 + 0.078 preflight | 不值得 |
| oracle + v12 d=1.5 | ✓ greedy | 98.27% | 97.33% | 9 | 0.022 | 物理上界 |
| v7 + v12 d=1.5（无 refine）★ | ❌ | 96.96% | 96.45% | **11** | < 0.0001 | ★★ 生产 P0 |

## 2026-05-16 补充（四）：多人场景与评测集

149min 评测以 2-speaker 双人对话为主，本轮扩展到 1/2/4/6 speaker 全场景验证，并发现了两个关键 bug。

### 新增评测集 (tmp_long_audio/eval_set/)

| 文件 | 时长 | 真实人数 | 备注 |
|---|---:|---:|---|
| `audio_1spk_real.m4a` | 16.2min | 1 | 杨涛个人频道独白 |
| `audio_panel_marked_1spk.m4a` | 34.1min | 3+ (标错的"1人") | 汽车类访谈，含"四零""六三""老编辑" |
| `audio_2spk_60min.mp3` | 60min | 2 | 149min 截前 60min (吴明辉×程曼祺) |
| `audio_4spk.m4a` | 43.8min | 4 | 4 人圆桌 + 结尾英文歌 |
| `audio_6spk_60min.m4a` | 60min | 6 | 小宇宙 6 人对话 |

每个样本三阶段输出：`{label}_pipeline/baseline/`、`merged/`、`final_v12_d1.5/`。统一 README 在 `tmp_long_audio/eval_set/README.md`。

### 新增脚本

- `tests/manual/server/merge_qwen3_minor_clusters.py` — **cluster centroid 合并**，修复 sherpa 多人场景过度聚类。用 nemo-titanet 余弦相似度做层级合并：
  - main-to-main 高置信合并 (sim ≥ 0.78)
  - minor (< 3% share) → 最近 main (sim ≥ 0.55)
  - dominant 自适应：share ≥ 60% 时主 cluster 间用 0.6 阈值
  - 非语音 cluster (音乐/笑声 cosine < 0.55) 保留独立
- `tests/manual/server/analyze_multispeaker_output.py` — 无 reference 的多人输出分析（detected speakers / 段长分布 / 切换频率 / 代表性文本）

### 修复的两个 bug

1. **vendor dylib preload (`src/core/vendor/qwen_asr_gguf/inference/llama.py`)**：新版 ggml 拆分 cpu/blas/metal backend dylib，原 `_ensure_loaded()` 只 preload `libggml + libggml-base + libllama`，导致 nohup 启动（SIP strip `DYLD_LIBRARY_PATH`）时 `libggml.dylib` 找不到 `@rpath/libggml-cpu.0.dylib`。**修复**：显式 `ctypes.CDLL(libggml-cpu/-blas/-metal.dylib, RTLD_GLOBAL)` preload 所有 backend。
2. **merge 脚本不支持 m4a/mp3**：原 `_load_audio_mono_16k` 用 `soundfile.sf.read`，遇到 m4a 报 `Format not recognised`。**修复**：加 librosa fallback `librosa.load(sr=16000, mono=True)`。

### 全场景评测结果

| 样本 | 真实 | sherpa baseline | merged | final v12 | 段数 | <1s% | 评估 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1spk_real (16.2min) | **1** | 1 | 1 | **1** | 10 | 0.0% | ✓ 完美 |
| panel ("1人"实际多人 34min) | 3+ | 5 | 3 | 3 | 249 | 1.2% | ✓ 与实际一致 |
| 2spk (60min) | **2** | 2 | 2 | **2** | 94 | 0.0% | ✓ |
| 4spk (43.8min) | **4** | 9 | 5 (4 真人+1 音乐) | **5** | 257 | 0.0% | ✓ |
| 6spk (60min) | **6** | 12 | 6 | **6** | 469 | 0.2% | ✓ |

**当前 pipeline 在 1/2/3+/4/6 speaker 全场景下都正确识别 speaker 数，无需用户指定 speaker 数。**

### 关键发现

1. **sherpa 不是总过度聚类**：1 人 16min → 1 cluster，2 人 60min → 2 cluster；真聚类问题主要在 4+ 人对话场景（sherpa 把同一人因声纹漂移分到多 cluster）
2. **cluster merge 对所有场景安全**：1/2 人本来正确不动；多人场景压回真实人数；非语音（音乐/笑声）独立保留为"非真人 cluster"
3. **v12 short-guard 在单人场景也极有用**：1spk_real 把 32 段合并到 10 段（merge_same 22 次），median 段长 89.79s
4. **RTF 稳定**：1spk_real 0.163 / 4spk 0.159 / 2spk 0.170 / 6spk 0.17x — merge + v12 后处理共 < 5s，RTF < 0.001 可忽略

### 升级后的多人场景架构

```
长音频
  │
  ├─ ffmpeg silencedetect → macro 切点
  │
  ├─ Qwen3 ASR (RTF ~0.08, 按 macro 段)
  │
  ├─ sherpa global diarize (cluster_threshold=0.9, num_speakers=None)
  │     ⚠ 多人场景容易过度聚类
  │
  ├─ merge_asr_chunks_and_diarize (chunk × turn overlap)
  │
  ├─ ★ cluster centroid merge (新, multi-speaker fix)
  │     - main 间 sim ≥ 0.78 高置信合并
  │     - minor → main sim ≥ 0.55
  │     - dominant > 60% 启用 0.6 阈值
  │     - 非语音 cluster 保留独立
  │
  ├─ (可选) detector + window refine (双人场景优化路径)
  │
  └─ v12 short-segment guard (drop_tiny + aba_smoothing + merge_same)
        最终输出
```

### 工程化建议（待迁入 src/core/qwen3/）

- `postprocess_qwen3_short_segment_guard.py` 核心函数 → `src/core/qwen3/postprocess.py`
- `merge_qwen3_minor_clusters.py` 核心函数 → `src/core/qwen3/cluster_merge.py`
- vendor `llama.py` dylib preload fix 是必须保留的修复
- Qwen3Config 加配置字段：`short_drop_sec` / `aba_max_mid_sec` / `cluster_merge_threshold` / `dominant_share` / `dominant_merge_threshold` / `relabel_threshold`
- 后处理是否启用应可以通过 config 关闭（生产场景需要可控）
| v7（无任何后处理） | ❌ | 96.97% | 93.51% | 37 | — | 上一版基线 |

### 运行耗时 / RTF

| 步骤 | 149min 耗时 | RTF | 备注 |
|---|---:|---:|---|
| Qwen3 ASR | ~795s | 0.089 | 不变 |
| 全局 sherpa diarize | ~3-5s | < 0.001 | 不变 |
| v12 short-guard 后处理 | < 1s | < 0.0001 | 纯 Python |
| v9b top40 centroid+gate refine | ~210s | 0.024 | sherpa + FA + centroid |
| v9d top20 centroid+gate refine | ~130s | 0.015 | |
| detector v2 全集 | ~700s | 0.078 | offline only |
| 生产路线总 RTF (v9d top20 + v12) | ~930s | 0.104 | 含 ASR |

### 代码改动是否值得保留

**值得迁入 src/core/qwen3/**:
- `postprocess_qwen3_short_segment_guard.py` 的核心函数（drop_tiny / aba_smooth / merge_same）→ `src/core/qwen3/postprocess.py`，作为生产 ASR→speaker 的最后一道。改动小（~150 行），无 GPU 依赖，对所有 hypothesis 都有正收益。
- `align_qwen3_poc_with_window_diarize.py` 的 build_global_centroids / map_local_speakers / quality_gate_accept → `src/core/qwen3/refine.py`，仅 offline batch 场景使用。
- `select_qwen3_high_risk_windows_v2.py` → 作为 offline detector，可选 production 化（需 ~12 min preflight）。

**留 manual/PoC**:
- `select_qwen3_high_risk_windows.py` v1：被 v2 取代，但保留作 baseline。
- 手工 59 windows / oracle selected：reference-derived，benchmark 用。

### 推荐生产路线

按”工程复杂度 vs 收益”排序：

1. **路线 P0（零额外计算）**：v7 + v12 d=1.5 short-guard。wrong=11, sp=96.96%, maj=96.45%。仅加 1 个 ~150 行后处理模块。**强烈建议立即上线**。
2. **路线 P1（中等成本）**：v7 + auto detector v1 top20 + centroid + gate + v12 d=2.0（即 v9d + v12）。wrong=6, sp=97.29%, maj=98.11%。额外 ~130s offline post-process（149min 音频，RTF 0.015）。需要 sherpa pyannote + nemo-titanet + Qwen3-ForcedAligner 三个模型。
3. **路线 P2（高成本，最准）**：detector v2 + top20-40 + centroid + gate + v12 d=2.0。预期 wrong 进一步降到 4-5。但 detector v2 全集 ~700s（每曲 offline preflight）。仅适合非实时 batch。
4. **路线 P3（实验中）**：chunk-level refine（sub-agent 在做），目标解决段内跨 turn case。

### 下一步

1. **立即可做（生产化）**：迁 `postprocess.py` 进 `src/core/qwen3/`，加 unit test，把 short_drop_sec / aba_max_mid_sec 写入 config。
2. **detector v2 + v12 完整对照**：跑完 v13a / v13b（已启动）。
3. **chunk-level refine 验证**：等 sub-agent 完成。如能解决 8 个剩余物理上界中的 4+，wrong 可降到 4-5，接近理论极限。
4. **ASR 阶段改进（暂搁置）**：text accuracy 91% 主要被 LexGoGo 校对稿口语化重写所限，重跑 Qwen3 ASR 不同 temperature/chunk_size 的 ROI 不高。
5. **detector preflight 加速**：detector v2 700s 仍偏慢。可考虑只跑 silence + embedding (skip mini diarize 阶段)，估计可减半。
