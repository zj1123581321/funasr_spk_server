# 设计讨论交接:qwen3 引擎在所有模式下输出"词级时间戳"(word-level timestamp)

> 用途:新开 session 专门聊这个**地基议题**用的交接 prompt。把背景、已核实的代码事实、与"diarize 开关"议题的耦合、待聊透的开放问题都打包好,新 session 不必重新挖代码。直接 `@` 引用本文件即可。
>
> 生成于 2026-06-09。配套议题文档:`docs/开发/2026-06-09-qwen3-diarize开关API-设计讨论-新session-prompt.md`(那个开关议题**依赖本议题先定**,见下文「与 diarize 开关的耦合」)。

## 任务性质
这是一次**设计讨论**(不是急着写代码)。目标:把"让 qwen3 在所有输出模式下都带词级时间戳"这件事想透 —— 要不要启用 aligner、模型从哪来、常开的性能代价多大、值不值。先讨论,达成共识再决定落地。

## 为什么单独开这个 session
原本在聊「diarize 开关 API」(只要文字 / 要文字+说话人)时,发现一个被遗忘的前提:**原计划是所有模式下都有词级时间戳,可选的差异只是区不区分说话人。** 但挖代码发现**生产现状根本没有词级时间戳** —— 现状与目标态有个缺口,这个缺口就是本议题。它是地基:聊清楚后,diarize 开关基本是顺手的事。

## 项目背景
- 项目:`funasr_spk_server`,WebSocket 语音转录服务,dev 在 `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`(端口 8867)。
- 主力引擎 `qwen3` = ASR(llama.cpp GGUF + 可选 Forced Aligner)+ diarize,在 `src/core/qwen3_transcriber.py:transcribe()` 缝合。
- ASR 引擎是携进的 vendor 包 `src/core/vendor/qwen_asr_gguf/`(来自 CapsWriter-Offline 的 qwen_asr_gguf,见自动记忆 `reference_capswriter_engine`:int4 ONNX encoder + q4_k GGUF decoder + **Forced Aligner 自带 timestamp**)。

## 核心问题
能否让 ASR 在**所有模式**下稳定输出**词级**(逐词/逐 token)时间戳?代价是什么、怎么落地最顺。

## 已核实的代码事实(2026-06-09 挖证,带 file:line)

### 1. 生产现状:aligner 默认关、连模型文件都没有
- `src/core/qwen3/asr.py:66` —— `enable_aligner: bool = False`(默认关)。
- `src/core/qwen3/asr.py:249` —— 注释明写"不启 aligner — production 没 aligner 模型, 只拿全文 + segment 级时间(40s chunk)"。
- `src/core/qwen3/asr.py:27` —— `WordItem` docstring:"词级时间戳(production 不带 aligner, 实际仅 segment 级)"。
- `src/core/vendor/qwen_asr_gguf/inference/schema.py:91` —— vendor 默认 `enable_aligner: bool = False`。
- **结论:线上手里只有 ① 带标点的整段全文 ② 40 秒一块的 chunk 级时间戳。没有词级。**

### 2. aligner 代码是现成的,只是没启用 / 没喂模型
- `src/core/vendor/qwen_asr_gguf/inference/asr.py:52-55` —— `self.aligner = None; if config.enable_aligner and config.align_config: ... self.aligner = QwenForcedAligner(config.align_config)`。两个条件都要满足才构造。
- `src/core/vendor/qwen_asr_gguf/inference/asr.py:316-340` —— 推理后 `if self.aligner and res.text.strip(): align_res = self.aligner.align(...)`,产出 `ForcedAlignResult(items=...)`。
- `src/core/qwen3/asr.py:196-199` —— `if result.alignment and result.alignment.items:` 才把 `ForcedAlignItem` 转成 `WordItem(text, start, end)`;否则 `items=[]` 空。
- aligner 实现在 `src/core/vendor/qwen_asr_gguf/inference/aligner.py`(逐字 CJK 拆分对齐,`aligner.py:74`)。

### 3. ASR 全文是带标点的(切句很便宜)
- `src/core/vendor/qwen_asr_gguf/inference/aligner.py:140 / 324` —— "重组**包含标点**的时间戳序列""将缺失的标点符号找回来并补全时间戳"。
- `src/core/vendor/qwen_asr_gguf/inference/asr.py:147` —— 流式输出按 `。？！：,.` 切行。
- 含义:句子边界靠标点就能切,词级时间戳主要解决的是"每句/每词**精确到秒**的起止"。

### 4. 当前 merge 是"无词级时间戳"下的近似方案
- `src/core/qwen3/merge.py:5` —— 头注释:"asr_text: ASR 输出的完整文本 (**无 timestamp,因为 production aligner 模型不存在**)"。
- `src/core/qwen3/merge.py:167-213` —— `_choose_text_boundary`(按标点找切点)+ `_split_text_by_weights`(按字符数比例把文本分摊进时间窗)。
- `src/core/qwen3/merge.py:216+` —— `merge_asr_chunks_and_diarize(chunks, turns)`:把 40 秒 chunk 文本按字符比例切进 diarize turn 边界。
- 第 5 层后处理 `apply_silence_align_to_segments`(`qwen3_transcriber.py` 内,RTF 影响 <1%)用 ffmpeg silencedetect 把切点**吸附到最近静音**,部分弥补近似切句的漂移(60s podcast 对齐率 +19pp,60min +33pp)。
- **关键推论:今天 diarize 开着时,句子时间戳本身也不是词级精确的** —— 是"标点 + 字符比例 + 静音吸附"的近似。词级 aligner 是要把这套近似换成真精确。

### 5. 那"省 40%"的钱花在哪
- 81min 音频:ASR ~109s + diarize/后处理 ~80s,关 diarize 省 ~40%(RTF 0.039→0.022)。来源见 diarize 开关交接文档。
- diarize 的开销 = pyannote-segmentation-3.0 + TitaNet embedding + 聚类(sherpa FastClustering / ort_cuda scipy 复刻),**不是切句本身**。

## 与 diarize 开关议题的耦合(必须在本 session 想清楚)

> **经济账耦合:** "关 diarize 省 40%"这个数字是建立在**当前没有 aligner**的现状上的。如果词级时间戳靠 aligner 且 aligner 在**所有模式常开**,那 aligner 自身的推理开销会吃掉一部分省下的时间。
>
> 必须实测/估算:**aligner 常开后的 RTF 增量是多少?** 关 diarize 的净收益要用 `(diarize 省下) − (aligner 常开新增)` 重算。这个结论直接决定 diarize 开关还值不值得做、收益宣传口径怎么写。

定了本议题(aligner 启不启、代价多大),diarize 开关那边就清爽了:**输出谱系塌缩成一个布尔"区不区分说话人"**(因为时间戳是恒有基线),实现照 A 方案(`diarize: bool` + 复用切句器 + 缓存 key 加一维),不再需要三档枚举或"关时才启 aligner"那种拧巴设计。

## 待聊透的开放问题
1. **aligner 模型从哪来?** vendor 的 `QwenForcedAligner` 需要 `config.align_config` 指向一个模型。这模型是什么(独立 aligner 模型?复用 encoder?)、CapsWriter upstream 有没有现成的、要不要转换/打包到生产。**这是最大未知,建议先查 vendor/aligner.py + CapsWriter upstream 确认模型依赖。**
2. **常开的性能代价。** aligner 推理在 Mac MPS / CUDA 上各自 RTF 增量多少?会不会成为新瓶颈?要不要也走池化?
3. **精度收益值不值。** 现有"标点+字符比例+静音吸附"近似的实际误差有多大(可拿现有 eval 音频测)?换成词级精确,下游(字幕、检索、对齐)收益有多大?
4. **要不要全模式常开,还是按需。** 如果常开代价大,是否退而求其次:只在需要时(某个 flag)启 aligner —— 但这又和"所有模式都有词级时间戳"的原始目标冲突,要权衡。
5. **WordItem → segments 的暴露路径。** 现在 `items` 即使非空也没暴露到最终 `TranscriptionSegment`,需要设计新的 merge(`词级 items → 句段`)替换/旁路掉 `merge_asr_chunks_and_diarize` 的字符比例近似。

## 约束
遵守 `CLAUDE.md`:严格 TDD(红→绿→commit,见自动记忆 `feedback_tdd_strict`)、低耦合高内聚、不过度抽象、改 Qwen3 路径后跑 parity(`FUNASR_RUN_INTEGRATION=1`)。本项目独立维护,**不要建议开 PR**(见 `feedback_no_pr_workflow`),讨论清楚后直接讲落地步骤。

## 提问风格(用户偏好)
向用户提问 / 给选项时:**先用大白话讲清问题背景**,再结合工程最佳实践讲选项差异和推荐(见自动记忆 `feedback_question_framing`)。不要直接抛选项卡。
