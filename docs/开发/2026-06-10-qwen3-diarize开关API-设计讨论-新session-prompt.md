# 设计讨论交接:qwen3 引擎对外暴露「是否区分说话人(diarize)」的 API 开关

> 用途:新开 session 继续这个设计议题的交接 prompt。背景、已达成结论、关键代码事实、**词级时间戳落地后的新形势**、待深入的开放问题都打包好,新 session 不必重新推导。直接 `@` 引用或粘贴即可。
>
> 生成于 2026-06-10。**这是 `2026-06-09-qwen3-diarize开关API-设计讨论-新session-prompt.md` 的更新版**——上次讨论后词级时间戳功能已落地,改写了核心难点,旧文档作存档,以本文为准。

## 任务性质
**设计讨论**(用户原话「聊一下」),目标把方案聊清楚、想透权衡,**暂不要急着写代码或改文件**。先讨论达成共识,再决定落地。

## 项目背景
- 项目:`funasr_spk_server`,WebSocket 语音转录服务,dev 在 `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`(8867);CUDA 远端 dev box 见记忆 `reference_cuda_box`。
- 主力引擎 `qwen3` = qwen3-diarize:ASR(llama.cpp GGUF,`src/core/qwen3/asr.py`)+ diarize(sherpa / ort_cuda,`src/core/qwen3/diarize.py` / `diarize_ort.py`)**两个解耦模块**,在 `src/core/qwen3_transcriber.py:transcribe()` 缝合。
- 先读 `CLAUDE.md`(「Qwen3 后处理 pipeline」现在是 **5.5 + 6 层**、「关键架构决策:为什么不过度抽象」、「Config 体系」)。

## 核心问题
能否通过 API(WebSocket `upload_request`)让客户端选择**只要 ASR(不区分说话人)/ 要 ASR+说话人分离**?怎么设计最顺架构、代价最小。

---

## ⚡ 自上次讨论(2026-06-09)以来的重大变化:词级时间戳已落地

**这是本次讨论与上次最大的不同,直接改写了原核心难点。**

上次讨论收口时点的开放问题是「关 diarize 后,segments 怎么切分句」——当时没有词级时间,只能靠 ASR 的粗 40s chunk + 标点猜,是个真难点。**现在不是了**:

- **词级时间戳功能已实现并上线**(分支 `spike/qwen3-diarize-poc`,12+ commits)。Qwen3 引擎可增量挂 `segment.words: [{text, start, end, confidence}]`(MMS-300M CTC-FA,`src/core/qwen3/word_align.py`)。
- `word_align_enabled`:字段默认关;**`cuda_prod`/`cuda_dev` profile 默认开**(CUDA 仅 +1% RTF 实测,Mac CPU +17%)。
- 这意味着**「无 speaker 但有精确词级时间」的基线已恒在**——关 diarize 后要切分句,直接用 `segment.words` 的词级时间 + 现成的 `apply_silence_align_to_segments`(纯切句、不依赖 speaker)即可,不再需要新发明分句逻辑。
- 详见 `CLAUDE.md`「Qwen3 后处理 pipeline」5.5 层 + `spikes/qwen3_word_timestamp/SUMMARY.md`。

**结论:原最棘手的开放问题塌缩了。本次讨论焦点应前移到 API 形态 + 缓存 + mode 语义。**

---

## 当前事实(已确认,2026-06-10 最新)
- diarize 在 `transcribe()` 里**无条件跑**(`qwen3_transcriber.py:397` `run_diarization_dispatched`),API 层**没有开关**。
- 后处理顺序(`transcribe()` 内):`filter_spurious_speakers` → `apply_cluster_centroid_merge`(speaker) → `merge_asr_chunks_and_diarize`(按 diarize turn 切文本) → `apply_short_segment_guard` → `apply_silence_align_to_segments`(**纯切句,不依赖 speaker**) → **word_align 挂词(5.5,不依赖 speaker)** → `relabel_segments_by_duration_desc`(speaker)。speaker 相关层:cluster_merge / merge_asr_chunks(turn 边界) / relabel。
- **per-request 字段穿透链路已经建好(词级时间戳时铺的,可直接照抄)**:`language` 字段从 `FileUploadRequest`(schema)→ `websocket_handler`(单传 + 分片 session 回填)→ `task_manager.create_task` / `TranscriptionTask` → pool(`qwen3_inproc_pool` + `qwen3_pool_transcriber` 的 `extra_task_fields`)/ `qwen3_worker_process` → `Qwen3DiarizeTranscriber.transcribe(language=...)`。**diarize 开关字段照这条链路加一个就行**,模板现成。
- **缓存 key 扩展机制也已经建好**:`src/core/database.py:compute_cache_engine(engine, word_align_enabled, language, word_align_language)` 把 word_align 状态折进 engine tag(`qwen3` → `qwen3+wa:chi`),`cache_lookup_params` 同时返回 strict cross-engine 标志。**diarize 开关照同样模式再折一维即可**(如 `qwen3+wa:chi+nospk`),task_manager 的 `_cache_lookup_params` / `_cache_save_engine` 是挂载点。
- dispatch(`transcriber_dispatch.py`)限制:per-request `engine` 必须 == `default_engine`,否则被拒。
- `upload_request.data` 现有字段:`file_name / file_size / file_hash / force_refresh / output_format / engine / language`。

## 已达成的结论(继承上次,可直接深化或质疑)
1. **顺架构的「减法」**:ASR 与 diarize 本就解耦,关 diarize 出纯文本不逆架构。
2. **难点三件事——现在只剩两件半**:
   - ~~分段路径要换一条~~ → **已被词级时间戳解决**(用 `segment.words` + silence_align 切句)。
   - **缓存 key 必须纳入开关**:否则带/不带 speaker 两种结果串。`compute_cache_engine` 已是现成可扩展点。
   - **输出 schema 的「无 speaker」约定**:`segments` 保留 `speaker` 字段统一填 `Speaker1` 还是 `null`?客户端零改动优先。
3. **性能收益实测**(上次数据,词级时间戳前):81min 音频 ASR 109s + diarize/后处理 80s;关 diarize 省 ~40%(RTF 0.039→0.022)。注:现在 CUDA 默认开词级时间戳后基线变了,关 diarize 的相对收益需重新估。
4. **输出是个谱系,不是二元**:①纯全文 ②带词级时间戳分句(无 speaker)③带说话人。中间档现在**质量很高**(有真词级时间),对「要 SRT 字幕但不要说话人」很有价值。
5. **实现姿势**:同引擎加 per-request flag(别新建引擎);默认开 diarize 保兼容;严格 TDD。

## 待深入的开放问题(本次讨论新焦点)
1. **flag 形态:二元布尔 vs 三档 mode 枚举**——鉴于输出是 ①纯文本 ②词级分句无 speaker ③带 speaker 的谱系,一个 `diarize: bool` 够吗,还是 `mode: text|segments|speakers` 更贴谱系?跟词级时间戳的 `words` 字段如何组合(纯文本要不要也带 words)?
2. **缓存 key 的具体形态**:`compute_cache_engine` 已折了 word_align 维,再加 diarize 维后 tag 会变成 `qwen3+wa:chi+nospk` 这种。维度组合爆炸怎么控?要不要把 word_align/diarize/language 统一成一个结构化的 cache variant?
3. **`num_speakers` 参数与这个开关的语义关系**:`FUNASR_QWEN3_NUM_SPEAKERS` 已存在;per-request 关 diarize 时它怎么处理;要不要也 per-request 化。
4. **「无 speaker」输出约定**:`speaker` 字段填 `Speaker1` / `null` / 省略,哪个对下游(SRT 渲染 `f"Speaker{i+1}"`、客户端)最省改动。

## 约束
遵守 `CLAUDE.md`:严格 TDD(红→绿→commit)、低耦合高内聚、不过度抽象、改 FunASR/Qwen3 路径跑 parity(`FUNASR_RUN_INTEGRATION=1`)。本项目独立维护,**不要建议开 PR**(记忆 `feedback_no_pr_workflow`),讨论清楚后直接讲落地步骤。

## 相关文档 / 记忆
- 上一版讨论存档:`docs/开发/2026-06-09-qwen3-diarize开关API-设计讨论-新session-prompt.md`
- 词级时间戳落地:`CLAUDE.md`「Qwen3 后处理 pipeline」5.5 层、`spikes/qwen3_word_timestamp/SUMMARY.md`(含 3060 CUDA RTF 数据)、`docs/部署.md` 五节
- 记忆:`project_word_timestamp_multilang`(词级时间戳决策+落地)、`reference_cuda_box`、`feedback_tdd_strict`、`feedback_no_pr_workflow`、`feedback_question_framing`
- TODOS `#11`(cam++ 剥离独立 SpeakerDiarizer Stage)、`#14`(词级时间戳替换式 merge,Phase 2)
