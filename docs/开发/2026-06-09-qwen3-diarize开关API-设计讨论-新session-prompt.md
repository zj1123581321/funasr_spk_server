# 设计讨论交接:qwen3 引擎对外暴露"是否开启说话人转录(diarize)"的 API 开关

> 用途:新开 session 继续讨论这个设计议题用的交接 prompt。把背景、已达成的结论、关键代码事实、待深入的开放问题都打包好,新 session 不必重新推导。直接把本文件 `@` 引用或粘贴即可。
>
> 生成于 2026-06-09。

## 任务性质
这是一次**设计讨论**(用户原话"单纯聊一下"),目标是把方案聊清楚、想透权衡,**暂不要急着写代码或改文件**。先讨论,达成共识后再决定是否落地。

## 项目背景
- 项目:`funasr_spk_server`,WebSocket 语音转录服务,本地 dev 在 `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server`(端口 8867)。
- 当前主力引擎 `qwen3`,实质是 **qwen3-diarize**:底层 ASR(llama.cpp GGUF,`src/core/qwen3/asr.py`)+ diarize(sherpa / ort_cuda,`src/core/qwen3/diarize.py` / `diarize_ort.py`)**两个解耦模块**,在 `src/core/qwen3_transcriber.py` 的 `transcribe()` 里缝合。
- 先读 `CLAUDE.md`(尤其「Qwen3 后处理 pipeline」「加新引擎/diarize backend 步骤」「关键架构决策:为什么不过度抽象」)和自动记忆里的 `reference_capswriter_engine`(qwen_asr_gguf 引擎自带 Forced Aligner timestamp)。

## 核心问题
能否对外通过 API(WebSocket `upload_request`)让客户端选择**只要 ASR 纯文本 / 要 ASR+说话人分离**?怎么设计最顺架构、代价最小。

## 当前事实(已确认)
- diarize 在 `transcribe()` 里**无条件跑**,API 层**没有开关**。`upload_request.data` 现有字段:`file_name / file_size / file_hash / force_refresh / output_format / engine`(解析在 `src/api/websocket_handler.py` + `file_handler.py`)。
- 缓存 key = `(file_hash, engine)`(`src/core/database.py:192`)。
- 后处理 6 层(顺序固定):`filter_spurious_speakers` → `apply_cluster_centroid_merge` → `merge_asr_chunks_and_diarize` → `apply_short_segment_guard` → `apply_silence_align_to_segments` → `relabel_segments_by_duration_desc`。其中 1/2/6 是 speaker 相关,5(silence_align)纯切句对齐**不依赖 speaker**。
- dispatch(`transcriber_dispatch.py`)有限制:per-request `engine` 必须 == `default_engine`,否则被拒。

## 已达成的结论(不必重复推导,可直接深化或质疑)
1. **顺架构的"减法"**:ASR 与 diarize 本就解耦,关 diarize 出纯文本不逆架构。
2. **难点不在"跳过 diarize",而在三件事**:
   - **分段路径要换一条**:现在 `segments` 是 `merge_asr_chunks_and_diarize` 按 diarize turn 边界切的;关 diarize 后没有 turn,需走**基于 ASR 自带 timestamp + 标点的分句**替代逻辑(新代码,非旁路)。
   - **缓存 key 必须纳入开关**:否则纯文本/带说话人两种结果在 `(file_hash, engine)` 下互相串。需扩成含 diarize_flag。
   - **输出 schema 的"无 speaker"约定**:`segments` 保留 `speaker` 字段统一填 `Speaker1`/`null`,客户端零改动。
3. **性能收益实测**:81min 音频 ASR 109s + diarize/后处理 80s;关 diarize 省 ~40%(RTF 0.039→0.022)。
4. **输出是个谱系,不是二元**:①纯全文 ②带时间戳分句(无 speaker,只靠 ASR timestamp)③带说话人。中间档对"生成 SRT 字幕但不要说话人"很有价值。
5. **实现姿势**:同引擎加 per-request flag(别新建引擎,符合"不过度抽象");默认开 diarize 保兼容;TDD 先写红测。

## 待深入的开放问题(收口时点到、希望新 session 展开)
**「基于 ASR timestamp 的分句」具体怎么切**——按标点?按停顿(silence)?按固定时长窗?三者组合?这个细节直接决定纯文本/字幕模式的输出质量。这是最该先聊透的点。

其他可延伸:flag 用二元布尔还是三档 mode 枚举;缓存 key 扩展的具体形态;`num_speakers` 参数与这个开关的语义关系。

## 约束
遵守 `CLAUDE.md`:严格 TDD(红→绿→commit)、低耦合高内聚、不过度抽象、改 FunASR/Qwen3 路径后跑 parity。本项目独立维护,**不要建议开 PR**,讨论清楚后直接讲落地步骤即可。
