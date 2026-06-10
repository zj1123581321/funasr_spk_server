# diarize 开关 API 设计定案（双引擎）

> ✅ **已落地（2026-06-10，本分支）**：落地顺序 0–6 全部按 TDD 完成（每步红→绿→commit）。
> 代码入口见 CLAUDE.md「diarize 开关」节；客户端字段文档见 `docs/使用/客户端交互指南.md`。
> 第 7 步（3060 RTF 重测 probe）为远端 CUDA box 验收项，待 merge 后在 3060 上执行。

> 来源：`/plan-ceo-review` + `/plan-eng-review`（2026-06-10 同 session）+ 两轮 Codex 外部声音挑战。
> 前序讨论：`2026-06-10-qwen3-diarize开关API-设计讨论-新session-prompt.md`（交接文档，其 5 个开放问题本文全部定案）。
> CEO 层 14 决策 + 工程层 10 决策，0 个未决。**评审全部完成，可直接按「落地顺序（终版）」实现。**
> ⚠️ 本分支须整体 merge 后才可部署（中间 commit 的 API 契约不完整，prod 只从 main 拉）。

## 设计主轴

**API 语义引擎无关，实现策略引擎相关。**

字段：`FileUploadRequest.diarize: bool = True`（行业标准形态，与 Deepgram/AssemblyAI/阿里云同形）。
对外契约：`diarize=false` ⇒ 响应不含说话人区分。各引擎自行决定达成方式：

- **qwen3**：真跳过 `run_diarization_dispatched` + speaker 后处理层（filter_spurious / cluster_merge / short_segment_guard / relabel），省算力（旧数据 ~40% RTF，需重测，见「性能注记」）
- **funasr**：照算、出口投影抹 speaker（cam++ embedding 提取只 gate 在模型加载，`venv .../funasr/auto/auto_model.py:488`；per-call `return_spk_res=False`(:572) 只能跳聚类小头，不值得为此折缓存维）

默认 `true` 完全向后兼容；老客户端永远走默认值，永远看到现状输出。

## 决策清单

### D2 API 形态 = 布尔字段（弃 mode 三档枚举 / 仅输出投影）
枚举为假设需求付复杂度；纯投影放弃核心性能动机。将来真需要第三档时 `diarize:false` 可平滑映射 `mode:segments`。

### D4 funasr 行为 = 输出投影
关键源码事实（评审中查证）：
- cam++ embedding 提取无 per-call 开关（embedding 是大头）
- `spk_mode=punc_segment` 下分段按标点切，**带不带 spk 边界一致** → 投影在 funasr 上无边界漂移
- **funasr 缓存免折维**：存一行 diarized，serve 时按需投影，一行通吃两种请求
- 「启动不加载 cam++」记 TODOS #16（部署形态，条件触发）

### D5+T1 无 speaker 分段 = 分层切段
事实修正：`silence_align` 只做切点吸附不做重切，关 diarize 后段 = ASR ~40s chunk，SRT 不可用。
解法（Codex 张力 T1 修正后）：diarize=false 时对超长段做**两层 fallback 切分**：
1. 有 `segment.words`（CUDA 主力路径 word_align 默认开）→ 词级时间精确切
2. 无 words → 静音表切（silencedetect 现成）+ char-ratio 文本归属（merge 现役机制）

切段 helper 照既有形状 `(关/空/异常)→fallback to input`，内置最小段时长阈值（吸收 short_segment_guard 的通用清理职责）。句级替换式 merge 仍留 TODOS #14。

### D8 无 speaker 输出 = JSON `speaker=null` + SRT 无前缀
- `TranscriptionSegment.speaker` 改 `Optional[str]`；null = 未区分，与「真只有一人」可区分
- SRT 不带 `SpeakerN:` 前缀（字幕场景正是要这个）
- 兼容性：diarize=false 是 opt-in，老客户端无感；**API 文档必须写明 SRT 形态与分段边界语义随 diarize 变化**（不只是少个字段）

### D9 缓存 = 字符串折维 `qwen3+wa:<lang>+nospk`
- 顺序固定、缺省不写；funasr 免折维（D4）、num_speakers 不进维（D7）后仅 2 维 4 形态
- **触发条件写死：维度 >3 升级结构化 variant**
- `compute_cache_engine` / `cache_lookup_params` 是现成挂载点

### E1+T2 缓存投影复用 = serve 层现场优化，**不回写**
- nospk 请求：查 exact tag → miss 且非 force_refresh → 查 diarized 行 → 命中则现场投影返回（标 `projected:true`）
- **不回写**：缓存里永远只有真算结果；词切质量升级自动惠及后续请求（Codex 张力 T2 修正：原定回写会固化 turn 边界投影）
- `force_refresh=true` 跳过投影回退强制重算
- 容错（D6）：投影读失败（具名 KeyError/ValidationError）→ 当 miss 重算 + warn；禁止 catch-all

### E2 effective options 回显
最终 `TranscriptionResult` 的 metadata 块（task status 不挂）：`engine / diarize / word_align / language / projected`。
- **metadata 由 serve 层组装，不随缓存内容存取**（projected 是请求级属性，防缓存污染）
- 合并优先级定义并文档化：**request > 分片 session 回填 > config > 引擎默认**

### E3 TranscribeOptions 收拢
language + diarize 收进一个结构整体穿透（schema → handler 含分片回填 → task_manager → 两套 pool → worker → transcribe）。
**硬约束：动 language 稳定路径，必须全量 parity（`FUNASR_RUN_INTEGRATION=1`）。**

### D7 num_speakers 本次不 per-request 化
记 TODOS #15（缓存折维 + 投影兼容规则连锁，无需求方）；diarize=false 时 config 值闲置打 info 日志 + API 文档写明。

## 部署顺序约束

老 server 的 Pydantic 忽略未知 `diarize` 字段（静默照常带 speaker）→ **server 先升级、客户端后启用字段**。
此假设须有 unit test 断言（把部署假设变成被测事实，Codex T4）。

## 可观测性

stats 日志补三维度：per-task diarize 生效值 / projected 命中计数 / 切段 stats（切分数、fallback 数）。

## 测试矩阵（显式）

`引擎(qwen3|funasr) × diarize(t|f) × word_align(t|f) × 输出(json|srt) × 缓存状态(命中|投影|force_refresh|坏行)`
加：options 穿透（两套 pool 序列化、分片回填）、缓存 tag 4 形态、部署假设断言、切段两分支。
凌晨 2 点门禁 = parity 全量绿 + 缓存矩阵绿。

## 性能注记

「关 diarize 省 ~40%」是 word_align 落地前数据；**3060 RTF 重测 probe 是落地验收项**（非事后任务）。

## 工程评审定案（/plan-eng-review，2026-06-10，10 决策）

### 模块切分（D1-D5）
- **D1 TranscribeOptions 彻底版**：模型放 `src/models/schemas.py`；`TranscriptionTask`
  挂嵌套 `options: TranscribeOptions`，**删平铺 `language`**（内部模型无兼容约束）；
  所有读点（task_manager / handler session 回填 / 两套 pool / worker）改走 `task.options`。
  file-based pool 必须 `model_dump()` 序列化，worker 读 options dict（缺省兜底）。
- **D2 切段归属**：新建 `src/core/qwen3/segment_split.py` 纯函数
  `split_long_segments(segments, *, speech_regions, audio_duration, max_dur, min_dur)
  -> (segments, stats)`（签名与 `silence_intervals_from_speech` 同型，不造模糊
  `silences` 参数）+ transcriber 薄 wrapper（照 `apply_silence_align_to_segments` 形状）。
- **D3+T-A 投影双出口**：纯函数 `src/core/result_projection.py`；应用点两个——
  **`db_manager.get_cached_result` 出口**（覆盖 handler 早返回 + task_manager 缓存读，
  SRT 在此从 segments 重渲染时带 no-prefix）+ **task_manager fresh 结果出口**。
  引擎代码零改动。get_cached_result 签名加 diarize/options 参数。
- **D4 缓存收拢**：`database.py` 公开 `cache_params_for(task)`，消灭 4 处重复
  （task_manager 两个私有 helper 升级迁入 + websocket_handler:199/582 两处手写）。
- **D5 切段 config 件套**：`nospk_split_enabled: bool = True` /
  `nospk_split_max_segment_sec` / `nospk_split_min_segment_sec` + env 覆盖
  （照 silence_align 三件套模式）。

### Codex 工程层挑战采纳（T-B/T-C/T-D）
- **T-B 前置修复现行 bug（commit 0）**：qwen3 + SRT 缓存命中返回空——
  `task_manager.py:307` SRT 模式存 `segments=[]`，而 qwen3 `raw_result` 无
  `sentence_info`（qwen3_transcriber.py:582），database SRT exact-hit 走 funasr
  私有重建路径返回 `""`。修法：SRT 模式 engine 返回携真 segments、缓存存真
  segments（同时是 T-A 投影/重渲染的地基）；复现测试先红后绿；顺带把
  `database.py:297-299` 的 catch-all 换具名异常+日志（坏行不静默吞）。
- **T-C 契约窗口**：不加临时门闩；分支整体 merge（见文首警告）。
- **T-D 机械补强 10 条**：
  1. options 序列化姿势（model_dump / worker dict 解析 / 缺省兜底）
  2. funasr pool 协议钉测试：options 不写进 funasr 任务文件
  3. `speaker: Optional` 涟漪清单：`funasr_transcriber:335` / `qwen3_transcriber:646`
     的 `sorted(set(...))` + 3 处渲染点；**内部 `Segment(speaker:int)` 永不为 None**，
     null 只在出口转换层出现；投影保证 `speakers=[]`
  4. `merge.py:segments_to_srt` 加 `include_speaker: bool` 参数
  5. SRT 的 nospk 切段**永远走静音 fallback**（word_align 保持 JSON-only 不变量），
     文档写明；测试矩阵 SRT 路径单列不与 JSON 合并
  6. split helper 签名精确化（见 D2）
  7. nospk 缓存回退 strict：`+nospk` tag 禁 cross-engine；diarized 回退显式只查
     同引擎同 wa-tag 行
  8. `compute_cache_engine` language 规范化：`strip() or fallback`，测 None/""/" eng "
  9. metadata 不入库：save_result exclude（或 response 层 wrapper），
     测「缓存读出不继承上次请求的 projected」
  10. 测试矩阵补 6 项：handler 早返回投影 / chunked finalize 早返回投影 /
      `.task` JSON 嵌套 options / worker 解析 options / SRT 三路径（fresh/cache/
      projected）无前缀 / 坏缓存行具名异常不静默

### 强制维护项
`qwen3_transcriber.py:8-11` 模块 docstring 管线图、CLAUDE.md「Qwen3 后处理
pipeline」节，随 diarize=false 分支落地同 commit 更新。

## 落地顺序（终版，TDD 红→绿→commit；D6 拆两步 + T-B 前置）

```
0.  T-B 修现行 bug: SRT 模式存真 segments + 复现测试 + 拆 catch-all
1a. 纯重构: TranscribeOptions + Task.options 嵌套, language 迁入
    (行为零变, test_language_propagation 原样通过 + parity 全量)
1b. diarize 字段进 options + 全链路穿透 + 部署假设单测
2.  cache_params_for(task) 收拢 + compute_cache_engine 折 +nospk 维
    (含 language 规范化) + 4 形态矩阵测试
3.  qwen3 transcribe diarize=False 跳层 + chunk 出段 + speaker=null
    schema(Optional 涟漪清单逐点处理) + segments_to_srt include_speaker
    + docstring 图更新
4.  segment_split.py 分层切段(词切→静音切) + config 件套
5.  result_projection.py + 双出口投影(get_cached_result + task_manager)
    + E1 strict 回退(不回写) + force_refresh + 投影容错 + metadata 不入库
6.  E2 metadata 回显 + stats 三维度 + 文档 + CLAUDE.md 更新
7.  3060 RTF 重测 probe(验收)
```

并行参考：1b 之后 `Lane A: 3→4`（qwen3_transcriber 串行）与 `Lane B: 2`
可并行；5 依赖 2+3。单人 TDD 顺序走即可（总量 CC ~3.5 小时）。

## NOT in scope

- mode 三档枚举（无需求方，bool 可平滑升级）
- num_speakers per-request（TODOS #15）
- funasr 启动级不加载 cam++（TODOS #16）
- 词级替换式 merge（TODOS #14 条件触发）
- 缓存结构化 variant（>3 维触发）

## Codex 外部声音裁决记录

18 条挑战：T1（分层切段）/ T2（不回写）两处修正原决策；T4 采纳 8 条机械补强（已融入上文）；
驳回 #5（opt-in 化解 schema 兼容焦虑）、#11-13（范围与缓存质疑，已知情决策）、#18（qwen3 跳 diarize 后无 turn 边界，"保持原分段"物理不可能）。
