# diarize 开关 API 设计定案（双引擎）

> 来源：`/plan-ceo-review`（2026-06-10，SELECTIVE EXPANSION 模式）+ Codex 外部声音挑战。
> 前序讨论：`2026-06-10-qwen3-diarize开关API-设计讨论-新session-prompt.md`（交接文档，其 5 个开放问题本文全部定案）。
> 共 14 个决策（D1-D10 + Codex 张力 T1-T4），0 个未决。下一步：`/plan-eng-review` 工程细化后落地。

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

## 落地顺序（commit 粒度，TDD 红→绿→commit）

1. 极简核心：diarize 字段 + 穿透（暂照 language 模板）+ qwen3 跳层 + 缓存折维 + 输出约定（null/SRT）+ 部署假设单测
2. E3：TranscribeOptions 收拢（language+diarize 迁入，全量 parity）
3. 切段分层 helper（词切 → 静音切+char-ratio 兜底）
4. E1 投影（serve 层、不回写）+ funasr 投影路径
5. E2 metadata 回显 + stats 三维度 + 文档（部署顺序、SRT 形态变化、num_speakers 闲置语义）
6. 3060 RTF 重测 probe（验收）

## NOT in scope

- mode 三档枚举（无需求方，bool 可平滑升级）
- num_speakers per-request（TODOS #15）
- funasr 启动级不加载 cam++（TODOS #16）
- 词级替换式 merge（TODOS #14 条件触发）
- 缓存结构化 variant（>3 维触发）

## Codex 外部声音裁决记录

18 条挑战：T1（分层切段）/ T2（不回写）两处修正原决策；T4 采纳 8 条机械补强（已融入上文）；
驳回 #5（opt-in 化解 schema 兼容焦虑）、#11-13（范围与缓存质疑，已知情决策）、#18（qwen3 跳 diarize 后无 turn 边界，"保持原分段"物理不可能）。
