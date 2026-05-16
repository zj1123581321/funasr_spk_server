# 新 session 启动 prompt — PR4 后处理工程化

> 日期: 2026-05-16
> 上一阶段产出: `docs/开发/PR4-149min校对稿对比分析.md` 三轮补充
> 评测集 README: `tmp_long_audio/eval_set/README.md`

## Prompt 正文（复制以下内容到新 session）

你接手 funasr_spk_server 的 PR4 工程化阶段。**PoC 阶段已完成**，跑通 1/2/3+/4/6 speaker 全场景，关键算法（short-segment guard + cluster centroid merge）已在 `tests/manual/server/` 验证有效。**本 session 的目标是把这些 PoC 算法落地到 production path，并加完整测试**。

### 项目路径
`/Users/zhanglixing/Dev/projects/250729_funasr_spk_server/funasr_spk_server`

### 必读（按顺序）
1. `docs/开发/PR4-149min校对稿对比分析.md` — 完整实验报告，特别是 **补充（三）** 和 **补充（四）**（多人场景）
2. `tmp_long_audio/eval_set/README.md` — 评测集说明、复现命令、当前结果
3. `tests/manual/server/postprocess_qwen3_short_segment_guard.py` — short-guard 完整实现
4. `tests/manual/server/merge_qwen3_minor_clusters.py` — cluster centroid merge 完整实现
5. `tests/unit/test_qwen3_short_segment_guard.py` — short-guard 已有 18 个单元测试（全过）
6. `src/core/qwen3/__init__.py` / `merge.py` / `diarize.py` / `asr.py` — 当前 production qwen3 模块
7. `src/core/qwen3_transcriber.py` / `qwen3_pool_transcriber.py` / `qwen3_worker_process.py` — 生产链路
8. `src/core/config.py` — Qwen3Config（已有 `cluster_threshold` / `num_speakers` 等）
9. `src/core/vendor/qwen_asr_gguf/inference/llama.py` — **已修复**的 vendor dylib preload bug（不要回退）

### PoC 阶段核心结论（不要重新验证）

**已验证的算法（5 个评测样本全过）**：

| 样本 | 真实 | sherpa baseline | + cluster merge | + v12 short-guard final |
|---|---:|---:|---:|---:|
| 1 人 16min | 1 | 1 | 1 | **1** ✓ |
| panel 标"1人"实多人 | 3+ | 5 | 3 | **3** ✓ |
| 2 人 60min | 2 | 2 | 2 | **2** ✓ |
| 4 人 44min | 4 | 9 (过聚) | 5 (4 真人+1 音乐) | **5** ✓ |
| 6 人 60min | 6 | 12 (过聚) | 6 | **6** ✓ |

**最佳生产路线（双人 149min 数据）**：v9d top20 + centroid + gate + v12 d=2.0 → wrong=6 / sp=97.29% / maj=98.11%（无 reference 必需）

**最低成本路线**：v7 + v12 d=1.5 → wrong=11 / sp=96.96% / maj=96.45% — 仅纯后处理，无 GPU

### 工程化的 4 个 PR 提议（按风险升序）

#### PR1: vendor dylib preload fix（已修，巩固）

- 改动文件: `src/core/vendor/qwen_asr_gguf/inference/llama.py:231` 附近
- **已修复**：显式 `ctypes.CDLL(.../libggml-cpu.dylib, RTLD_GLOBAL)` preload cpu/blas/metal backend
- 行为：无 behavioral change（修一个之前 crash 的 bug）
- 风险：极低
- 任务：把这个修复提交成独立 commit + 加 smoke 测试（确保 `build_engine()` 不再因 dylib 加载失败 crash）
- 验证：跑 `tests/integration/test_qwen3_diarize_e2e.py`（已有 e2e）

#### PR2: short-segment guard 后处理工程化（核心改动）

- 目标：把 `tests/manual/server/postprocess_qwen3_short_segment_guard.py` 的核心函数迁入 production path
- 新增文件: `src/core/qwen3/postprocess.py`（暂用此名，避免与 `src/core/qwen3_postprocess.py` 冲突 — 那个是 ASR 文本 glossary）
  - 函数：`drop_tiny_segments` / `aba_smoothing` / `merge_consecutive_same_speaker` / `is_backchannel` / `is_question_tail`
  - 入口：`apply_short_segment_guard(segments, short_drop_sec, aba_max_mid_sec, merge_same)` → 返回新 segments + stats
- `src/core/config.py` 加 Qwen3Config 字段：
  - `short_segment_guard_enabled: bool = True`
  - `short_segment_drop_sec: float = 1.5`
  - `short_segment_aba_max_mid_sec: float = 1.5`
  - `short_segment_merge_same: bool = True`
  - 加 env override：`FUNASR_QWEN3_SHORT_GUARD_ENABLED` / `FUNASR_QWEN3_SHORT_DROP_SEC` 等
- `.env.example` 同步加示例
- 集成点：`src/core/qwen3_transcriber.py`（或 worker 端）在拿到 segments 之后调 `apply_short_segment_guard`
- 单元测试：把 `tests/unit/test_qwen3_short_segment_guard.py` 18 个 case 全部跑通（路径切到 `src/core/qwen3/postprocess.py`）
- 集成测试：`tests/integration/test_qwen3_diarize_e2e.py` 加一个 case 验证短段 guard 工作（构造一个有 0.0s 幽灵段的 hypothesis，确认 guard 合并掉）
- 风险：低（纯后处理，不动 ASR/diarize）
- 用户决策点：
  - 是否默认开启？建议 default `True`（实验数据显示对所有 baseline 都正向）
  - 是否需要 SRT 输出也走 guard？建议是

#### PR3: cluster centroid merge 工程化（多人场景关键修复）

- 目标：把 `tests/manual/server/merge_qwen3_minor_clusters.py` 迁入 production
- 新增文件: `src/core/qwen3/cluster_merge.py`
  - 函数：`build_extractor` / `cluster_centroids` / `cosine` / `merge_clusters_by_centroid`
  - 入口：`apply_cluster_centroid_merge(segments, audio_path, **thresholds)` → 返回新 segments + merge log
  - 注意：函数需要加载 audio + sherpa embedding extractor，要复用 `src/core/qwen3/diarize.py` 已有的 `_load_audio_mono_16k`（加 librosa m4a/mp3 fallback）
- `src/core/qwen3/diarize.py` 改造：把 `_load_audio_mono_16k` 加 librosa fallback（同步修 PoC 脚本里临时写的版本）
- `src/core/config.py` Qwen3Config 加字段：
  - `cluster_merge_enabled: bool = True`
  - `cluster_merge_min_main_share: float = 0.03`
  - `cluster_merge_relabel_threshold: float = 0.55`
  - `cluster_merge_main_threshold: float = 0.78`
  - `cluster_merge_dominant_share: float = 0.6`
  - `cluster_merge_dominant_threshold: float = 0.6`
  - 全部加 env override
- 集成点：在全局 diarize 之后、merge_asr_chunks_and_diarize 之前调
- 单元测试：新增 `tests/unit/test_qwen3_cluster_merge.py`
  - mock embedding 数据（合成 cluster centroids）+ 各场景：1 dominant / 多 main / 含音乐 / 全部相似
  - 覆盖三个合并路径：main 高置信、minor→main、dominant 模式
- 集成测试：`tests/integration/test_qwen3_diarize_e2e.py` 加 4-speaker 短样本 smoke 测试
- 风险：中（要加载 audio + 跑 embedding extractor，耗时 5-30s 看音频长度）
- 用户决策点：
  - 是否默认开启？2 人场景 sherpa 输出 2 个时 cluster_merge 一次没动 — 几乎无副作用 → 建议 default `True`
  - 单 audio 加载 + extractor 一次性算 centroid，需要保证不影响并发（pool worker 各自加载？还是 main 进程加载？）

#### PR4: 端到端集成 + 评测集 smoke test + 文档同步

- 把 PR2 + PR3 串成完整 pipeline 验证
- 在 `tests/integration/` 加 `test_qwen3_multispeaker_pipeline.py`：
  - 跑 1/2/4 speaker 的小 sample（< 60s 各一段，从 eval_set 切出）
  - 验证 detected speaker 数与预期一致 ± 1
- 文档同步：
  - `README.md` 加多人场景支持声明
  - `docs/部署.md` 检查是否要提模型大小（nemo-titanet ~30MB）
  - `docs/开发/重构计划-ASR引擎抽象.md` 同步 PR4 完成状态
  - **不要**删除 `tests/manual/server/` 下的 PoC 脚本，它们仍是 reproducibility 证据
- 风险：低（只是粘合 + 文档）

### 已经拍板的决策（继承自 PoC 阶段，不再讨论）

1. **不工程化 detector v2 / chunk-level refine**：实测收益不足（detector v2 反而比 v1 差；chunk-level 全场景退步），保留作 PoC 诊断脚本不删
2. **不工程化窗口级 ForcedAligner refine**：双人场景才有意义；多人场景靠 cluster_merge + v12 即可
3. **保留 PoC 脚本不删**：tests/manual/server/ 下 18 个 PoC 脚本都是 reproducibility 资产
4. **eval_set 不入 git**：音频文件按 .gitignore 排除，但 README 入 git
5. **不重跑 ASR 实验**：text accuracy 91% 受 reference 校对稿口语化重写所限，重跑 ASR 不同 temperature/chunk_size 的 ROI 不足

### 工作方式约束（与 PoC 阶段相同）

- 用中文回应
- 用 ctx_batch_execute / ctx_execute_file 处理大输出
- venv: `venv/bin/python`
- 设环境：`unset TMPDIR; export TMPDIR=/tmp; export DYLD_LIBRARY_PATH="$PWD/src/core/vendor/qwen_asr_gguf/inference/bin"` （即使 vendor dylib fix 后，DYLD_LIBRARY_PATH 加上更稳）
- 跑测试：`venv/bin/python -m pytest tests/unit/ -v`
- 跑 e2e（含 ASR/diarize）：`FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/`
- 不要把音频/模型/tmp_long_audio/eval_set/*.{m4a,mp3,wav} 提交 git
- 后台长任务用 `Bash run_in_background: true`，不要在 ctx_execute 里 nohup（已知 TMPDIR 污染坑）

### 已知 TODO（用户已拒收，不要做）

- 不做 detector v2 工程化
- 不做 ASR temperature/chunk_size 调参
- 不做 chunk-level refine 集成
- 不做 ForcedAligner 集成

### TDD 工作流（强制约束 — 不可放宽）

**本阶段严格走 TDD（test-driven development），每一行 production 代码都必须先有一个失败的测试。** 这不是建议，是强制。原因：算法逻辑有边界（短段阈值、ABA 模式、centroid 余弦阈值、dominant 自适应、非语音保留），生产环境出错只能靠测试兜底；PoC 已经覆盖 18 个 case，工程化阶段要把这个测试网做得更密。

**严格遵循以下循环（红 → 绿 → 重构 → commit）**：

1. **红（Red）**：写一个**新的失败测试**。测试名要描述具体行为（e.g. `test_drop_tiny_merges_zero_duration_ghost_into_prev`）。
   - 跑 `pytest tests/unit/test_qwen3_<module>.py::<TestClass>::<test_name> -v`
   - **必须看到测试失败**（红）。如果测试莫名通过，说明测试本身有 bug（写得太宽松）或 production 代码已经实现（说明这个测试无意义）。
   - 失败原因要是"功能未实现"或"行为不符"，**不能**是 import 错 / fixture 错 / typo。
2. **绿（Green）**：写**最少**的 production 代码让这个测试通过。
   - 不要顺手实现其他功能。
   - 不要"预见性"地写 if/else 分支 — 那些分支没有测试覆盖。
   - 测试通过 = 这一步结束。
3. **重构（Refactor）**：在测试**全部通过**的前提下，整理代码（命名、抽取函数、消除重复）。
   - 重构**不能**引入新行为。如果发现需要新行为，停止重构，回到红阶段写新测试。
   - 重构后再跑一次全套测试，确保还是绿。
4. **commit**：每个红绿循环（或一组相关循环）单独 commit。commit 信息格式：
   - `test(qwen3): 加测试 <行为>` (红阶段，加测试但故意先不实现，看到红再 commit；可选)
   - `feat(qwen3): 实现 <行为>` (绿阶段，让前一个测试转绿)
   - `refactor(qwen3): <重构内容>` (重构阶段)
   - 一般可以把红+绿合并成一个 commit：`feat(qwen3): 加 X 函数 + 单测`
   - 但每个 commit 必须保证 `pytest` 全绿（不要让 commit 中间出现红）

**禁止的反模式**：
- ❌ 先把 PoC 脚本里的函数整段复制到 `src/core/qwen3/postprocess.py`，再补几个测试 — 这是"测试跟随实现"，不是 TDD
- ❌ 一次写 5 个测试再一起实现 — 失去了"最少代码"的约束
- ❌ 把 `tests/manual/server/postprocess_qwen3_short_segment_guard.py` 直接 `cp` 到 `src/core/qwen3/postprocess.py` — 必须从空文件开始，每个函数都由测试驱动
- ❌ 复制 PoC 的单元测试 `tests/unit/test_qwen3_short_segment_guard.py` 然后改路径 — 已有的 18 个测试是参考，但工程化新 module 的测试要重新写（命名、组织、断言可以参考）
- ❌ commit 之间留下失败测试 — 每个 commit 必须 `pytest` 全绿

**允许的"加速"**：
- ✅ 把 PoC 函数当作"参考实现"对照：看现在测试要求的行为是不是已有的，写自己的版本
- ✅ 把 PoC 单测的 case 翻译成新 module 的测试（同样的测试场景，路径换一下）
- ✅ 一个 PR（PR2 或 PR3）内分成 5-10 个红绿循环 commit，最后压成一个 PR（但每个 commit 都要绿）

**测试覆盖要求（最低）**：

PR2 short-segment guard（`src/core/qwen3/postprocess.py`）：
- `is_backchannel`: 空字符串 / 单字 "对" / 多字 backchannel / 长文本不匹配 / 标点只 — **至少 5 个测试**
- `is_question_tail`: 短问句 / 过长不匹配 / 无 marker — **至少 3 个测试**
- `drop_tiny_segments`: 正常段不动 / 0.0s 幽灵段合并 / 选最近邻 prev vs next / 边界（首段无 prev 末段无 next）— **至少 5 个测试**
- `aba_smoothing`: backchannel 平滑 / 长段不动 / 非 ABA 不动 / 高密度短碎片触发 — **至少 4 个测试**
- `merge_consecutive_same_speaker`: 相邻同合并 / 不同 speaker 不合并 / gap 太大不合并 — **至少 3 个测试**
- `apply_short_segment_guard` 入口：禁用 flag / 默认全启 / 集成 ghost+ABA+merge — **至少 3 个测试**
- 总计 **≥ 23 个单元测试**（比 PoC 多 5+）

PR3 cluster merge（`src/core/qwen3/cluster_merge.py`）：
- `cosine`: 正向 / 反向 / 正交 / null 处理 — **至少 4 个测试**
- `build_centroids`: 单 cluster / 多 cluster / 段太短被跳过 — **至少 3 个测试**（用 mock extractor）
- `merge_main_high_conf`: 两 main 余弦 ≥ 0.78 合并 / < 0.78 不合并 / 多轮直到稳定 — **至少 3 个测试**
- `merge_minor_to_main`: minor → 最近 main 满足阈值合并 / 不满足保留 / 多个 minor — **至少 3 个测试**
- `merge_dominant_mode`: dominant share ≥ 0.6 时其他 main 用 0.6 阈值 / < 0.6 不触发 / dominant 模式不影响 minor 步骤 — **至少 3 个测试**
- `apply_cluster_centroid_merge` 入口：1 spk (无操作) / 2 spk 不合并 / 4 spk+音乐 cluster 保留独立 / 6 spk 多人合并 — **至少 4 个测试**（用合成 embedding fixture）
- 总计 **≥ 20 个单元测试**

PR4 集成测试（`tests/integration/test_qwen3_multispeaker_pipeline.py`）：
- 用 eval_set 切短的 sample（< 60s 各一段）跑完整 pipeline，验证 detected speakers ± 1
- 至少覆盖 1 / 2 / 4 speaker 三个 case
- 标记 `@pytest.mark.skipif(not FUNASR_RUN_INTEGRATION)`，因为要加载模型

**红阶段诊断技巧**：
- 测试失败信息要"指着具体行为说"。`assert result == expected` 不够，加 `assert result == expected, f"expected {expected}, got {result}"` 或用 `pytest.approx` 等
- 用 `parametrize` 写多个相似 case，避免复制粘贴 — 但每个 param 必须语义不同
- 边界 case 优先：空输入 / 单元素 / 边界值（0.0s / share=阈值±ε）

**Commit 前 checklist**：
1. `venv/bin/python -m pytest tests/unit/ -v` 全绿
2. 改动文件最少必要（不要顺手"清理"无关代码）
3. commit 信息明确说明加了什么测试 + 实现了什么函数
4. 如果是重构 commit，确认行为不变

### 开始

1. 跑 `git status`，确认评测集 + 新脚本都已存在
2. 跑 `venv/bin/python -m pytest tests/unit/test_qwen3_short_segment_guard.py -v`，确认 PoC 阶段 18 个测试全过（这是参考基线，工程化新 module 的测试**不要直接复用**这个文件，必须从空开始按 TDD 写新的）
3. 决定 PR1/PR2/PR3/PR4 的顺序（推荐：PR1 → PR2 → PR3 → PR4）
4. 从 PR1 开始（vendor dylib fix 已实现，但要按 TDD 思路补一个"smoke test 验证 build_engine 不 crash"的测试。先写测试看红 — 但 dylib 已修不会红 → 这是"测试已存在行为"的少数允许情况，commit 信息要写明"加保护性测试，无 production 改动"）
5. 进入 PR2 时，**严格从 `src/core/qwen3/postprocess.py` 空文件开始**：
   - 写第一个测试 `test_is_backchannel_empty_string` 在 `tests/unit/test_qwen3_postprocess.py`
   - 跑测试看红（`ImportError: cannot import name 'is_backchannel'`）
   - 在 `postprocess.py` 写最少代码 `def is_backchannel(s): return True if not s else False`（故意 minimal）
   - 跑测试看绿
   - 写下一个测试 `test_is_backchannel_pure_对` 看红 — 当前实现 `return True if not s else False` 处理"对"时会 return False，**测试红** ✓
   - 继续

### 工程化完成的标准

- [ ] 4 个 PR 全部合入主分支，每个 PR 都有完整 TDD 历史（commit log 体现红绿循环）
- [ ] `tests/unit/` 新增 ≥ 23（postprocess）+ ≥ 20（cluster_merge）= **≥ 43 个新单元测试**，全绿
- [ ] 跑 e2e 集成测试通过（含多人 smoke）
- [ ] Qwen3Config 新字段都有 env override 和 .env.example
- [ ] 默认配置在 eval_set 5 个样本上行为不变（speaker 数符合预期）
- [ ] 文档同步完成（README + 部署 + 重构计划）
- [ ] 评测集 README 标注"production 已支持"
- [ ] **每个 commit 都是绿的**（CI 视角，任何 commit checkout 出来都能 `pytest` 全过）

### TDD 自检：每次提交前问自己

1. 我有没有跳过红阶段，直接写实现？
2. 我有没有为还没写的测试预先实现了功能（YAGNI 违反）？
3. 我有没有为了让测试通过，写了多余的"防御性"代码？
4. 这个 commit 单独 checkout 出来，pytest 还能全绿吗？

任何一个回答 yes（或 last 一个回答 no），停下来回退、整改。
