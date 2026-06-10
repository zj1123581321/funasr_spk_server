# cluster_merge embedding extractor 122.88s 段长崩溃 — 修复结案

> 课题来源：`2026-06-10-cluster_merge-extractor-122s崩溃-新session-prompt.md`
> （ort_cuda 短音频 under-detect 修复验收时发现的遗留事项）。
> 修复 commit：`6c25729`（分支 `spike/qwen3-diarize-poc`）。

## 问题回顾

TitaNet ONNX 导出图的 Where mask 维硬编码 12288 帧（×10ms hop = 122.88s），
≥123s 的单段输入必抛 RuntimeError（sherpa wrapper 与 ORT 直跑同边界）。
生产 transcribe 的 try/except 兜住异常后，**整层 cluster_merge 静默失效**：
2spk_60min（含 >4min 单人长 turn）上该层从来没真正生效过，over-detect
防护全靠 filter_spurious 独自扛。

## 修复内容（两层，A 治根 + B 兜底）

### A. extractor 段长上限（`src/core/qwen3_transcriber.py`）

- 新增模块常量 `MAX_EXTRACTOR_SEGMENT_SEC = 120.0`（< 122.88s 留安全余量；
  这是模型导出产物的物理约束，不是可调参数，故不进 config）。
- 新增 `_cap_extractor_fn(raw_fn, max_segment_sec)` 包装器：
  - ≤ 上限的段原样透传；
  - 超限段切成 `n = ceil(dur / max)` 个**等宽连续窗**（每窗 ≥ 60s，不会出
    碎窗），逐窗 embedding 后平均再 L2 归一化（centroid 语义上分块平均无损）；
  - `end` 先 clamp 到音频实际长度再判窗，避免越界请求切出无意义窗；
  - 窗级 None（太短/越界）跳过，全 None 返回 None（与 raw 语义一致）。
- `build_embedding_extractor_fn` 返回值统一套上 cap —— 一处改动全局生效，
  所有消费方（transcriber 生产路径 / matrix / eval 脚本）自动受益。

### B. build_centroids per-段容错（`src/core/qwen3/cluster_merge.py`）

单段 embedding 失败（任何异常）跳过该段 + warning 日志，centroid 用剩余段
算；某 speaker 全部段失败则该 speaker 无 centroid（既有跳过语义），不再让
整层 cluster_merge 因一段阵亡。

## 验收结果（2026-06-10，全部达标）

1. **单测**：新增 `tests/unit/test_cluster_merge_long_segment.py` 11 个，
   钉死：底层调用 ≤ 上限 / 窗连续覆盖无缝隙 / 平均后 L2 归一化 / None 窗
   跳过 / end clamp / fake sherpa 接线验证 / per-段容错。全量 unit 套件
   584 过（6 个既有 .env 污染失败与本课题无关）。
2. **Mac 真实模型最小复现**：123s / 240s / 350s 段全部正常返回 192 维归一
   化 embedding（修复前 ≥123s 必崩）。
3. **短音频矩阵**（3060，`scripts/_remote_short_audio_matrix.py`）：
   全程**零 "cluster_merge 失败" warning**（修复前 1spk/2spk 300s/600s 稳定
   打）；1spk/2spk 全 6 组 parity OK 且人数全对；4/6spk 5 组 mismatch 落在
   上轮结案钉死的 ±1~3 双向差异既有属性内（4spk 600s sherpa=3/ort=4 与基线
   举例一字不差），非回归。
4. **长音频评测**（3060，`scripts/_remote_diarize_accuracy_eval.py`）：
   - 全 log 零 "cluster_merge 失败"（修复前该音频两轮评测 4 次全崩）；
   - **cluster_merge 首次在 60min 长音频上真实跑完**：2spk_60min default
     post_wall 19.5s vs no_cluster_merge 4.9s（差值 = 该层 embedding 计算
     真实发生，含 >4min 长 turn 的切窗平均）；
   - 该层真实干活的证据：panel_3+ default=3 vs no_cluster_merge=6（收掉
     +3 over）、6spk_60min 6 vs 11（收掉 +5）、4spk 5 vs 10；
   - 5 段 spk_count 回归与基线**逐项一致**（ort_cuda 与 sherpa 双 backend）：

   | 段 | true | ort_cuda | sherpa | 判定 |
   |---|---|---|---|---|
   | 1spk_real 16min | 1 | 1 | 1 | ✓ |
   | panel 3+ 34min | 3 | 3 | 3 | ✓ |
   | 2spk_60min | 2 | 2 | 2 | ✓ |
   | 4spk 44min | 4 | 5 | 5 | expected（4 真人 + 英文歌 noise，基线同款） |
   | 6spk_60min | 6 | 6 | 6 | ✓ |

   注：评测日志 grep "失败" 命中的两行是 ASR 转写文本逐字流（音频内容
   "尝试以失败告终"），非错误。

## 未做（按交接文档定案）

- **C. 重导出 TitaNet ONNX**（去掉 12288 固定 mask）：A/B 已够用，不动模型
  artifact，免去全量精度回归 + 模型分发。
