# spikes/ — 可行性验证脚本

本目录下脚本用于在写正式集成代码前快速验证关键假设，**不进入生产路径**。

## qwen3_spike.py

回答 codex review T7：「Qwen3-ASR 1.7B 是否能在我们的系统下等价输出」。

### 用法

```bash
# 1) probe 模式：列候选 model id
venv/bin/python spikes/qwen3_spike.py --probe

# 2) 实跑（需要 modelscope 或 huggingface 已配置好）
venv/bin/python spikes/qwen3_spike.py \
    --model-id <实际确认的 model id> \
    --audio tests/fixtures/audio/tts_1speaker_5s.wav

# 3) 自定义设备 / 输出
venv/bin/python spikes/qwen3_spike.py \
    --model-id <id> --audio <wav> \
    --device cpu \
    --report spikes/qwen3_spike_report_v2.md
```

### 预期产出

- 控制台日志：每步耗时、是否成功、内存增量
- `spikes/qwen3_spike_report.md`：结构化结论，覆盖 codex review T7 提到的 6 个问题：
  1. 模型加载（model id / 下载耗时）
  2. 单条推理（是否跑通 + 输出结构）
  3. 是否带 timestamps / speaker
  4. 并发安全（spike v1 仅验单调用）
  5. 资源占用（内存 + 设备）
  6. 集成复杂度估算

### Spike 失败的处理

- **失败 = 重大信号**：方案 v2 第 5.6 节明确：spike 失败 → PR2 永远不触发，重构停在 PR1
- 不要给失败的 spike 写补丁强行让它过，直接归档 report 让用户决策
