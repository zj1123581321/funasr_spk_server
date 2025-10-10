# VAD 并发问题解决方案

## 问题描述

Python 版本的 FunASR VAD（语音活动检测）模型在并发请求时会报错：`IndexError: list index out of range`。这是因为 Python 版本的 VAD 不支持多线程并发访问，当多个线程同时调用模型时会导致内部状态冲突。

相关 Issue: [modelscope/FunASR#865](https://github.com/modelscope/FunASR/issues/865)

## 问题原因分析

1. **全局共享模型实例**：项目使用了全局的 transcriber 实例，其中包含 FunASR 模型
2. **多线程并发访问**：TaskManager 创建多个工作线程并发处理任务
3. **线程不安全**：Python 版本的 FunASR VAD 模型不是线程安全的
4. **状态冲突**：并发访问导致模型内部状态冲突，产生索引越界错误

## 解决方案

### 方案一：线程锁保护（默认）

使用 `threading.Lock` 序列化对模型的访问，确保同一时刻只有一个线程能访问 VAD 模型。

**优点**：
- 实现简单，改动最小
- 内存占用低（只需一个模型实例）
- 稳定性高

**缺点**：
- 会降低并发性能（模型推理部分串行执行）

**配置方式**：
```json
{
  "transcription": {
    "concurrency_mode": "lock"
  }
}
```

### 方案二：模型池（可选）

创建多个独立的模型实例，每个并发任务使用独立的模型，实现真正的并发。

**优点**：
- 真正的并发执行
- 性能最优

**缺点**：
- 内存占用高（每个模型实例约 2GB）
- 初始化时间较长

**配置方式**：
```json
{
  "transcription": {
    "concurrency_mode": "pool",
    "max_concurrent_tasks": 4
  }
}
```

注意：模型池大小等于 `max_concurrent_tasks` 的值。

## 实现细节

### 1. FunASRTranscriber 改进

```python
class FunASRTranscriber:
    def __init__(self, config_path: str = "config.json"):
        # 根据配置选择并发模式
        self.concurrency_mode = config["transcription"].get("concurrency_mode", "lock")
        
        if self.concurrency_mode == "lock":
            # 线程锁模式
            self._model_lock = threading.Lock()
        elif self.concurrency_mode == "pool":
            # 模型池模式
            self.model_pool = ModelPool(config_path)
```

### 2. 模型调用保护

线程锁模式下的模型调用：
```python
def _generate_with_lock():
    with self._model_lock:
        result = self.model.generate(...)
        return result

result = await loop.run_in_executor(None, _generate_with_lock)
```

模型池模式下的模型调用：
```python
result = await self.model_pool.generate_with_pool(
    audio_path=audio_path,
    batch_size_s=batch_size_s,
    hotword=hotword
)
```

## 性能对比

| 模式 | 并发数 | 内存占用 | 处理速度 | 稳定性 |
|------|--------|----------|----------|--------|
| 线程锁 | 4 | ~2GB | 中等 | 高 |
| 模型池 | 4 | ~8GB | 快 | 高 |
| 无保护 | 4 | ~2GB | 快 | 低（会崩溃）|

## 测试验证

### 运行并发测试

```bash
# 测试VAD并发功能
python tests/core/test_vad_concurrency.py

# 运行服务器并发测试
python tests/server/test_concurrent_transcription.py 4
```

### 测试结果示例

```
并发测试完成:
  总任务数: 4
  成功: 4
  失败: 0
  总耗时: 15.23秒
  平均耗时: 3.81秒/任务
```

## 建议配置

### 开发环境
- 使用线程锁模式，节省内存
- 并发数设置为 2-4

### 生产环境（内存充足）
- 使用模型池模式，获得最佳性能
- 并发数根据 CPU 核心数和内存大小调整

### 生产环境（内存受限）
- 使用线程锁模式
- 适当降低并发数

## 注意事项

1. **内存管理**：模型池模式需要更多内存，建议至少 16GB RAM
2. **初始化时间**：模型池初始化需要更长时间（加载多个模型）
3. **配置切换**：切换并发模式需要重启服务器
4. **错误处理**：已增强错误日志，便于诊断并发问题

## 未来优化

1. **动态模型池**：根据负载动态调整模型实例数量
2. **进程池方案**：使用多进程替代多线程，完全隔离模型实例
3. **C++ 运行时**：考虑使用支持并发的 C++ 版本 FunASR

## 参考资料

- [FunASR GitHub Issue #865](https://github.com/modelscope/FunASR/issues/865)
- [FunASR C++ Runtime](https://github.com/modelscope/FunASR/tree/main/runtime)
- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)