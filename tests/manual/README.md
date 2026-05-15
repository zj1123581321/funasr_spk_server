# tests/manual — 历史手工测试脚本

本目录下的脚本是 PR1 重构前留下的**手工诊断/集成脚本**，**不在 pytest 自动收集范围**（见 `pytest.ini` 的 `testpaths`）。

## 现状
- 这些文件大多是 `if __name__ == "__main__"` 风格，需要本地手动启动服务器/加载模型后才能跑
- 包含若干 Windows 硬编码路径，跨平台无法直接执行
- 多数是历史 bug 复现 / MPS 调优诊断脚本，保留作参考

## 何时使用
- 复现历史问题：MPS 多进程并发、VAD 段切分、大文件分片上传等
- 性能基准测试：`performance/` 下脚本对硬件做端到端 benchmark
- 调试时本地手跑：`server/test_server_transcription.py` 可对运行中的服务做完整端到端验证

## 未来计划
TODOS.md 中 P3 项已记录：把仍有价值的诊断意图逐步转写为 pytest regression test，迁出 manual 目录。

## 子目录
| 目录 | 用途 |
|------|------|
| `core/` | FunASR 核心逻辑早期实验脚本（merge、VAD、转录原型） |
| `server/` | WebSocket 服务端到端测试 |
| `client/` | 客户端连接脚本 |
| `diagnostics/` | MPS / 并发问题复现 |
| `performance/` | 性能 benchmark |
| `docs/` | 旧的预期输出格式说明 |
