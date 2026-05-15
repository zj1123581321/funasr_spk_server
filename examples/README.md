# examples/ — 客户端示例与工具

本目录存放**调用 funasr-server 的客户端示例**，跟项目根的 `run_server.py`（服务器入口）严格区分。

## 文件清单

| 文件 | 用途 |
|---|---|
| [`transcribe_media.py`](transcribe_media.py) | 独立 CLI 工具：传入一个音视频文件 → 输出 JSON 转录结果。本身是个完整的 WebSocket 客户端实现，可作集成参考。 |

## 用法

确保 funasr-server 已经在跑（默认 prod 端口 8767，dev 端口 8867）：

```bash
# 用默认配置（连接本机 8767）
venv/bin/python examples/transcribe_media.py audio.mp3

# 指定输出文件
venv/bin/python examples/transcribe_media.py video.mp4 result.json
```

## 编写自己客户端的参考

更系统的 WebSocket 协议文档见 [docs/使用/客户端交互指南.md](../docs/使用/客户端交互指南.md)。
