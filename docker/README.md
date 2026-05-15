# Docker 配置（社区参考，未官方支持）

⚠️ **本目录下的 Dockerfile / docker-compose.yml 在本项目的生产环境从不使用。**

## 为什么不能用于生产

本项目核心依赖 Apple Silicon 的 MPS GPU 加速进行 FunASR 推理。
**Metal / CoreML / Vision 是 macOS 宿主机级 API，无法穿透容器**。
强行容器化 = 退回 CPU，性能 5-10× 倒退（60s 音频从 ~10s 退到 ~60-100s）。

详见项目根 [CLAUDE.md](../CLAUDE.md) 「部署约定」章节。

## 为什么仍然保留

- 项目开源后，可能有人在 **Linux + CPU** 环境下尝试部署
- 保留 Dockerfile 作为社区起点，**不承诺维护或测试**

## 已知限制

| 维度 | 状态 |
|---|---|
| macOS 上跑 | ❌ 不可用（MPS 不能穿透） |
| Linux + CPU | ⚠️ 理论可行，未官方测试 |
| Linux + CUDA | ⚠️ 未测，需自行替换 `requirements.txt` 中 PyTorch CPU 版为 CUDA 版 |
| Windows + Docker Desktop | ⚠️ 未测 |
| 维护状态 | 历史保留，社区可贡献 |

## 如果你要尝试

```bash
# 从项目根目录跑
cd .. && docker build -t funasr-server -f docker/Dockerfile .
cd .. && docker-compose -f docker/docker-compose.yml up -d
```

注意：
- 跑之前需要修改 `docker/Dockerfile` 中的 base image / Python 版本以适配你的环境
- 跑之前需要修改 `requirements.txt` 中跟 MPS 相关的 PyTorch 配置
- 如能跑通，欢迎提 PR 完善 README
