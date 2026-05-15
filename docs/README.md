# 文档索引

本目录文档分三类：**使用** / **开发** / **背景资料**。

## 📘 使用（给 API 调用方看）

| 文档 | 主题 |
|---|---|
| [客户端交互指南](使用/客户端交互指南.md) | WebSocket 协议、单文件/分片上传、Python 客户端完整示例 |
| [部署指南](部署.md) | prod (PM2) / dev (前台) 部署流程、为什么不能 Docker |

## 🛠 开发（给改源码的人看）

| 文档 | 主题 |
|---|---|
| [重构计划-ASR引擎抽象](开发/重构计划-ASR引擎抽象.md) | **PR1 当前方案**（v2，PR1 已完成）：ASR 引擎可插拔架构 |
| [Server-Client 交互协议](开发/Server-Client%20交互协议.md) | 服务端视角：任务状态机、并发控制、错误重试机制 |
| [WebSocket大文件传输最佳实践](开发/WebSocket大文件传输最佳实践.md) | 大文件分片上传方案 + 客户端实现参考 |
| [兼容性开发文档](开发/FunASR音频转文本服务器兼容性开发文档.md) | 给想开发兼容服务器的人看的架构详解 |
| [gpu加速/](开发/gpu加速/) | MPS 加速实施记录 + 长音频补丁方案（3 篇按日期） |

## 📚 背景资料

| 文档 | 主题 |
|---|---|
| [VAD并发问题解决方案](funasr相关/VAD并发问题解决方案.md) | FunASR Python 版 VAD 并发不安全的历史问题及解决路径（pool 模式由来） |

---

## 文档维护原则

- **使用文档**：API/部署方式变化时立即同步
- **开发文档**：架构调整时跟进；历史 ADR（架构决策记录）不删除，按日期归档
- **背景资料**：FunASR 上游内容不复制到本仓库（保持 modelscope/huggingface 官方为权威）

## 历史清理记录（2026-05-15）

- 删除：`README_WINDOWS.md`（项目 mac-only）、`DEPLOYMENT.md`（被 `部署.md` 替代）
- 删除：`funasr相关/funasr readme.md` + `non-streaming.md`（FunASR 上游冗余内容）
- 删除：`funasr python 并发错误问题/funasr-onnx-offline-vad.cpp`（无关 C++ 片段）
- 移动：`使用/server-client-interaction.md` → `开发/Server-Client 交互协议.md`（受众是服务端开发者）
- 新建：`部署.md` + `README.md` 索引
