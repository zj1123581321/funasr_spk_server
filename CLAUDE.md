Always respond in 中文

# 代码设计上的要求
- 各部分功能尽量低耦合，高内聚。
- 各个函数代码做好注释。
- 有完备的日志系统，方便后期调试确认问题。
- 避免写冗余代码，提高项目的可复用性。
- 尽力遵循工程上的最佳实践。适当使用文件夹来增加整个项目的可读性，但不要添加过多无关文件。
- 及时完善 .gitignore 文件。

# 测试约定（PR1 之后启用）

## pytest 套件
- 测试目录三层：
  - `tests/unit/` — 单元测试（mock 外部依赖，毫秒级）
  - `tests/integration/` — 端到端集成（含真实 FunASR 模型）
  - `tests/manual/` — 历史手工脚本（**不在 pytest 收集范围**），仅作复现参考
- 推荐命令：`venv/bin/python -m pytest`（venv 的 pytest binary shebang 漂移，用 `-m pytest` 走 venv python）
- integration 默认 skip，需 `FUNASR_RUN_INTEGRATION=1` 才跑

## TDD 流程
- 新功能 / 改 bug 都先写测试再改代码
- 红 → 绿 → commit 是最小单位，**不要积累多个改动一次性提交**
- 每个 commit message 写清楚改了什么 + 解决什么问题

## Parity 测试
- 改动 FunASR 路径（schemas / database / task_manager / websocket_handler / funasr_transcriber）后必跑：
  ```
  FUNASR_RUN_INTEGRATION=1 venv/bin/python -m pytest tests/integration/
  ```
- 通过 = 改动安全；失败 = 你改坏了 FunASR 路径，回去查
- golden baseline 在 `tests/fixtures/golden/`，由首次运行自动生成

# ASR 引擎抽象（PR1 落地）

## 现状
- `src/core/transcriber_dispatch.py` 是 30 行薄 dispatch 函数（不是 ABC 抽象）
- 支持 `funasr`（生产）和 `qwen3`（占位）
- 引擎选择优先级：`upload_request.engine` > `config.transcription.default_engine` (`FUNASR_DEFAULT_ENGINE`) > `funasr`
- 缓存 key 已按 engine 区分

## 设计原则
- PR1 阶段**不引入** ABC 抽象 / factory / contract test 体系（避免在 Qwen3 可行性未验证前过早抽象）
- Qwen3 真要落地前先跑 `spikes/qwen3_spike.py` 验证
- PR2 是否触发由 Qwen3 spike 结果决定，详见 `docs/开发/重构计划-ASR引擎抽象.md` 第 8 节

## 加新引擎的步骤（PR1 阶段）
1. 在 `src/core/` 加 `<engine>_transcriber.py`，提供 `get_<engine>_transcriber()` 单例工厂
2. 在 `src/core/transcriber_dispatch.py` 的 `resolve_transcriber()` 加 `if name == "<engine>": ...` 分支
3. 加 unit test 到 `tests/unit/test_transcriber_dispatch.py`
4. 跑 parity 确认 FunASR 路径无回归

# 部署约定（macOS only）

本项目仅在 macOS Apple Silicon 上运行（依赖 MPS GPU 加速）。**不要调用全局 `docker-deploy` skill**。

prod/dev 物理隔离：

| 环境 | 目录 | 端口 | 守护 |
|---|---|---|---|
| prod | `~/Production/funasr_spk_server/` | 8767 | **PM2** (`funasr-server`) |
| dev | `~/Dev/projects/250729_funasr_spk_server/funasr_spk_server/` | 8867 | **不挂 PM2**（前台直跑） |

## prod 部署
```bash
cd ~/Production/funasr_spk_server
git pull origin main
venv/bin/pip install -r requirements.txt   # 仅 requirements 有变化时
pm2 restart funasr-server                  # 启动数据库自动迁移
```

## dev 运行
默认前台直跑，方便看日志和调试：
```bash
venv/bin/python run_server.py
```
仅在需要长跑调试时才用 PM2：`pm2 start ecosystem.config.cjs`，用完 `pm2 delete funasr-server-dev && pm2 save`。

详细部署文档：`docs/部署.md`
