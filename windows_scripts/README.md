# Windows 脚本（社区参考，未官方支持）

⚠️ **本目录下脚本是项目早期 Windows 支持遗留，当前 mac-only 维护，不保证可用。**

## 文件说明

| 文件 | 作用 |
|---|---|
| `start_server.bat` | 启动服务器（显示命令行窗口） |
| `start_server_background.bat` | 后台启动 |
| `start_server_silent.vbs` | VBS 无窗口启动辅助 |
| `stop_server.bat` | 停止服务器 |
| `manage.bat` | 管理工具菜单 |
| `setup_autostart.ps1` | 开机自启 PowerShell 管理脚本 |

项目根目录的 `start.bat` / `start_background.bat` / `manage.bat` 是本目录脚本的 1 行 wrapper，方便 Windows 用户在项目根双击启动。

## 已知限制

| 维度 | 状态 |
|---|---|
| Apple Silicon 上的 MPS 加速 | ❌ Windows 不支持，会退回 CPU |
| 测试覆盖 | ❌ 当前迭代未在 Windows 上测试过 |
| 维护承诺 | 无；功能性 bug 可能存在，欢迎社区 PR |

## 如果你在 Windows 上跑

1. 装 Python 3.11.9 + ffmpeg
2. 项目根目录跑：`venv\Scripts\python.exe -m venv venv` 创建虚拟环境
3. 装依赖：`venv\Scripts\pip install -r requirements.txt`
4. 改 `.env` 配置端口、关闭 MPS 优先：`FUNASR_DEVICE_PRIORITY=cpu`
5. 启动：在项目根双击 `start.bat`，或直接 `venv\Scripts\python.exe run_server.py`

如能跑通，欢迎反馈或提 PR。
