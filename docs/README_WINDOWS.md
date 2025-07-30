# FunASR转录服务器 Windows 部署指南

## 快速开始

### 方法一：使用管理工具（推荐）
双击运行根目录的 `manage.bat`，选择相应功能：
- 启动/停止服务器
- 设置开机自启
- 查看状态和日志
- 清理临时文件

### 方法二：直接启动
1. **一键启动**：双击根目录的 `start.bat`
2. **后台启动**：双击根目录的 `start_background.bat`
3. **停止服务器**：双击 `windows_scripts\stop_server.bat`

## 开机自启设置

### 自动设置（推荐）
1. 右键点击PowerShell，选择"以管理员身份运行"
2. 执行以下命令：
   ```powershell
   cd "项目目录路径"
   .\windows_scripts\setup_autostart.ps1 -Install
   ```

### 手动设置
1. **方法一：任务计划程序**
   - Win+R 打开运行，输入 `taskschd.msc`
   - 创建基本任务，设置在启动时运行 `start_server_background.bat`

2. **方法二：启动文件夹**
   - Win+R 打开运行，输入 `shell:startup`
   - 将 `start_background.bat` 的快捷方式放入此文件夹

## 文件说明

| 文件名 | 功能 |
|--------|------|
| `start.bat` | 启动服务器（显示命令行窗口） |
| `start_background.bat` | 后台启动服务器（无窗口） |
| `manage.bat` | 服务器管理工具（图形界面） |
| `windows_scripts/` | Windows脚本文件夹 |
| `windows_scripts/start_server.bat` | 实际的启动脚本 |
| `windows_scripts/stop_server.bat` | 停止服务器 |
| `windows_scripts/setup_autostart.ps1` | 开机自启管理脚本 |

## 系统要求

- Windows 10/11
- Python 3.10 或更高版本
- 至少 8GB 内存（推荐 16GB）
- 足够的磁盘空间存储模型文件

## 常见问题

### 1. 服务器启动失败
- 检查Python是否正确安装并在PATH中
- 确认虚拟环境和依赖是否正确安装
- 查看 `logs` 目录中的错误日志

### 2. 开机自启失败
- 确保以管理员权限运行设置脚本
- 检查Windows防火墙和杀毒软件设置
- 使用 `windows_scripts\setup_autostart.ps1 -Status` 查看状态

### 3. 性能优化
- 在 `config.json` 中调整以下参数：
  - `ncpu`: CPU核心数（默认16）
  - `batch_size_s`: 批处理大小（默认500）
  - `max_concurrent_tasks`: 最大并发任务数（默认4）

### 4. 防火墙设置
如果需要外部访问，请在Windows防火墙中：
1. 允许Python通过防火墙
2. 开放端口8767（或config.json中配置的端口）

## 目录结构

```
funasr_spk_server/
├── start.bat                     # 快捷启动脚本
├── start_background.bat          # 快捷后台启动脚本  
├── manage.bat                    # 快捷管理工具
├── windows_scripts/              # Windows脚本文件夹
│   ├── start_server.bat          # 实际启动脚本
│   ├── start_server_background.bat # 后台启动脚本
│   ├── start_server_silent.vbs   # VBS辅助脚本
│   ├── stop_server.bat           # 停止脚本
│   ├── setup_autostart.ps1       # 开机自启管理
│   └── manage.bat                # 管理工具
├── config.json                   # 配置文件
├── src/                          # 源代码
├── logs/                         # 日志文件
├── temp/                         # 临时文件
├── uploads/                      # 上传文件
├── models/                       # 模型文件
└── venv/                         # Python虚拟环境
```

## 监控和维护

### 查看服务状态
```cmd
# 查看Python进程
tasklist /FI "IMAGENAME eq python.exe"

# 查看端口占用
netstat -an | findstr 8767
```

### 日志文件
- 服务器日志：`logs/` 目录
- 实时监控：使用 `manage.bat` 中的日志查看功能

### 性能监控
- 任务管理器中监控CPU和内存使用
- 检查磁盘空间（模型文件和临时文件）

## 卸载

1. 停止服务器
2. 移除开机自启：
   ```powershell
   .\windows_scripts\setup_autostart.ps1 -Uninstall
   ```
3. 删除项目文件夹

## 技术支持

如遇问题，请：
1. 检查 `logs` 目录中的日志文件
2. 使用 `manage.bat` 中的状态检查功能
3. 查看配置文件是否正确