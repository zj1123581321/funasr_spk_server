● 根据我的深入分析,这个 Memo 程序在 Windows 上使用 GPU 加速 Whisper 模型转录的方案如下:

  GPU 加速方案总结

  1. 核心技术栈

  该程序基于 whisper.cpp 和 GGML 框架实现,使用 DirectX 11 (D3D11) Compute Shaders 作为 Windows 平台的 GPU 加速方案。

  2. 关键文件结构

  resources/addon/whisper/
  ├── whisper.dll (913KB)          # CPU 版本
  ├── whisper-addon.node           # Node.js 原生绑定
  ├── gpu/
  │   ├── Whisper.dll (601KB)     # GPU 加速版本
  │   └── main.exe                 # GPU 可执行文件
  ├── cublas/                      # CUDA 版本目录(当前为空)
  └── bin/
      ├── ggml.dll                 # GGML 核心库
      ├── ggml-base.dll            # GGML 基础库
      ├── ggml-cpu.dll             # GGML CPU 后端
      ├── whisper.dll              # Whisper 实现
      ├── whisper-cli.exe          # 命令行工具
      └── whisper-server.exe       # 服务器模式

  3. GPU 加速实现方案

  DirectX 11 Compute Shader 方案 (当前使用)

  - 依赖库: d3d11.dll, dxgi.dll, MF.dll, MFReadWrite.dll
  - 优势:
    - 兼容性好,支持 AMD、Intel、NVIDIA 所有支持 DirectX 11 的显卡
    - 不需要 CUDA 环境
    - Windows 原生支持
  - 实现位置: resources/addon/whisper/gpu/Whisper.dll

  CUDA/cuBLAS 方案 (预留但未启用)

  - 日志显示程序会检测 NVIDIA GPU:
  [error] nvcc --version error: Command failed: nvcc --version
  [error] nvidia-smi error: Command failed: nvidia-smi
  - cublas/ 目录在启动时会被清空,文件被移动到用户目录后删除

  4. GPU 加速检测与切换逻辑

  从日志中可以看到关键流程:

  1. 启动时检测 (logs/2025-08-13.log:127-128):
  C:\Users\zlx\AppData\Local\Programs\Memo\resources\addon\whisper\cublas\whisper-addon.node
  --> C:\Users\zlx\.memo-ai\addon\whisper\win32\x64\cublas
  C:\Users\zlx\AppData\Local\Programs\Memo\resources\addon\whisper\cublas\whisper.dll
  --> C:\Users\zlx\.memo-ai\addon\whisper\win32\x64\cublas

  2. 清理 cublas 文件 (logs/2025-10-10.log:80-81):
  remove: C:\Users\zlx\AppData\Local\Programs\Memo\resources\addon\whisper\cublas\whisper.dll removed successfully
  remove: C:\Users\zlx\AppData\Local\Programs\Memo\resources\addon\whisper\cublas\whisper-addon.node removed successfully

  3. 尝试检测 NVIDIA GPU:
  nvcc --version error
  nvidia-smi error

  5. GPU 加速代码特征

  从 gpu/Whisper.dll 中发现的关键字符串:
  - "GPU Tasks"
  - "Loaded %zu GPU tensors, %g MB VRAM"
  - "eModelImplementation.GPU model requires a CPU with SSE 4.1 and F16C support"
  - "GPU model doesn't implement the SpeedupAudio flag"
  - "Compute Shaders"
  - "D3D11CreateDevice"
  - "listGPUs" / "loadGpu"

  6. GPU 加速策略

  多级加速策略:
  1. 优先: 检测 NVIDIA GPU → 使用 CUDA/cuBLAS (如果可用)
  2. 回退: 使用 DirectX 11 Compute Shaders (通用 GPU 方案)
  3. 最终: 使用 CPU AVX/SSE 指令集

  当前系统状态 (从日志):
  - CPU: AMD Ryzen 7 6800H (16核)
  - GPU: AMD Radeon Graphics (4096MB)
  - 使用方案: DirectX 11 (因为没有 NVIDIA GPU)

  7. 技术要点

  1. 基于 whisper.cpp: 源代码路径显示 F:\Develop\whisper.cpp\ggml\src\
  2. GGML 后端架构: 模块化设计,支持多种计算后端
  3. 动态 GPU 选择: 通过 --use-gpu 参数指定 GPU
  4. 零拷贝优化: 使用 Media Foundation (MF) 进行音视频处理
  5. 模型格式: 使用 GGML 量化格式 (*.bin 文件)

  8. 关键发现

  - 程序不依赖 CUDA,可在任何支持 DX11 的 Windows GPU 上运行
  - GPU 版本 DLL (601KB) 比 CPU 版本 (913KB) 更小,说明优化了代码路径
  - 使用 Compute Shaders 实现通用 GPU 计算,这是跨平台 GPU 加速的现代方案
  - 系统会在启动时动态检测并选择最佳加速方案

  这种设计使得 Whisper 转录可以在没有 NVIDIA 显卡的情况下仍然获得 GPU 加速,是一个非常实用的跨硬件加速方案!