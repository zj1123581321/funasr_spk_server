"""
Vendor 三方代码命名空间。

集成进主项目的、不修改源码的第三方代码放在这里,目的:
- 避免污染 src/core/ 顶层
- 保留 upstream 原始结构,方便后续 sync upstream
- 通过本 __init__ 暴露主项目共享资源(如 logger),让 vendor 包能复用,不需要私建

当前 vendor:
- qwen_asr_gguf — CapsWriter-Offline 的 GGUF/ONNX 混合 ASR 引擎 (HaujetZhao/CapsWriter-Offline)
"""
from loguru import logger as _logger

# qwen_asr_gguf/__init__.py 末尾有:
#   try: from .. import logger
#   except: logger = setup_logging(...)
# 我们在这里暴露 logger,避免 vendor 引擎自建文件 handler 把日志写到 src/core/logs/
logger = _logger
