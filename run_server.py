"""
启动FunASR转录服务器
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 添加项目目录到Python路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# 导入并运行主程序
from src.main import main
import asyncio

if __name__ == "__main__":
    # Windows平台特殊处理
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行服务器
    asyncio.run(main())