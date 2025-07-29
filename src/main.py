"""
FunASR转录服务器主程序
"""
import asyncio
import signal
import sys
from pathlib import Path
import websockets
from loguru import logger

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from src.core.database import db_manager
from src.core.task_manager import task_manager
# 使用FunASR转录器
from src.core.funasr_transcriber import transcriber
logger.info("使用FunASR转录器")
from src.api.websocket_handler import ws_handler
from src.utils.notification import send_custom_notification


class FunASRServer:
    """FunASR转录服务器"""
    
    def __init__(self):
        self.server = None
        self.is_running = False
    
    async def start(self):
        """启动服务器"""
        try:
            logger.info("正在启动FunASR转录服务器...")
            
            # 初始化数据库
            await db_manager.init_db()
            
            # 初始化转录器
            logger.info("初始化FunASR模型...")
            await transcriber.initialize()
            
            # 启动任务管理器
            await task_manager.start()
            
            # 启动WebSocket服务器
            self.server = await websockets.serve(
                ws_handler.handle_connection,
                config.server.host,
                config.server.port,
                max_size=config.server.max_file_size_mb * 1024 * 1024,
                max_queue=config.server.max_connections,
                ping_interval=30,  # 每30秒发送一次 ping
                ping_timeout=60,   # ping 响应超时60秒（给客户端更多处理时间）
                close_timeout=60   # 关闭连接超时60秒
            )
            
            self.is_running = True
            
            logger.success(f"服务器已启动: ws://{config.server.host}:{config.server.port}")
            
            # 发送启动通知
            await send_custom_notification(
                "🚀 FunASR转录服务器已启动",
                f"地址: ws://{config.server.host}:{config.server.port}\n"
                f"最大并发: {config.transcription.max_concurrent_tasks}\n"
                f"认证: {'启用' if config.auth.enabled else '禁用'}",
                "markdown"
            )
            
            # 定期清理任务
            asyncio.create_task(self._periodic_cleanup())
            
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务器"""
        logger.info("正在停止服务器...")
        
        self.is_running = False
        
        # 停止任务管理器
        await task_manager.stop()
        
        # 关闭WebSocket服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # 发送停止通知
        await send_custom_notification(
            "🛑 FunASR转录服务器已停止",
            "服务器已正常关闭",
            "text"
        )
        
        logger.info("服务器已停止")
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        while self.is_running:
            try:
                # 清理过期缓存
                await db_manager.clean_old_cache()
                
                # 清理临时文件
                from src.utils.file_utils import cleanup_temp_files
                await cleanup_temp_files()
                
                # 获取统计信息
                stats = task_manager.get_stats()
                cache_stats = await db_manager.get_cache_stats()
                
                logger.info(f"系统状态 - 任务: {stats}, 缓存: {cache_stats}")
                
            except Exception as e:
                logger.error(f"定期清理失败: {e}")
            
            # 每小时执行一次
            await asyncio.sleep(3600)


async def main():
    """主函数"""
    server = FunASRServer()
    
    # 设置信号处理
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在关闭服务器...")
        asyncio.create_task(server.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动服务器
        await server.start()
        
        # 保持运行
        while server.is_running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"服务器错误: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    # Windows平台特殊处理
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主程序
    asyncio.run(main())