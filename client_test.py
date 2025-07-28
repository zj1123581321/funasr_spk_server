"""
FunASR转录服务客户端测试脚本
"""
import asyncio
import json
import base64
import hashlib
import os
from datetime import datetime
import websockets
from loguru import logger


class FunASRClient:
    """FunASR客户端"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.auth_token = None
    
    async def connect(self, username: str = "admin", password: str = "admin123"):
        """连接到服务器"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"已连接到服务器: {self.server_url}")
            
            # 等待认证请求
            message = await self.websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "auth_required":
                # 获取认证令牌
                self.auth_token = await self._get_auth_token(username, password)
                
                # 发送认证信息
                await self.send_message("auth", {"token": self.auth_token})
                
                # 等待认证结果
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "auth_success":
                    logger.success("认证成功")
                else:
                    raise Exception("认证失败")
            
            # 接收欢迎消息
            message = await self.websocket.recv()
            data = json.loads(message)
            
            if data.get("type") == "connected":
                logger.info(f"连接成功: {data['data']['message']}")
                return True
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def _get_auth_token(self, username: str, password: str) -> str:
        """获取认证令牌（简化版，实际应该调用认证API）"""
        from src.utils.auth import authenticate_user, create_access_token
        
        user = authenticate_user(username, password)
        if not user:
            raise Exception("用户名或密码错误")
        
        token = create_access_token({"sub": username})
        return token
    
    async def send_message(self, msg_type: str, data: dict):
        """发送消息"""
        message = {
            "type": msg_type,
            "data": data
        }
        await self.websocket.send(json.dumps(message))
    
    async def receive_message(self) -> dict:
        """接收消息"""
        message = await self.websocket.recv()
        return json.loads(message)
    
    async def transcribe_file(self, file_path: str, force_refresh: bool = False) -> dict:
        """转录文件"""
        if not os.path.exists(file_path):
            raise Exception(f"文件不存在: {file_path}")
        
        # 读取文件
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        file_name = os.path.basename(file_path)
        file_size = len(file_data)
        file_hash = hashlib.md5(file_data).hexdigest()
        
        logger.info(f"准备上传文件: {file_name} (大小: {file_size/1024/1024:.2f}MB)")
        
        # 发送上传请求
        await self.send_message("upload_request", {
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "force_refresh": force_refresh
        })
        
        # 等待响应
        response = await self.receive_message()
        
        if response["type"] == "error":
            raise Exception(f"上传请求失败: {response['data']['message']}")
        
        if response["type"] != "upload_ready":
            raise Exception(f"意外的响应: {response['type']}")
        
        task_id = response["data"]["task_id"]
        logger.info(f"任务ID: {task_id}")
        
        # 发送文件数据
        file_data_base64 = base64.b64encode(file_data).decode('utf-8')
        await self.send_message("upload_data", {
            "task_id": task_id,
            "file_data": file_data_base64
        })
        
        # 等待处理结果
        result = await self._wait_for_result(task_id)
        
        return result
    
    async def _wait_for_result(self, task_id: str) -> dict:
        """等待处理结果"""
        while True:
            message = await self.receive_message()
            msg_type = message["type"]
            data = message["data"]
            
            if msg_type == "task_progress":
                if data["task_id"] == task_id:
                    logger.info(f"进度: {data['progress']:.1f}% - {data.get('message', '')}")
            
            elif msg_type == "task_complete":
                if data["task_id"] == task_id:
                    logger.success("转录完成!")
                    return data["result"]
            
            elif msg_type == "error":
                logger.error(f"错误: {data['message']}")
                raise Exception(data['message'])
            
            elif msg_type == "upload_complete":
                logger.info("文件上传完成，等待处理...")
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            logger.info("已断开连接")


async def test_transcription():
    """测试转录功能"""
    client = FunASRClient()
    
    try:
        # 连接到服务器
        connected = await client.connect()
        if not connected:
            return
        
        # 获取样本文件
        samples_dir = "samples"
        if not os.path.exists(samples_dir):
            logger.error(f"样本目录不存在: {samples_dir}")
            return
        
        # 查找样本文件
        sample_files = []
        for ext in ['.wav', '.mp3', '.mp4']:
            sample_files.extend([
                os.path.join(samples_dir, f) 
                for f in os.listdir(samples_dir) 
                if f.endswith(ext)
            ])
        
        if not sample_files:
            logger.error("没有找到样本文件")
            return
        
        # 测试第一个文件
        test_file = sample_files[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"测试文件: {test_file}")
        logger.info(f"{'='*60}")
        
        # 执行转录
        result = await client.transcribe_file(test_file)
        
        # 显示结果
        logger.info(f"\n转录结果:")
        logger.info(f"文件名: {result['file_name']}")
        logger.info(f"时长: {result['duration']:.2f}秒")
        logger.info(f"处理时间: {result['processing_time']:.2f}秒")
        logger.info(f"说话人: {', '.join(result['speakers'])}")
        logger.info(f"片段数: {len(result['segments'])}")
        
        # 显示前5个片段
        logger.info(f"\n前5个转录片段:")
        for i, segment in enumerate(result['segments'][:5]):
            logger.info(
                f"{i+1}. [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] "
                f"{segment['speaker']}: {segment['text']}"
            )
        
        if len(result['segments']) > 5:
            logger.info(f"... 还有 {len(result['segments']) - 5} 个片段")
        
        # 保存结果到文件
        output_file = f"result_{os.path.basename(test_file)}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        await client.disconnect()


async def test_batch_transcription():
    """批量测试转录"""
    client = FunASRClient()
    
    try:
        # 连接到服务器
        connected = await client.connect()
        if not connected:
            return
        
        # 获取所有样本文件
        samples_dir = "samples"
        sample_files = []
        for ext in ['.wav', '.mp3', '.mp4']:
            sample_files.extend([
                os.path.join(samples_dir, f) 
                for f in os.listdir(samples_dir) 
                if f.endswith(ext)
            ])
        
        logger.info(f"找到 {len(sample_files)} 个样本文件")
        
        # 批量转录
        for i, file_path in enumerate(sample_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"处理文件 {i+1}/{len(sample_files)}: {os.path.basename(file_path)}")
            logger.info(f"{'='*60}")
            
            try:
                result = await client.transcribe_file(file_path)
                logger.success(f"转录成功: {len(result['segments'])}个片段")
            except Exception as e:
                logger.error(f"转录失败: {e}")
        
    except Exception as e:
        logger.error(f"批量测试失败: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import sys
    
    # 设置日志级别
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # 运行测试
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(test_batch_transcription())
    else:
        asyncio.run(test_transcription())