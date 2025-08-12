#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立音视频转录脚本

基于 FunASR 服务器进行音视频文件转录，生成 JSON 格式的转录结果。

使用方法:
    python transcribe_media.py <音视频文件路径> [输出文件路径]

参数:
    - 音视频文件路径: 要转录的音频或视频文件路径
    - 输出文件路径: 可选，指定输出的 JSON 文件路径，默认为原文件名_transcription.json

示例:
    python transcribe_media.py audio.mp3
    python transcribe_media.py video.mp4 result.json
"""

import os
import sys
import json
import time
import asyncio
import websockets
import hashlib
import base64
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger


class MediaTranscriber:
    """音视频转录器"""
    
    def __init__(self, server_url="ws://localhost:8767"):
        """
        初始化转录器
        
        Args:
            server_url: WebSocket 服务器地址
        """
        self.server_url = server_url
        self.websocket = None
        
    async def connect_to_server(self):
        """连接到转录服务器"""
        try:
            logger.info(f"正在连接到转录服务器: {self.server_url}")
            
            # 连接 WebSocket 服务器（优化大文件传输配置）
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=60,   # 60秒发送一次心跳
                ping_timeout=120,   # 心跳响应超时120秒
                close_timeout=60,   # 关闭连接超时60秒
                max_size=10 * 1024 * 1024,  # 单消息最大10MB
                # 增加读写缓冲区
                read_limit=2**20,   # 1MB读缓冲
                write_limit=2**20   # 1MB写缓冲
            )
            
            # 接收服务器连接确认消息
            welcome_message = await self.receive_message(timeout=10)
            if welcome_message.get("type") == "connected":
                logger.info("服务器连接成功")
                return True
            else:
                logger.warning(f"意外的服务器响应: {welcome_message.get('type')}")
                return True
                
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            return False
    
    async def disconnect_from_server(self):
        """断开服务器连接"""
        if self.websocket:
            await self.websocket.close()
            logger.info("已断开服务器连接")
    
    def calculate_file_hash(self, file_path):
        """
        计算文件 MD5 哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: MD5 哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def send_message(self, message):
        """
        发送消息到服务器
        
        Args:
            message: 要发送的消息字典
        """
        if not self.websocket:
            raise Exception("未连接到服务器")
        
        message_json = json.dumps(message, ensure_ascii=False)
        await self.websocket.send(message_json)
        logger.debug(f"发送消息: {message['type']}")
    
    async def receive_message(self, timeout=30):
        """
        接收服务器消息
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            dict: 接收到的消息字典
        """
        if not self.websocket:
            raise Exception("未连接到服务器")
        
        try:
            message_json = await asyncio.wait_for(
                self.websocket.recv(), timeout=timeout
            )
            message = json.loads(message_json)
            logger.debug(f"接收消息: {message.get('type', 'unknown')}")
            return message
        except asyncio.TimeoutError:
            raise Exception(f"接收消息超时 ({timeout}秒)")
    
    async def transcribe_file(self, audio_path, output_format="json", force_refresh=False):
        """
        转录音视频文件
        
        Args:
            audio_path: 音视频文件路径
            output_format: 输出格式，支持 'json' 或 'srt'
            force_refresh: 是否强制刷新缓存
            
        Returns:
            dict: 转录结果
        """
        logger.info(f"开始转录文件: {os.path.basename(audio_path)}")
        start_time = time.time()
        
        # 验证文件存在性
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"文件不存在: {audio_path}")
        
        file_name = os.path.basename(audio_path)
        file_size = os.path.getsize(audio_path)
        file_hash = self.calculate_file_hash(audio_path)
        
        logger.info(f"文件信息: {file_name}, 大小: {file_size/1024/1024:.2f}MB, 哈希: {file_hash[:8]}...")
        
        # 判断是否使用分片上传（大于5MB的文件）
        chunk_threshold = 5 * 1024 * 1024  # 5MB
        use_chunked_upload = file_size > chunk_threshold
        
        if use_chunked_upload:
            logger.info(f"文件较大（{file_size/1024/1024:.2f}MB），使用分片上传")
            return await self._transcribe_file_chunked(audio_path, file_name, file_size, file_hash, output_format, force_refresh, start_time)
        else:
            logger.info(f"文件较小（{file_size/1024/1024:.2f}MB），使用单文件上传")
            return await self._transcribe_file_single(audio_path, file_name, file_size, file_hash, output_format, force_refresh, start_time)
    
    async def _transcribe_file_single(self, audio_path, file_name, file_size, file_hash, output_format, force_refresh, start_time):
        """
        单文件上传转录
        """
        # 读取音频文件
        with open(audio_path, 'rb') as f:
            file_data = f.read()
        
        file_data_b64 = base64.b64encode(file_data).decode('utf-8')
        
        # 1. 发送上传请求
        upload_request = {
            "type": "upload_request",
            "data": {
                "file_name": file_name,
                "file_size": file_size,
                "file_hash": file_hash,
                "force_refresh": force_refresh,
                "output_format": output_format
            }
        }
        
        await self.send_message(upload_request)
        response = await self.receive_message()
        
        if response["type"] == "error":
            raise Exception(f"上传请求失败: {response['data']['message']}")
        
        # 处理不同的响应类型
        task_id = None
        
        if response["type"] == "upload_ready":
            task_id = response["data"]["task_id"]
            logger.info(f"获得任务ID: {task_id}")
            
            # 2. 上传文件数据
            upload_data = {
                "type": "upload_data",
                "data": {
                    "task_id": task_id,
                    "file_data": file_data_b64
                }
            }
            
            await self.send_message(upload_data)
            response = await self.receive_message()
            
            if response["type"] == "error":
                raise Exception(f"文件上传失败: {response['data']['message']}")
            elif response["type"] == "upload_complete":
                logger.info("文件上传成功，开始转录...")
            
        elif response["type"] == "task_complete":
            # 直接返回缓存结果
            logger.info("使用缓存结果，转录完成")
            transcription_result = response["data"]["result"]
            processing_time = time.time() - start_time
            
            return {
                "file_name": file_name,
                "file_size": file_size,
                "file_hash": file_hash,
                "task_id": response["data"].get("task_id", "cached"),
                "processing_time": processing_time,
                "transcription_result": transcription_result,
                "cached_result": True
            }
        
        # 3. 等待转录结果
        transcription_result = None
        
        while True:
            try:
                response = await self.receive_message(timeout=300)  # 5分钟超时
                
                if response["type"] == "task_progress":
                    progress = response["data"]["progress"]
                    logger.info(f"转录进度: {progress}%")
                
                elif response["type"] == "transcription_progress":
                    progress = response["data"]["progress"]
                    logger.info(f"转录进度: {progress}%")
                
                elif response["type"] == "task_complete":
                    transcription_result = response["data"]["result"]
                    logger.info("转录完成")
                    break
                
                elif response["type"] == "transcription_complete":
                    transcription_result = response["data"]
                    logger.info("转录完成")
                    break
                
                elif response["type"] == "error":
                    raise Exception(f"转录失败: {response['data']['message']}")
                
            except Exception as e:
                logger.error(f"等待转录结果时出错: {e}")
                raise
        
        processing_time = time.time() - start_time
        
        # 验证结果
        if not transcription_result:
            raise Exception("未收到转录结果")
        
        logger.info(f"转录完成: {len(transcription_result.get('segments', []))} 个片段, "
                   f"{len(transcription_result.get('speakers', []))} 个说话人, "
                   f"总耗时 {processing_time:.2f}秒")
        
        return {
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "task_id": task_id,
            "processing_time": processing_time,
            "transcription_result": transcription_result,
            "cached_result": False
        }
    
    async def _transcribe_file_chunked(self, audio_path, file_name, file_size, file_hash, output_format, force_refresh, start_time):
        """
        分片上传转录大文件
        """
        chunk_size = 1024 * 1024  # 1MB分片大小
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        
        logger.info(f"分片信息: 分片大小={chunk_size/1024/1024:.2f}MB, 总分片数={total_chunks}")
        
        # 1. 发送分片上传请求
        upload_request = {
            "type": "upload_request",
            "data": {
                "file_name": file_name,
                "file_size": file_size,
                "file_hash": file_hash,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
                "upload_mode": "chunked",  # 标识分片上传
                "output_format": output_format,
                "force_refresh": force_refresh
            }
        }
        
        await self.send_message(upload_request)
        response = await self.receive_message()
        
        if response["type"] == "error":
            raise Exception(f"上传请求失败: {response['data']['message']}")
        
        # 处理不同的响应类型
        task_id = None
        
        if response["type"] == "task_complete":
            # 直接返回缓存结果
            logger.info("使用缓存结果，转录完成")
            transcription_result = response["data"]["result"]
            processing_time = time.time() - start_time
            
            return {
                "file_name": file_name,
                "file_size": file_size,
                "file_hash": file_hash,
                "task_id": response["data"].get("task_id", "cached"),
                "processing_time": processing_time,
                "transcription_result": transcription_result,
                "cached_result": True
            }
        
        if response["type"] != "upload_ready":
            raise Exception(f"未知的响应类型: {response['type']}")
        
        task_id = response["data"]["task_id"]
        logger.info(f"获得任务ID: {task_id}，开始分片上传")
        
        # 2. 分片读取和上传
        with open(audio_path, 'rb') as f:
            for chunk_index in range(total_chunks):
                # 读取分片数据
                chunk_data = f.read(chunk_size)
                chunk_hash = hashlib.md5(chunk_data).hexdigest()
                
                # 发送分片
                chunk_message = {
                    "type": "upload_chunk",
                    "data": {
                        "task_id": task_id,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_data),
                        "chunk_hash": chunk_hash,
                        "chunk_data": base64.b64encode(chunk_data).decode(),
                        "is_last": chunk_index == total_chunks - 1
                    }
                }
                
                await self.send_message(chunk_message)
                
                # 等待分片确认
                chunk_response = await self.receive_message(timeout=60)  # 60秒超时
                if chunk_response["type"] != "chunk_received":
                    raise Exception(f"分片 {chunk_index} 上传失败: {chunk_response.get('type', 'unknown')}")
                
                # 显示进度
                progress = (chunk_index + 1) / total_chunks * 100
                logger.info(f"上传进度: {progress:.1f}% ({chunk_index + 1}/{total_chunks})")
        
        logger.info("文件分片上传完成，等待处理结果...")
        
        # 3. 等待转录结果
        transcription_result = None
        
        while True:
            try:
                response = await self.receive_message(timeout=300)  # 5分钟超时
                
                if response["type"] == "task_progress":
                    progress = response["data"]["progress"]
                    logger.info(f"转录进度: {progress}%")
                
                elif response["type"] == "transcription_progress":
                    progress = response["data"]["progress"]
                    logger.info(f"转录进度: {progress}%")
                
                elif response["type"] == "task_complete":
                    transcription_result = response["data"]["result"]
                    logger.info("转录完成")
                    break
                
                elif response["type"] == "transcription_complete":
                    transcription_result = response["data"]
                    logger.info("转录完成")
                    break
                
                elif response["type"] == "upload_complete":
                    # 文件上传完成，继续等待转录结果
                    logger.info("文件上传完成，开始转录...")
                    continue
                
                elif response["type"] == "task_queued":
                    # 任务排队中
                    queue_position = response["data"].get("queue_position", "N/A")
                    estimated_wait = response["data"].get("estimated_wait_minutes", "N/A")
                    logger.info(f"任务排队中，位置: {queue_position}，预计等待: {estimated_wait}分钟")
                    continue
                
                elif response["type"] == "error":
                    raise Exception(f"转录失败: {response['data']['message']}")
                
            except Exception as e:
                logger.error(f"等待转录结果时出错: {e}")
                raise
        
        processing_time = time.time() - start_time
        
        # 验证结果
        if not transcription_result:
            raise Exception("未收到转录结果")
        
        logger.info(f"转录完成: {len(transcription_result.get('segments', []))} 个片段, "
                   f"{len(transcription_result.get('speakers', []))} 个说话人, "
                   f"总耗时 {processing_time:.2f}秒")
        
        return {
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "task_id": task_id,
            "processing_time": processing_time,
            "transcription_result": transcription_result,
            "cached_result": False
        }
    
    def save_transcription_result(self, result, output_path):
        """
        保存转录结果到文件
        
        Args:
            result: 转录结果字典
            output_path: 输出文件路径
        """
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 添加转录时间戳
        result["transcription_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存为 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"转录结果已保存到: {output_path}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="独立音视频转录脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python transcribe_media.py audio.mp3
  python transcribe_media.py video.mp4 result.json
  python transcribe_media.py audio.wav --server ws://localhost:8767
  python transcribe_media.py video.mp4 --format srt --force-refresh
        """
    )
    
    parser.add_argument("input_file", help="要转录的音视频文件路径")
    parser.add_argument("output_file", nargs="?", help="输出文件路径（可选）")
    parser.add_argument("--server", default="ws://localhost:8767", 
                       help="服务器地址 (默认: ws://localhost:8767)")
    parser.add_argument("--format", choices=["json", "srt"], default="json",
                       help="输出格式 (默认: json)")
    parser.add_argument("--force-refresh", action="store_true",
                       help="强制刷新缓存，重新进行转录")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    # 验证输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return 1
    
    # 确定输出文件路径 - 默认与音视频文件在同一目录
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # 在音视频文件同一目录下生成输出文件
        if args.format == "json":
            output_path = input_path.parent / f"{input_path.stem}_transcription.json"
        else:  # srt
            output_path = input_path.parent / f"{input_path.stem}.srt"
    
    logger.info("=" * 60)
    logger.info("FunASR 音视频转录脚本")
    logger.info("=" * 60)
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"服务器: {args.server}")
    logger.info(f"输出格式: {args.format}")
    logger.info(f"强制刷新: {args.force_refresh}")
    
    # 创建转录器
    transcriber = MediaTranscriber(server_url=args.server)
    
    try:
        # 连接服务器
        if not await transcriber.connect_to_server():
            logger.error("无法连接到转录服务器")
            return 1
        
        # 执行转录
        result = await transcriber.transcribe_file(
            str(input_path), 
            output_format=args.format,
            force_refresh=args.force_refresh
        )
        
        # 保存结果
        if args.format == "srt":
            # 对于 SRT 格式，直接保存内容
            transcription_data = result.get("transcription_result", {})
            if transcription_data.get("format") == "srt":
                srt_content = transcription_data.get("content", "")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                logger.info(f"SRT 文件已保存到: {output_path}")
            else:
                logger.error("未收到 SRT 格式的转录结果")
                return 1
        else:
            # 保存 JSON 格式结果
            transcriber.save_transcription_result(result, str(output_path))
        
        logger.info("✓ 转录完成")
        return 0
        
    except KeyboardInterrupt:
        logger.info("转录被用户中断")
        return 1
    except Exception as e:
        logger.error(f"转录过程中发生错误: {e}")
        return 1
    finally:
        await transcriber.disconnect_from_server()


if __name__ == "__main__":
    # 运行转录脚本
    exit_code = asyncio.run(main())
    sys.exit(exit_code)