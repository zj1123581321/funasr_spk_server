#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并发转写能力测试

测试服务器的并发处理能力：
1. 多个客户端同时连接
2. 并发上传和转写多个音频文件
3. 测试服务器的响应时间和吞吐量
4. 验证结果的正确性和一致性
"""

import os
import sys
import json
import time
import asyncio
import websockets
import hashlib
import base64
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import List, Dict, Any, Optional
import statistics

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ConcurrentClient:
    """并发测试客户端"""
    
    def __init__(self, client_id: int, server_url: str = "ws://localhost:8767"):
        self.client_id = client_id
        self.server_url = server_url
        self.websocket = None
        self.results = []
        self.connection_time = None
        self.disconnection_time = None
        
    async def connect(self):
        """连接到服务器"""
        try:
            start_time = time.time()
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=30,
                ping_timeout=60,
                close_timeout=60,
                max_size=100 * 1024 * 1024
            )
            self.connection_time = time.time() - start_time
            
            # 接收欢迎消息
            welcome = await self.receive_message(timeout=10)
            if welcome.get("type") == "connected":
                logger.info(f"客户端 {self.client_id} 连接成功 (耗时: {self.connection_time:.2f}s)")
                return True
            else:
                logger.warning(f"客户端 {self.client_id} 收到意外的欢迎消息: {welcome.get('type')}")
                return False
                
        except Exception as e:
            logger.error(f"客户端 {self.client_id} 连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            start_time = time.time()
            await self.websocket.close()
            self.disconnection_time = time.time() - start_time
            logger.info(f"客户端 {self.client_id} 已断开连接")
    
    async def send_message(self, message: Dict[str, Any]):
        """发送消息"""
        if not self.websocket:
            raise Exception(f"客户端 {self.client_id} 未连接")
        
        await self.websocket.send(json.dumps(message, ensure_ascii=False))
        
    async def receive_message(self, timeout: int = 30) -> Dict[str, Any]:
        """接收消息"""
        if not self.websocket:
            raise Exception(f"客户端 {self.client_id} 未连接")
            
        try:
            message_json = await asyncio.wait_for(
                self.websocket.recv(), timeout=timeout
            )
            return json.loads(message_json)
        except asyncio.TimeoutError:
            raise Exception(f"客户端 {self.client_id} 接收消息超时")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def transcribe_file(self, audio_path: str, force_refresh: bool = False) -> Dict[str, Any]:
        """转写单个文件"""
        start_time = time.time()
        file_name = os.path.basename(audio_path)
        
        logger.info(f"客户端 {self.client_id} 开始转写: {file_name}")
        
        # 读取文件
        with open(audio_path, 'rb') as f:
            file_data = f.read()
        
        file_size = len(file_data)
        file_hash = self.calculate_file_hash(audio_path)
        file_data_b64 = base64.b64encode(file_data).decode('utf-8')
        
        # 1. 发送上传请求
        await self.send_message({
            "type": "upload_request",
            "data": {
                "file_name": file_name,
                "file_size": file_size,
                "file_hash": file_hash,
                "force_refresh": force_refresh
            }
        })
        
        response = await self.receive_message()
        
        if response["type"] == "error":
            raise Exception(f"上传请求失败: {response['data']['message']}")
        
        # 处理缓存结果
        if response["type"] == "task_complete":
            logger.info(f"客户端 {self.client_id} 使用缓存结果: {file_name}")
            processing_time = time.time() - start_time
            return {
                "client_id": self.client_id,
                "file_name": file_name,
                "file_size": file_size,
                "task_id": response["data"].get("task_id", "cached"),
                "processing_time": processing_time,
                "server_processing_time": response["data"]["result"].get("processing_time", 0),
                "cached": True,
                "success": True,
                "segments_count": len(response["data"]["result"].get("segments", [])),
                "speakers_count": len(response["data"]["result"].get("speakers", []))
            }
        
        # 2. 上传文件数据
        task_id = response["data"]["task_id"]
        
        await self.send_message({
            "type": "upload_data",
            "data": {
                "task_id": task_id,
                "file_data": file_data_b64
            }
        })
        
        response = await self.receive_message()
        
        if response["type"] == "error":
            raise Exception(f"文件上传失败: {response['data']['message']}")
        
        # 3. 等待转写结果
        while True:
            response = await self.receive_message(timeout=300)
            
            if response["type"] in ["task_progress", "transcription_progress"]:
                progress = response["data"]["progress"]
                logger.debug(f"客户端 {self.client_id} 转写进度: {progress}%")
                
            elif response["type"] in ["task_complete", "transcription_complete"]:
                processing_time = time.time() - start_time
                result = response["data"].get("result", response["data"])
                
                logger.info(f"客户端 {self.client_id} 转写完成: {file_name} (耗时: {processing_time:.2f}s)")
                
                return {
                    "client_id": self.client_id,
                    "file_name": file_name,
                    "file_size": file_size,
                    "task_id": task_id,
                    "processing_time": processing_time,
                    "server_processing_time": result.get("processing_time", 0),
                    "cached": False,
                    "success": True,
                    "segments_count": len(result.get("segments", [])),
                    "speakers_count": len(result.get("speakers", []))
                }
                
            elif response["type"] == "error":
                raise Exception(f"转写失败: {response['data']['message']}")


class ConcurrentTranscriptionTester:
    """并发转写测试器"""
    
    def __init__(self, server_url: str = "ws://localhost:8767", num_clients: int = 4):
        self.server_url = server_url
        self.num_clients = num_clients
        self.clients: List[ConcurrentClient] = []
        self.test_results = {
            "connection_times": [],
            "transcription_results": [],
            "errors": []
        }
        
    async def setup_clients(self):
        """创建并连接所有客户端"""
        logger.info(f"创建 {self.num_clients} 个并发客户端...")
        
        # 创建客户端实例
        self.clients = [
            ConcurrentClient(i + 1, self.server_url)
            for i in range(self.num_clients)
        ]
        
        # 并发连接所有客户端
        connection_tasks = [client.connect() for client in self.clients]
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # 统计连接结果
        connected_count = 0
        for i, (client, result) in enumerate(zip(self.clients, results)):
            if isinstance(result, Exception):
                logger.error(f"客户端 {i + 1} 连接异常: {result}")
                self.test_results["errors"].append({
                    "client_id": i + 1,
                    "error": str(result),
                    "phase": "connection"
                })
            elif result:
                connected_count += 1
                self.test_results["connection_times"].append(client.connection_time)
        
        logger.info(f"成功连接 {connected_count}/{self.num_clients} 个客户端")
        return connected_count > 0
    
    async def cleanup_clients(self):
        """断开所有客户端连接"""
        logger.info("断开所有客户端连接...")
        disconnect_tasks = [
            client.disconnect() 
            for client in self.clients 
            if client.websocket
        ]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
    
    async def run_concurrent_transcriptions(self, audio_files: List[Path]):
        """并发运行转写任务"""
        logger.info(f"开始并发转写 {len(audio_files)} 个文件...")
        
        # 为每个客户端分配文件
        transcription_tasks = []
        for i, audio_file in enumerate(audio_files):
            client_idx = i % len(self.clients)
            client = self.clients[client_idx]
            
            if client.websocket:  # 只使用已连接的客户端
                task = client.transcribe_file(str(audio_file))
                transcription_tasks.append(task)
        
        # 并发执行所有转写任务
        start_time = time.time()
        results = await asyncio.gather(*transcription_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 处理结果
        successful_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"转写任务失败: {result}")
                self.test_results["errors"].append({
                    "error": str(result),
                    "phase": "transcription"
                })
            else:
                successful_count += 1
                self.test_results["transcription_results"].append(result)
        
        logger.info(f"并发转写完成: {successful_count}/{len(transcription_tasks)} 成功")
        logger.info(f"总耗时: {total_time:.2f}s, 平均: {total_time/len(transcription_tasks):.2f}s/任务")
        
        return {
            "total_tasks": len(transcription_tasks),
            "successful_tasks": successful_count,
            "failed_tasks": len(transcription_tasks) - successful_count,
            "total_time": total_time,
            "average_time": total_time / len(transcription_tasks) if transcription_tasks else 0
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            "connection_analysis": {},
            "transcription_analysis": {},
            "performance_metrics": {},
            "error_summary": {}
        }
        
        # 连接性能分析
        if self.test_results["connection_times"]:
            connection_times = self.test_results["connection_times"]
            analysis["connection_analysis"] = {
                "total_clients": self.num_clients,
                "connected_clients": len(connection_times),
                "average_connection_time": statistics.mean(connection_times),
                "min_connection_time": min(connection_times),
                "max_connection_time": max(connection_times),
                "std_deviation": statistics.stdev(connection_times) if len(connection_times) > 1 else 0
            }
        
        # 转写性能分析
        if self.test_results["transcription_results"]:
            results = self.test_results["transcription_results"]
            
            # 按客户端分组
            client_results = {}
            for result in results:
                client_id = result["client_id"]
                if client_id not in client_results:
                    client_results[client_id] = []
                client_results[client_id].append(result)
            
            # 计算各项指标
            all_processing_times = [r["processing_time"] for r in results]
            server_processing_times = [r["server_processing_time"] for r in results if not r["cached"]]
            cached_results = [r for r in results if r["cached"]]
            
            analysis["transcription_analysis"] = {
                "total_files": len(results),
                "cached_files": len(cached_results),
                "new_transcriptions": len(results) - len(cached_results),
                "cache_hit_rate": len(cached_results) / len(results) * 100 if results else 0,
                "average_processing_time": statistics.mean(all_processing_times),
                "min_processing_time": min(all_processing_times),
                "max_processing_time": max(all_processing_times),
                "std_deviation": statistics.stdev(all_processing_times) if len(all_processing_times) > 1 else 0
            }
            
            if server_processing_times:
                analysis["transcription_analysis"]["server_metrics"] = {
                    "average_server_time": statistics.mean(server_processing_times),
                    "min_server_time": min(server_processing_times),
                    "max_server_time": max(server_processing_times)
                }
            
            # 客户端性能分析
            client_metrics = {}
            for client_id, client_data in client_results.items():
                client_times = [r["processing_time"] for r in client_data]
                client_metrics[f"client_{client_id}"] = {
                    "tasks_completed": len(client_data),
                    "average_time": statistics.mean(client_times),
                    "min_time": min(client_times),
                    "max_time": max(client_times)
                }
            
            analysis["performance_metrics"]["client_performance"] = client_metrics
        
        # 错误分析
        if self.test_results["errors"]:
            error_types = {}
            for error in self.test_results["errors"]:
                phase = error.get("phase", "unknown")
                if phase not in error_types:
                    error_types[phase] = 0
                error_types[phase] += 1
            
            analysis["error_summary"] = {
                "total_errors": len(self.test_results["errors"]),
                "errors_by_phase": error_types
            }
        
        return analysis
    
    async def run_test(self):
        """运行完整的并发测试"""
        logger.info("=== 开始并发转写能力测试 ===")
        
        # 查找测试音频文件
        samples_dir = project_root / "samples/concurrency"
        audio_files = []
        
        for ext in ['.wav', '.mp3', '.mp4', '.m4a', '.flac']:
            audio_files.extend(samples_dir.glob(f"*{ext}"))
        
        if not audio_files:
            logger.error("未找到测试音频文件")
            return False
        
        logger.info(f"找到 {len(audio_files)} 个测试文件")
        
        # 如果文件数少于客户端数，复制使用
        if len(audio_files) < self.num_clients:
            audio_files = audio_files * (self.num_clients // len(audio_files) + 1)
            audio_files = audio_files[:self.num_clients * 2]  # 每个客户端至少2个文件
        
        try:
            # 1. 设置客户端
            if not await self.setup_clients():
                logger.error("客户端连接失败")
                return False
            
            # 2. 运行并发转写
            transcription_summary = await self.run_concurrent_transcriptions(audio_files)
            
            # 3. 分析结果
            analysis = self.analyze_results()
            
            # 4. 保存测试报告
            self.save_test_report(analysis, transcription_summary)
            
            return True
            
        finally:
            # 清理客户端连接
            await self.cleanup_clients()
    
    def save_test_report(self, analysis: Dict[str, Any], transcription_summary: Dict[str, Any]):
        """保存测试报告"""
        report = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_configuration": {
                "server_url": self.server_url,
                "num_clients": self.num_clients
            },
            "transcription_summary": transcription_summary,
            "analysis": analysis,
            "raw_results": self.test_results
        }
        
        # 保存报告
        output_dir = project_root / "tests" / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{timestamp}_concurrent_test_report.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试报告已保存: {output_file.name}")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("并发测试结果摘要")
        print("=" * 60)
        print(f"客户端数量: {self.num_clients}")
        print(f"成功连接: {len(self.test_results['connection_times'])}")
        print(f"总任务数: {transcription_summary['total_tasks']}")
        print(f"成功任务: {transcription_summary['successful_tasks']}")
        print(f"失败任务: {transcription_summary['failed_tasks']}")
        print(f"总耗时: {transcription_summary['total_time']:.2f}s")
        print(f"平均耗时: {transcription_summary['average_time']:.2f}s/任务")
        
        if analysis["transcription_analysis"]:
            trans_analysis = analysis["transcription_analysis"]
            print(f"\n缓存命中率: {trans_analysis['cache_hit_rate']:.1f}%")
            print(f"平均处理时间: {trans_analysis['average_processing_time']:.2f}s")
            print(f"最快处理时间: {trans_analysis['min_processing_time']:.2f}s")
            print(f"最慢处理时间: {trans_analysis['max_processing_time']:.2f}s")
        
        if self.test_results["errors"]:
            print(f"\n错误总数: {len(self.test_results['errors'])}")
        
        print("=" * 60 + "\n")


async def main():
    """主函数"""
    print("=" * 60)
    print("FunASR 并发转写能力测试")
    print("=" * 60)
    
    # 设置日志级别
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 从命令行参数获取客户端数量
    num_clients = 4  # 默认4个客户端
    if len(sys.argv) > 1:
        try:
            num_clients = int(sys.argv[1])
            print(f"使用 {num_clients} 个并发客户端")
        except ValueError:
            print(f"无效的客户端数量参数，使用默认值 {num_clients}")
    
    # 创建测试器
    tester = ConcurrentTranscriptionTester(num_clients=num_clients)
    
    try:
        success = await tester.run_test()
        if success:
            print("\n✓ 并发测试完成")
        else:
            print("\n✗ 并发测试失败")
            return 1
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        logger.exception("详细错误信息:")
        return 1
    
    return 0


if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code)