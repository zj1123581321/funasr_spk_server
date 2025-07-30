"""
CPU优化性能测试脚本
测试不同优化策略对CPU使用率和转录速度的影响
"""
import os
import sys
import time
import psutil
import asyncio
import argparse
import platform
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from funasr import AutoModel
from src.utils.file_utils import get_audio_duration


class CPUOptimizationTester:
    """CPU优化性能测试器"""
    
    def __init__(self, audio_file: str, config_path: str = "config.json"):
        self.audio_file = audio_file
        self.config = self._load_config(config_path)
        self.results = []
        
        # 系统信息
        self.system_info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "cpu_count": mp.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        logger.info(f"系统信息: {json.dumps(self.system_info, indent=2)}")
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def monitor_cpu_usage(self, duration: float) -> Tuple[float, float, List[float]]:
        """监控CPU使用率"""
        cpu_percentages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_percentages.append(cpu_percent)
        
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        
        return avg_cpu, max_cpu, cpu_percentages
    
    async def test_single_thread(self) -> Dict:
        """测试单线程性能"""
        logger.info("\n=== 测试单线程性能 ===")
        
        # 初始化模型
        model = AutoModel(
            model=self.config["funasr"]["model"],
            model_revision=self.config["funasr"]["model_revision"],
            vad_model=self.config["funasr"]["vad_model"],
            vad_model_revision=self.config["funasr"]["vad_model_revision"],
            punc_model=self.config["funasr"]["punc_model"],
            punc_model_revision=self.config["funasr"]["punc_model_revision"],
            spk_model=self.config["funasr"]["spk_model"],
            spk_model_revision=self.config["funasr"]["spk_model_revision"],
            cache_dir=self.config["funasr"]["model_dir"],
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            ncpu=1  # 单线程
        )
        
        # 获取音频时长
        audio_duration = get_audio_duration(self.audio_file)
        logger.info(f"音频文件: {self.audio_file}, 时长: {audio_duration:.2f}秒")
        
        # 开始监控CPU
        import threading
        cpu_monitor_stop = threading.Event()
        cpu_results = {"avg": 0, "max": 0, "samples": []}
        
        def monitor_cpu():
            cpu_samples = []
            while not cpu_monitor_stop.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_samples.append(cpu_percent)
            
            if cpu_samples:
                cpu_results["avg"] = sum(cpu_samples) / len(cpu_samples)
                cpu_results["max"] = max(cpu_samples)
                cpu_results["samples"] = cpu_samples
        
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        # 执行转录
        start_time = time.time()
        result = model.generate(
            input=self.audio_file,
            batch_size_s=self.config["funasr"]["batch_size_s"],
            hotword=''
        )
        end_time = time.time()
        
        # 停止CPU监控
        cpu_monitor_stop.set()
        cpu_thread.join()
        
        processing_time = end_time - start_time
        speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
        
        test_result = {
            "method": "单线程",
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "speed_ratio": speed_ratio,
            "avg_cpu": cpu_results["avg"],
            "max_cpu": cpu_results["max"],
            "cpu_samples": cpu_results["samples"]
        }
        
        logger.info(f"单线程结果: 处理时间={processing_time:.2f}秒, 速度比={speed_ratio:.2f}x, 平均CPU={cpu_results['avg']:.1f}%, 最大CPU={cpu_results['max']:.1f}%")
        
        return test_result
    
    async def test_multi_thread(self, num_threads: int = None) -> Dict:
        """测试多线程性能"""
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        logger.info(f"\n=== 测试多线程性能 (线程数: {num_threads}) ===")
        
        # 初始化模型
        model = AutoModel(
            model=self.config["funasr"]["model"],
            model_revision=self.config["funasr"]["model_revision"],
            vad_model=self.config["funasr"]["vad_model"],
            vad_model_revision=self.config["funasr"]["vad_model_revision"],
            punc_model=self.config["funasr"]["punc_model"],
            punc_model_revision=self.config["funasr"]["punc_model_revision"],
            spk_model=self.config["funasr"]["spk_model"],
            spk_model_revision=self.config["funasr"]["spk_model_revision"],
            cache_dir=self.config["funasr"]["model_dir"],
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            ncpu=num_threads  # 多线程
        )
        
        # 获取音频时长
        audio_duration = get_audio_duration(self.audio_file)
        
        # 开始监控CPU
        import threading
        cpu_monitor_stop = threading.Event()
        cpu_results = {"avg": 0, "max": 0, "samples": []}
        
        def monitor_cpu():
            cpu_samples = []
            while not cpu_monitor_stop.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_samples.append(cpu_percent)
            
            if cpu_samples:
                cpu_results["avg"] = sum(cpu_samples) / len(cpu_samples)
                cpu_results["max"] = max(cpu_samples)
                cpu_results["samples"] = cpu_samples
        
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        # 执行转录
        start_time = time.time()
        result = model.generate(
            input=self.audio_file,
            batch_size_s=self.config["funasr"]["batch_size_s"],
            hotword=''
        )
        end_time = time.time()
        
        # 停止CPU监控
        cpu_monitor_stop.set()
        cpu_thread.join()
        
        processing_time = end_time - start_time
        speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
        
        test_result = {
            "method": f"多线程({num_threads}线程)",
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "speed_ratio": speed_ratio,
            "avg_cpu": cpu_results["avg"],
            "max_cpu": cpu_results["max"],
            "cpu_samples": cpu_results["samples"]
        }
        
        logger.info(f"多线程结果: 处理时间={processing_time:.2f}秒, 速度比={speed_ratio:.2f}x, 平均CPU={cpu_results['avg']:.1f}%, 最大CPU={cpu_results['max']:.1f}%")
        
        return test_result
    
    async def test_chunk_processing(self, chunk_size: int = 60) -> Dict:
        """测试分块处理性能"""
        logger.info(f"\n=== 测试分块处理性能 (块大小: {chunk_size}秒) ===")
        
        # 初始化模型
        model = AutoModel(
            model=self.config["funasr"]["model"],
            model_revision=self.config["funasr"]["model_revision"],
            vad_model=self.config["funasr"]["vad_model"],
            vad_model_revision=self.config["funasr"]["vad_model_revision"],
            punc_model=self.config["funasr"]["punc_model"],
            punc_model_revision=self.config["funasr"]["punc_model_revision"],
            spk_model=self.config["funasr"]["spk_model"],
            spk_model_revision=self.config["funasr"]["spk_model_revision"],
            cache_dir=self.config["funasr"]["model_dir"],
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            ncpu=mp.cpu_count()
        )
        
        # 获取音频时长
        audio_duration = get_audio_duration(self.audio_file)
        
        # 开始监控CPU
        import threading
        cpu_monitor_stop = threading.Event()
        cpu_results = {"avg": 0, "max": 0, "samples": []}
        
        def monitor_cpu():
            cpu_samples = []
            while not cpu_monitor_stop.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_samples.append(cpu_percent)
            
            if cpu_samples:
                cpu_results["avg"] = sum(cpu_samples) / len(cpu_samples)
                cpu_results["max"] = max(cpu_samples)
                cpu_results["samples"] = cpu_samples
        
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        
        # 执行转录（使用较小的batch_size_s来模拟分块）
        start_time = time.time()
        result = model.generate(
            input=self.audio_file,
            batch_size_s=chunk_size,  # 使用较小的批次大小
            hotword=''
        )
        end_time = time.time()
        
        # 停止CPU监控
        cpu_monitor_stop.set()
        cpu_thread.join()
        
        processing_time = end_time - start_time
        speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
        
        test_result = {
            "method": f"分块处理({chunk_size}秒/块)",
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "speed_ratio": speed_ratio,
            "avg_cpu": cpu_results["avg"],
            "max_cpu": cpu_results["max"],
            "cpu_samples": cpu_results["samples"]
        }
        
        logger.info(f"分块处理结果: 处理时间={processing_time:.2f}秒, 速度比={speed_ratio:.2f}x, 平均CPU={cpu_results['avg']:.1f}%, 最大CPU={cpu_results['max']:.1f}%")
        
        return test_result
    
    async def test_gpu_acceleration(self) -> Dict:
        """测试GPU加速性能（如果可用）"""
        import torch
        
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            logger.warning("GPU/MPS不可用，跳过GPU测试")
            return {
                "method": "GPU加速",
                "status": "不可用",
                "message": "系统未检测到GPU或Apple Silicon GPU"
            }
        
        device = "cuda" if torch.cuda.is_available() else "mps"
        logger.info(f"\n=== 测试GPU加速性能 (设备: {device}) ===")
        
        try:
            # 初始化模型
            model = AutoModel(
                model=self.config["funasr"]["model"],
                model_revision=self.config["funasr"]["model_revision"],
                vad_model=self.config["funasr"]["vad_model"],
                vad_model_revision=self.config["funasr"]["vad_model_revision"],
                punc_model=self.config["funasr"]["punc_model"],
                punc_model_revision=self.config["funasr"]["punc_model_revision"],
                spk_model=self.config["funasr"]["spk_model"],
                spk_model_revision=self.config["funasr"]["spk_model_revision"],
                cache_dir=self.config["funasr"]["model_dir"],
                device=device,  # 使用GPU
                disable_update=True,
                disable_pbar=True
            )
            
            # 获取音频时长
            audio_duration = get_audio_duration(self.audio_file)
            
            # 开始监控CPU
            import threading
            cpu_monitor_stop = threading.Event()
            cpu_results = {"avg": 0, "max": 0, "samples": []}
            
            def monitor_cpu():
                cpu_samples = []
                while not cpu_monitor_stop.is_set():
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    cpu_samples.append(cpu_percent)
                
                if cpu_samples:
                    cpu_results["avg"] = sum(cpu_samples) / len(cpu_samples)
                    cpu_results["max"] = max(cpu_samples)
                    cpu_results["samples"] = cpu_samples
            
            cpu_thread = threading.Thread(target=monitor_cpu)
            cpu_thread.start()
            
            # 执行转录
            start_time = time.time()
            result = model.generate(
                input=self.audio_file,
                batch_size_s=self.config["funasr"]["batch_size_s"],
                hotword=''
            )
            end_time = time.time()
            
            # 停止CPU监控
            cpu_monitor_stop.set()
            cpu_thread.join()
            
            processing_time = end_time - start_time
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            test_result = {
                "method": f"GPU加速({device})",
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "speed_ratio": speed_ratio,
                "avg_cpu": cpu_results["avg"],
                "max_cpu": cpu_results["max"],
                "cpu_samples": cpu_results["samples"]
            }
            
            logger.info(f"GPU加速结果: 处理时间={processing_time:.2f}秒, 速度比={speed_ratio:.2f}x, 平均CPU={cpu_results['avg']:.1f}%, 最大CPU={cpu_results['max']:.1f}%")
            
            return test_result
            
        except Exception as e:
            logger.error(f"GPU测试失败: {e}")
            return {
                "method": f"GPU加速({device})",
                "status": "失败",
                "error": str(e)
            }
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始CPU优化性能测试")
        logger.info(f"测试文件: {self.audio_file}")
        logger.info(f"{'='*60}")
        
        # 运行各种测试
        self.results.append(await self.test_single_thread())
        self.results.append(await self.test_multi_thread())
        self.results.append(await self.test_multi_thread(num_threads=mp.cpu_count() // 2))
        self.results.append(await self.test_chunk_processing(chunk_size=60))
        self.results.append(await self.test_chunk_processing(chunk_size=120))
        self.results.append(await self.test_gpu_acceleration())
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成性能测试报告"""
        logger.info(f"\n{'='*60}")
        logger.info(f"性能测试报告")
        logger.info(f"{'='*60}")
        
        # 创建报告
        report = {
            "test_time": datetime.now().isoformat(),
            "system_info": self.system_info,
            "audio_file": self.audio_file,
            "results": self.results
        }
        
        # 保存报告
        report_dir = project_root / "tests" / "performance" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"cpu_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"报告已保存到: {report_file}")
        
        # 打印摘要
        logger.info("\n性能测试摘要:")
        logger.info(f"{'方法':<30} {'处理时间(秒)':<15} {'速度比':<10} {'平均CPU%':<10} {'最大CPU%':<10}")
        logger.info("-" * 85)
        
        for result in self.results:
            if "status" in result:
                logger.info(f"{result['method']:<30} {result.get('status', 'N/A'):<15}")
            else:
                logger.info(
                    f"{result['method']:<30} "
                    f"{result['processing_time']:<15.2f} "
                    f"{result['speed_ratio']:<10.2f} "
                    f"{result['avg_cpu']:<10.1f} "
                    f"{result['max_cpu']:<10.1f}"
                )
        
        # 找出最佳方案
        valid_results = [r for r in self.results if "processing_time" in r]
        if valid_results:
            best_speed = max(valid_results, key=lambda x: x['speed_ratio'])
            best_cpu = max(valid_results, key=lambda x: x['avg_cpu'])
            
            logger.info(f"\n最快处理速度: {best_speed['method']} (速度比: {best_speed['speed_ratio']:.2f}x)")
            logger.info(f"最高CPU利用率: {best_cpu['method']} (平均CPU: {best_cpu['avg_cpu']:.1f}%)")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FunASR CPU优化性能测试")
    parser.add_argument(
        "--audio",
        type=str,
        default="samples/test_audio.wav",
        help="测试音频文件路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 检查音频文件
    if not os.path.exists(args.audio):
        logger.error(f"音频文件不存在: {args.audio}")
        return
    
    # 运行测试
    tester = CPUOptimizationTester(args.audio, args.config)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())