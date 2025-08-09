"""
独立的工作进程 - 处理转录任务
每个进程完全独立，通过文件系统通信
"""
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from funasr import AutoModel


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_model(config: dict):
    """初始化FunASR模型"""
    funasr_config = config["funasr"]
    cache_dir = funasr_config["model_dir"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Worker] 初始化模型...")
    model = AutoModel(
        model=funasr_config["model"],
        model_revision=funasr_config["model_revision"],
        vad_model=funasr_config["vad_model"],
        vad_model_revision=funasr_config["vad_model_revision"],
        punc_model=funasr_config["punc_model"],
        punc_model_revision=funasr_config["punc_model_revision"],
        spk_model=funasr_config["spk_model"],
        spk_model_revision=funasr_config["spk_model_revision"],
        cache_dir=cache_dir,
        ncpu=funasr_config.get("ncpu", 8),
        device=funasr_config["device"],
        disable_update=funasr_config.get("disable_update", True),
        disable_pbar=funasr_config.get("disable_pbar", True)
    )
    print(f"[Worker] 模型初始化完成")
    return model


def process_task(model, task_file: str, config: dict):
    """处理单个任务"""
    # 读取任务
    with open(task_file, 'r', encoding='utf-8') as f:
        task = json.load(f)
    
    task_id = task['task_id']
    audio_path = task['audio_path']
    batch_size_s = task.get('batch_size_s', config["funasr"].get("batch_size_s", 300))
    hotword = task.get('hotword', '')
    
    print(f"[Worker] 处理任务 {task_id}: {os.path.basename(audio_path)}")
    
    # 结果文件路径
    result_file = task_file.replace('.task', '.result')
    
    try:
        # 执行模型推理
        result = model.generate(
            input=audio_path,
            batch_size_s=batch_size_s,
            hotword=hotword
        )
        
        # 保存结果 - 使用压缩格式减小文件大小
        result_data = {
            'task_id': task_id,
            'success': True,
            'result': result,
            'worker_pid': os.getpid()
        }
        
        # 使用紧凑格式，不添加额外的空格和换行
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, separators=(',', ':'))
        
        print(f"[Worker] 任务 {task_id} 完成")
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[Worker] 任务 {task_id} 失败: {error_msg}")
        
        # 保存错误结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'task_id': task_id,
                'success': False,
                'error': error_msg,
                'traceback': traceback_str,
                'worker_pid': os.getpid()
            }, f, ensure_ascii=False)
    
    finally:
        # 删除任务文件（表示任务已处理）
        try:
            os.remove(task_file)
        except:
            pass


def worker_loop(worker_id: int, config_path: str, task_dir: str):
    """工作进程主循环"""
    print(f"[Worker-{worker_id}] 启动 (PID: {os.getpid()})")
    
    # 加载配置
    config = load_config(config_path)
    
    # 初始化模型
    model = initialize_model(config)
    
    # 创建就绪标记文件
    ready_file = os.path.join(task_dir, f"worker_{worker_id}.ready")
    with open(ready_file, 'w') as f:
        f.write(str(os.getpid()))
    
    print(f"[Worker-{worker_id}] 就绪，等待任务...")
    
    # 监听任务
    while True:
        try:
            # 检查停止信号
            stop_file = os.path.join(task_dir, f"worker_{worker_id}.stop")
            if os.path.exists(stop_file):
                print(f"[Worker-{worker_id}] 收到停止信号")
                try:
                    os.remove(stop_file)
                    os.remove(ready_file)
                except:
                    pass
                break
            
            # 查找分配给此工作进程的任务
            task_pattern = f"worker_{worker_id}_*.task"
            task_files = list(Path(task_dir).glob(task_pattern))
            
            if task_files:
                # 处理第一个任务
                task_file = str(task_files[0])
                process_task(model, task_file, config)
            else:
                # 没有任务，短暂休眠
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"[Worker-{worker_id}] 收到中断信号")
            break
        except Exception as e:
            print(f"[Worker-{worker_id}] 循环异常: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    print(f"[Worker-{worker_id}] 退出")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="FunASR工作进程")
    parser.add_argument("--worker-id", type=int, required=True, help="工作进程ID")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--task-dir", type=str, default="./temp/tasks", help="任务目录")
    
    args = parser.parse_args()
    
    # 运行工作进程
    worker_loop(args.worker_id, args.config, args.task_dir)