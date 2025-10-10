"""
DirectML 快速验证脚本（仅测试 DML，不测试 CPU）
"""
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置输出编码为 UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# 应用补丁
from src.core.patches.directml_patch import apply_directml_patch
apply_directml_patch()
print("[OK] DirectML 补丁已应用")

# 测试
import torch
import torch_directml

print(f"DirectML 可用: {torch_directml.is_available()}")
if torch_directml.is_available():
    print(f"DirectML 设备数量: {torch_directml.device_count()}")
    print(f"DirectML 设备名称: {torch_directml.device_name(0)}")

from funasr import AutoModel

audio_file = str(project_root / "samples" / "test.m4a")
if not os.path.exists(audio_file):
    print(f"错误：测试文件不存在 {audio_file}")
    sys.exit(1)

print(f"\n开始测试 DirectML 加速...")
print(f"音频文件: {audio_file}")

start_time = time.time()

model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    punc_model="ct-punc-c",
    punc_model_revision="v2.0.4",
    spk_model="cam++",
    spk_model_revision="v2.0.2",
    device="dml",  # 使用 DirectML
    disable_update=True,
    disable_pbar=True
)

init_time = time.time() - start_time
print(f"模型初始化耗时: {init_time:.2f} 秒")

# 检查实际设备
actual_device = str(next(model.model.parameters()).device)
print(f"实际使用的设备: {actual_device}")

# 推理测试
print("开始推理...")
infer_start = time.time()
result = model.generate(input=audio_file, batch_size_s=300, hotword='')
infer_time = time.time() - infer_start

print(f"\n====== 测试结果 ======")
print(f"推理时间: {infer_time:.2f} 秒")

# 获取音频时长
import subprocess
try:
    proc_result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
        capture_output=True,
        text=True
    )
    duration = float(proc_result.stdout.strip())
    rtf = infer_time / duration
    print(f"音频时长: {duration:.2f} 秒")
    print(f"RTF: {rtf:.4f}")
    print(f"速度倍率: {1/rtf:.2f}x")
except:
    print("无法获取音频时长")

# 保存结果
if result and len(result) > 0:
    text = result[0].get('text', '')
    print(f"\n转录结果预览:\n{text[:200]}...")

print("\n[OK] DirectML 测试完成！")
