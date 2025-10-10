"""
DirectML Conv1D 兼容性测试脚本

测试 DirectML 对 Conv1D 操作的支持
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch_directml

print("=" * 60)
print("DirectML Conv1D 兼容性测试")
print("=" * 60)

print(f"\nPyTorch 版本: {torch.__version__}")
print(f"DirectML 可用: {torch_directml.is_available()}")
if torch_directml.is_available():
    print(f"DirectML 设备数: {torch_directml.device_count()}")
    print(f"DirectML 设备名: {torch_directml.device_name(0)}")

# 测试1: 简单的 Conv1D
print("\n" + "=" * 60)
print("测试1: 简单的 Conv1D (CPU)")
print("=" * 60)

try:
    # CPU 测试
    conv = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
    input_data = torch.randn(1, 16, 50)

    output = conv(input_data)
    print(f"✅ CPU Conv1D 成功")
    print(f"   输入形状: {input_data.shape}")
    print(f"   输出形状: {output.shape}")
except Exception as e:
    print(f"❌ CPU Conv1D 失败: {e}")

# 测试2: DirectML Conv1D (使用设备对象)
print("\n" + "=" * 60)
print("测试2: DirectML Conv1D (使用设备对象)")
print("=" * 60)

try:
    device_obj = torch_directml.device()
    print(f"设备对象: {device_obj}")
    print(f"设备类型: {device_obj.type}")

    conv_dml = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3).to(device_obj)
    input_dml = torch.randn(1, 16, 50).to(device_obj)

    output_dml = conv_dml(input_dml)
    print(f"✅ DirectML Conv1D (设备对象) 成功")
    print(f"   输入形状: {input_dml.shape}, 设备: {input_dml.device}")
    print(f"   输出形状: {output_dml.shape}, 设备: {output_dml.device}")
except Exception as e:
    print(f"❌ DirectML Conv1D (设备对象) 失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试3: DirectML Conv1D (使用字符串 "privateuseone")
print("\n" + "=" * 60)
print("测试3: DirectML Conv1D (使用字符串 privateuseone)")
print("=" * 60)

try:
    conv_str = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3).to("privateuseone")
    input_str = torch.randn(1, 16, 50).to("privateuseone")

    output_str = conv_str(input_str)
    print(f"✅ DirectML Conv1D (字符串) 成功")
    print(f"   输入形状: {input_str.shape}, 设备: {input_str.device}")
    print(f"   输出形状: {output_str.shape}, 设备: {output_str.device}")
except Exception as e:
    print(f"❌ DirectML Conv1D (字符串) 失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 带 padding 的 Conv1D
print("\n" + "=" * 60)
print("测试4: 带 padding 的 Conv1D (DirectML)")
print("=" * 60)

try:
    conv_pad = torch.nn.Conv1d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    ).to("privateuseone")
    input_pad = torch.randn(1, 16, 50).to("privateuseone")

    output_pad = conv_pad(input_pad)
    print(f"✅ DirectML Conv1D (padding) 成功")
    print(f"   输入形状: {input_pad.shape}")
    print(f"   输出形状: {output_pad.shape}")
except Exception as e:
    print(f"❌ DirectML Conv1D (padding) 失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
