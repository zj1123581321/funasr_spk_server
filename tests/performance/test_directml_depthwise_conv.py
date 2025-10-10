"""
DirectML Depthwise Conv1D 测试

测试 DirectML 对深度可分离卷积 (groups=channels) 的支持
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch_directml

print("=" * 70)
print("DirectML Depthwise Conv1D 测试")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"DirectML 可用: {torch_directml.is_available()}")

# 测试1: 普通 Conv1D (CPU)
print("\n【测试1】普通 Conv1D (CPU)")
print("-" * 70)

try:
    conv_normal = torch.nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=0, groups=1, bias=False)
    input_data = torch.randn(1, 256, 100)
    output = conv_normal(input_data)
    print(f"✅ 成功")
    print(f"   输入: {input_data.shape}")
    print(f"   输出: {output.shape}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试2: Depthwise Conv1D (groups=channels) - CPU
print("\n【测试2】Depthwise Conv1D (groups=channels) - CPU")
print("-" * 70)

try:
    n_feat = 256
    kernel_size = 11
    conv_depthwise_cpu = torch.nn.Conv1d(
        n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=0, groups=n_feat, bias=False
    )
    input_cpu = torch.randn(1, n_feat, 100)
    output_cpu = conv_depthwise_cpu(input_cpu)
    print(f"✅ 成功")
    print(f"   channels: {n_feat}, kernel_size: {kernel_size}, groups: {n_feat}")
    print(f"   输入: {input_cpu.shape}")
    print(f"   输出: {output_cpu.shape}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 普通 Conv1D - DirectML
print("\n【测试3】普通 Conv1D - DirectML")
print("-" * 70)

try:
    conv_normal_dml = torch.nn.Conv1d(256, 256, kernel_size=11, stride=1, padding=0, groups=1, bias=False).to("privateuseone")
    input_dml = torch.randn(1, 256, 100).to("privateuseone")
    output_dml = conv_normal_dml(input_dml)
    print(f"✅ 成功")
    print(f"   输入: {input_dml.shape}, 设备: {input_dml.device}")
    print(f"   输出: {output_dml.shape}, 设备: {output_dml.device}")
except Exception as e:
    print(f"❌ 失败")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Depthwise Conv1D (groups=channels) - DirectML
print("\n【测试4】Depthwise Conv1D (groups=channels) - DirectML")
print("-" * 70)

try:
    n_feat = 256
    kernel_size = 11
    conv_depthwise_dml = torch.nn.Conv1d(
        n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=0, groups=n_feat, bias=False
    ).to("privateuseone")
    input_depthwise_dml = torch.randn(1, n_feat, 100).to("privateuseone")

    print(f"   配置: channels={n_feat}, kernel_size={kernel_size}, groups={n_feat}")
    print(f"   输入形状: {input_depthwise_dml.shape}")
    print(f"   开始推理...")

    output_depthwise_dml = conv_depthwise_dml(input_depthwise_dml)

    print(f"✅ 成功!")
    print(f"   输出: {output_depthwise_dml.shape}, 设备: {output_depthwise_dml.device}")
except Exception as e:
    print(f"❌ 失败 - 这就是问题所在!")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 不同 groups 值的测试
print("\n【测试5】不同 groups 值的测试 - DirectML")
print("-" * 70)

for groups in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    try:
        if 256 % groups != 0:
            print(f"groups={groups}: 跳过 (256 不能被 {groups} 整除)")
            continue

        conv_test = torch.nn.Conv1d(
            256, 256, kernel_size=11, stride=1, padding=0, groups=groups, bias=False
        ).to("privateuseone")
        input_test = torch.randn(1, 256, 100).to("privateuseone")
        output_test = conv_test(input_test)
        print(f"groups={groups:3d}: ✅ 成功")
    except Exception as e:
        print(f"groups={groups:3d}: ❌ 失败 - {type(e).__name__}: {str(e)[:50]}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
