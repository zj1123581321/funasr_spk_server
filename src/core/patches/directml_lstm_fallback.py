"""
DirectML LSTM 回退补丁

由于 torch-directml 不支持 LSTM 操作,将 LSTM 模块回退到 CPU 执行
同时保持其他部分在 DirectML 上运行
"""
from loguru import logger
import torch

# 全局标志
_lstm_fallback_applied = False


def apply_lstm_cpu_fallback():
    """
    应用 LSTM CPU 回退补丁

    将 FunASR 中的 LSTM 模块强制运行在 CPU 上,
    避免 DirectML 不支持 LSTM 导致的错误
    """
    global _lstm_fallback_applied

    if _lstm_fallback_applied:
        logger.debug("LSTM CPU 回退补丁已应用,跳过重复应用")
        return

    # Monkey patch torch.nn.LSTM 的 forward 方法
    original_lstm_forward = torch.nn.LSTM.forward

    def lstm_forward_with_cpu_fallback(self, input, hx=None):
        """
        LSTM forward 方法的 CPU 回退版本

        如果输入在 DirectML 设备上,将其转移到 CPU,
        在 CPU 上执行 LSTM,然后将结果转回 DirectML
        """
        # 检查输入设备
        input_device = input.device
        is_directml = input_device.type == 'privateuseone'

        if is_directml:
            # 转移到 CPU
            input_cpu = input.cpu()

            # 转移 hidden state
            if hx is not None:
                hx_cpu = tuple(h.cpu() for h in hx)
            else:
                hx_cpu = None

            # 确保 LSTM 模块在 CPU 上
            if next(self.parameters()).device.type != 'cpu':
                logger.debug(f"LSTM 模块从 {next(self.parameters()).device} 转移到 CPU")
                self.cpu()

            # 在 CPU 上执行
            output, hidden = original_lstm_forward(self, input_cpu, hx_cpu)

            # 将结果转回 DirectML
            output = output.to(input_device)
            hidden = tuple(h.to(input_device) for h in hidden)

            logger.debug(f"LSTM CPU 回退: 输入 {input.shape} -> CPU -> DirectML")

            return output, hidden
        else:
            # 非 DirectML 设备,正常执行
            return original_lstm_forward(self, input, hx)

    # 应用补丁
    torch.nn.LSTM.forward = lstm_forward_with_cpu_fallback
    _lstm_fallback_applied = True

    logger.info("✅ LSTM CPU 回退补丁已应用 (DirectML 不支持 LSTM,将回退到 CPU)")
