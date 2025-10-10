"""
FunASR SANM 模型修复补丁

修复 SANM attention 中的数据类型问题,确保 Conv1D 输入始终为浮点类型
这个问题在 DirectML 后端特别明显,因为 DirectML 不支持布尔类型的 Conv1D
"""
from loguru import logger

# 全局标志
_sanm_fix_applied = False


def apply_sanm_fix():
    """
    应用 SANM 模型修复补丁

    修复问题:
    1. forward_fsmn 中 mask 与 inputs 相乘后可能产生 bool 类型
    2. DirectML 的 Conv1D 不支持 bool 输入,导致 UnicodeDecodeError

    解决方案:
    - 在 Conv1D 之前确保输入为浮点类型
    """
    global _sanm_fix_applied

    if _sanm_fix_applied:
        logger.debug("SANM 修复补丁已应用,跳过重复应用")
        return

    try:
        from funasr.models.sanm import attention
    except ImportError:
        logger.error("无法导入 FunASR SANM attention 模块")
        raise

    import torch

    # 保存原始方法
    original_forward_fsmn = attention.MultiHeadedAttentionSANM.forward_fsmn

    def fixed_forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        """
        修复后的 forward_fsmn 方法

        确保传入 Conv1D 的数据始终为浮点类型
        """
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        # 修复关键点:确保数据类型为浮点
        # DirectML 的 Conv1D 不支持 bool 类型
        x = inputs.transpose(1, 2)

        # 类型检查和转换
        if x.dtype == torch.bool:
            logger.debug(f"检测到 bool 类型输入,转换为 float32 (shape={x.shape})")
            x = x.float()

        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    # 替换方法(而不是替换类)
    attention.MultiHeadedAttentionSANM.forward_fsmn = fixed_forward_fsmn
    _sanm_fix_applied = True

    logger.debug("SANM 修复补丁已成功应用")
