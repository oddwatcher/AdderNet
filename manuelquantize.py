import torch
import copy

def float_state_dict_to_int32_fixed_point(state_dict):
    """
    将 state_dict 中的所有浮点张量转换为 int32 定点表示。
    
    参数:
        state_dict (dict): PyTorch 模型的 state_dict，包含浮点参数
    
    返回:
        dict: 包含 int32 定点张量的新 state_dict
        dict: 元数据，包含每个张量的 scale 和 zero_point（用于反量化）
    """
    state_dict_int32 = {}
    metadata = {}

    qmin = -2147483648  # int32 最小值
    qmax = 2147483647   # int32 最大值

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            # 非浮点张量直接复制（如缓冲区中的布尔值、整数等）
            state_dict_int32[name] = tensor.clone() if isinstance(tensor, torch.Tensor) else copy.deepcopy(tensor)
            continue

        # 移动到 CPU 并展开为一维以获取全局范围
        flat_tensor = tensor.detach().cpu().flatten()
        min_val = flat_tensor.min().item()
        max_val = flat_tensor.max().item()

        # 处理极端情况：全为常数
        if min_val == max_val:
            if min_val == 0:
                scale = 1.0
            else:
                scale = abs(min_val) / ((qmax - qmin) / 2)
        else:
            scale = (max_val - min_val) / (qmax - qmin)

        # 计算 zero_point（非对称）
        zero_point = round(qmin - min_val / scale)
        zero_point = int(max(qmin, min(qmax, zero_point)))  # clamp

        # 量化：q = round(x / scale + zero_point)
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax).to(torch.int32)

        # 存储
        state_dict_int32[name] = quantized
        metadata[name] = {
            'scale': scale,
            'zero_point': zero_point,
            'original_min': min_val,
            'original_max': max_val
        }

    return state_dict_int32, metadata


# 可选：反量化函数（用于验证）
def dequantize_state_dict(state_dict_int32, metadata):
    """
    根据 metadata 反量化回浮点数
    """
    state_dict_fp32 = {}
    for name, tensor in state_dict_int32.items():
        if name not in metadata or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            state_dict_fp32[name] = tensor.clone()
            continue

        info = metadata[name]
        scale = info['scale']
        zero_point = info['zero_point']

        # 反量化：x = (q - zero_point) * scale
        fp32_tensor = (tensor.float() - zero_point) * scale
        state_dict_fp32[name] = fp32_tensor

    return state_dict_fp32