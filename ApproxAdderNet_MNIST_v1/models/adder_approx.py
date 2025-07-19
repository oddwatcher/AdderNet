# models/adder_approx.py
import torch
import torch.nn as nn
from typing import Optional
from typing import Dict
import numpy as np

def approx_full_adder_logic(a_bit: int, b_bit: int, cin: int):
    """
    近似全加器的逻辑实现。
    :param a_bit: 第一位
    :param b_bit: 第二位
    :param cin: 进位输入
    :return: 和 bit 和进位输出
    """
    s = a_bit ^ b_bit ^ cin
    carry = b_bit
    return s, carry

def approx_full_adder_table(a_bit: int, b_bit: int, cin: int, table: Dict[tuple, tuple]):
    """
    近似全加器的查表实现。
    :param a_bit: 第一位
    :param b_bit: 第二位
    :param cin: 进位输入
    :param table: 查表表
    :return: 和 bit 和进位输出
    """
    return table.get((a_bit, b_bit, cin), (0, 0))

def add_mixed_precision(a: int, b: int, n_bit=32, n_approx_bit=8, mode='logic', table: Optional[Dict[tuple, tuple]] = None) -> int:
    """
    混合精度加法，支持逻辑实现和查表实现。
    :param a: 进行加法操作的第一个整数值
    :param b: 进行加法操作的第二个整数值
    :param n_bit: 量化位数
    :param n_approx_bit: 近似计算的位数
    :param mode: 近似加法模式，支持 'logic'（逻辑实现）和 'table'（查表实现）
    :param table: 如果 mode='table'，则需要传入查表表
    :return: 加法结果
    """
    result = 0
    carry = 0
    for i in range(n_bit):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        if i < n_approx_bit:
            if mode == 'table' and table:
                s, carry = approx_full_adder_table(a_bit, b_bit, carry, table)
            else:
                s, carry = approx_full_adder_logic(a_bit, b_bit, carry)
        else:
            # 对精确部分直接相加
            s = a_bit ^ b_bit ^ carry
            carry = (a_bit & b_bit) | (b_bit & carry) | (a_bit & carry)
        result |= (s << i)
    return result

def add_tensor_approx(a_tensor: torch.Tensor, b_tensor: torch.Tensor,
                      n_bit=32, n_approx_bit=8,
                      mode='logic', table: Optional[Dict[tuple, tuple]] = None) -> torch.Tensor:
    """
    使用近似加法对张量进行逐元素计算。

    :param a_tensor: 张量 a
    :param b_tensor: 张量 b
    :param n_bit: 量化位数
    :param n_approx_bit: 近似计算的位数
    :param mode: 近似加法模式，支持 'logic'（逻辑实现）和 'table'（查表实现）
    :param table: 如果 mode='table'，则需要传入查表表
    :return: 计算结果张量
    """
    # 广播形状
    a_tensor, b_tensor = torch.broadcast_tensors(a_tensor, b_tensor)

    # 1. 统计全局/固定 min/max（可根据实际情况调整为训练集统计值）
    # 这里以推理时当前 batch 的 min/max 为例
    min_a, max_a = a_tensor.min().item(), a_tensor.max().item()
    min_b, max_b = b_tensor.min().item(), b_tensor.max().item()

    # 2. 对称量化到有符号 int32
    def quantize_symmetric(x, max_abs, n_bit=32):
        scale = (2**(n_bit-1) - 1) / (max_abs + 1e-8)
        x_int = (x * scale).round().clamp(-(2**(n_bit-1)), 2**(n_bit-1)-1).to(torch.int32)
        return x_int, scale

    max_abs_a = max(abs(min_a), abs(max_a))
    max_abs_b = max(abs(min_b), abs(max_b))
    a_int, a_scale = quantize_symmetric(a_tensor, max_abs_a, n_bit)
    b_int, b_scale = quantize_symmetric(b_tensor, max_abs_b, n_bit)

    # 3. 直接取负号，得到 -b 的有符号补码
    b_neg_int = (-b_int).to(torch.int32)

    # 4. 用 numpy+numba 批量加速近似加法（有符号int32输入）
   

    # numba 只支持 logic 模式的加速
    if mode == 'logic' and table is None:
        try:
            from numba import njit
        except ImportError:
            def njit(x):
                return x

        @njit
        def add_mixed_precision_numba(a, b, n_bit, n_approx_bit):
            result = 0
            carry = 0
            for i in range(n_bit):
                a_bit = (a >> i) & 1
                b_bit = (b >> i) & 1
                if i < n_approx_bit:
                    # 近似加法逻辑
                    s = a_bit ^ b_bit ^ carry
                    carry = b_bit
                else:
                    s = a_bit ^ b_bit ^ carry
                    carry = (a_bit & b_bit) | (b_bit & carry) | (a_bit & carry)
                result |= (s << i)
            return result

        @njit
        def add_mixed_precision_vec(a_arr, b_arr, n_bit, n_approx_bit):
            out = np.empty_like(a_arr)
            for idx in range(a_arr.size):
                a = int(a_arr.flat[idx])
                b = int(b_arr.flat[idx])
                result = add_mixed_precision_numba(a, b, n_bit, n_approx_bit)
                out.flat[idx] = np.int32(result)
            return out

        a_np = a_int.cpu().numpy().astype(np.int32)
        b_np = b_neg_int.cpu().numpy().astype(np.int32)
        result_np = add_mixed_precision_vec(a_np, b_np, n_bit, n_approx_bit)
        result_int = torch.from_numpy(result_np).to(a_int.device)
    else:
        # fallback: 用python for循环（支持table模式）
        a_flat = a_int.view(-1)
        b_flat = b_neg_int.view(-1)
        result_list = []
        for i in range(a_flat.shape[0]):
            result = add_mixed_precision(int(a_flat[i].item()), int(b_flat[i].item()), n_bit, n_approx_bit, mode, table)
            result_list.append(result)
        result_int = torch.tensor(result_list, dtype=torch.int32, device=a_int.device).view(a_int.shape)

    # 5. 反量化回 float32
    result_float = result_int.float() / a_scale
    return result_float

# 示例调用
if __name__ == "__main__":
    # 创建示例张量
    a_tensor = torch.tensor([-4.0, 0.5, 3.0], dtype=torch.float32).view(1, 1, 3)
    b_tensor = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32).view(1, 1, 3)

    # 调用 add_tensor_approx 函数
    result = add_tensor_approx(a_tensor, b_tensor, n_bit=32, n_approx_bit=8, mode='logic')

    print(f"Result: {result}")