import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from typing import Optional, Dict
import numpy as np


def add_mixed_precision(
    a: int,
    b: int,
    n_bit=32,
    n_approx_bit=8,
    mode="logic",
    table: Optional[Dict[tuple, tuple]] = None,
) -> int:
    result = 0
    carry = 0
    for i in range(n_bit):
        a_bit = (a >> i) & 1
        b_bit = (b >> i) & 1
        if i < n_approx_bit:
            if mode == "table" and table:
                s, carry = approx_full_adder_table(a_bit, b_bit, carry, table)
            else:
                s, carry = approx_full_adder_logic(a_bit, b_bit, carry)
        else:
            s = a_bit ^ b_bit ^ carry
            carry = (a_bit & b_bit) | (b_bit & carry) | (a_bit & carry)
        result |= s << i
    return result


def approx_full_adder_logic(a_bit, b_bit, carry_in):
    s = a_bit ^ b_bit ^ carry_in
    carry_out = (a_bit & b_bit) | (b_bit & carry_in) | (a_bit & carry_in)
    return s, carry_out


def approx_full_adder_table(a_bit, b_bit, carry_in, table):
    key = (a_bit, b_bit, carry_in)
    return table.get(key, (0, 0))  # 默认值


def tensor_add_p(
    a: torch.Tensor, b: torch.Tensor, n_bit=8, n_approx_bit=4, mode="logic", table=None
):
    # 量化为整数
    a_int = (a * 2**n_bit).clamp(0, 2**n_bit - 1).to(torch.int32)
    b_int = (b * 2**n_bit).clamp(0, 2**n_bit - 1).to(torch.int32)

    a_np = a_int.cpu().numpy()
    b_np = b_int.cpu().numpy()

    v_add = np.vectorize(
        lambda x, y: add_mixed_precision(
            x, y, n_bit=n_bit, n_approx_bit=n_approx_bit, mode=mode, table=table
        )
    )

    result_int = v_add(a_np, b_np)
    result = torch.tensor(result_int, device=a.device, dtype=a.dtype) / (2**n_bit)
    return result


class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col, n_bit=8, n_approx_bit=4, mode="logic", table=None):
        ctx.save_for_backward(W_col, X_col)

        # 使用 p 函数进行加法
        output = tensor_add_p(
            W_col.unsqueeze(2), X_col.unsqueeze(0), n_bit, n_approx_bit, mode, table
        )
        output = -output.abs().sum(1)  # 保持与原逻辑一致的符号

        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = (
            (X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)
        ).sum(2)
        grad_W_col = (
            grad_W_col
            / grad_W_col.norm(p=2).clamp(min=1e-12)
            * math.sqrt(W_col.size(1) * W_col.size(0))
            / 5
        )
        grad_X_col = (
            -(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1)
            * grad_output.unsqueeze(1)
        ).sum(0)
        return grad_W_col, grad_X_col, None, None, None, None


def adder2d_function(
    X, W, stride=1, padding=0, n_bit=8, n_approx_bit=4, mode="logic", table=None
):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1

    X_col = F.unfold(
        X.view(1, -1, h_x, w_x), h_filter, padding=padding, stride=stride
    ).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)

    W_col = W.view(n_filters, -1)

    output = adder.apply(W_col, X_col, n_bit, n_approx_bit, mode, table)

    output = output.view(n_filters, h_out, w_out, n_x)
    output = output.permute(3, 0, 1, 2).contiguous()
    return output


class adder2d(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        n_bit=8,
        n_approx_bit=4,
        mode="logic",
    ):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = nn.Parameter(
            torch.randn(output_channel, input_channel, kernel_size, kernel_size)
        )
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(output_channel))

        # 添加配置参数
        self.n_bit = n_bit
        self.n_approx_bit = n_approx_bit
        self.mode = mode
        self.table = None  # 可以传入查表实现

    def forward(self, x):
        output = adder2d_function(
            x,
            self.adder,
            self.stride,
            self.padding,
            self.n_bit,
            self.n_approx_bit,
            self.mode,
            self.table,
        )
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output
