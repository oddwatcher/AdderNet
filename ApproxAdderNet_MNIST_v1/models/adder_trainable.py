# models/adder_trainable.py
import torch
import torch.nn as nn
import math
from torch.autograd import Function
from models.adder_approx import add_tensor_approx

def adder2d_function(X, W, stride=1, padding=0, approx_bit=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1

    X_col = torch.nn.functional.unfold(
        X.view(1, -1, h_x, w_x),
        kernel_size=h_filter,
        dilation=1,
        padding=padding,
        stride=stride
    ).view(n_x, -1, h_out * w_out)

    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = adder.apply(W_col, X_col, approx_bit)
    out = out.view(n_filters, h_out, w_out, n_x).permute(3, 0, 1, 2).contiguous()
    return out

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col, approx_bit):
        ctx.save_for_backward(W_col, X_col)
        ctx.approx_bit = approx_bit
        # output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        if approx_bit == 0:
            output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        else:
            W_exp = W_col.unsqueeze(2).to(torch.int32)
            X_exp = X_col.unsqueeze(0).to(torch.int32)
            add_result = add_tensor_approx(W_exp, X_exp, n_bit=32, n_approx_bit=approx_bit).to(torch.float32)
            output = -(add_result.to(torch.float32)).abs().sum(1)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        approx_bit = ctx.approx_bit  # 恢复 approx_bit
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col / grad_W_col.norm(p=2).clamp(min=1e-12) * math.sqrt(W_col.size(1) * W_col.size(0)) / 5
        grad_X_col = (-(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)).sum(0)
        return grad_W_col, grad_X_col, None

class Adder2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, approx_bit=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.adder = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size).normal_()
        )
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        self.approx_bit = approx_bit

    def forward(self, x):
        out = adder2d_function(x, self.adder, self.stride, self.padding, approx_bit=self.approx_bit)
        if self.bias:
            out += self.b.view(1, -1, 1, 1)
        return out
