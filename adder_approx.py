import torch
from torch.autograd import Function
import numpy as np


def approx_add_B(a: np.uint32, b: np.uint32, approx_bits: int) -> int:
    a = np.uint32(a)
    b = np.uint32(b)
    mask_32 = np.uint32((1 << 32) - 1)
    mask_approx = np.uint32((1 << approx_bits) - 1)

    c0 = np.uint(0)
    a_approx_low = a & mask_approx
    b_approx_low = b & mask_approx
    b_approx_low_1 = np.uint32(b_approx_low >> 1)
    a_approx_high = a - a_approx_low
    b_approx_high = b - b_approx_low

    b0 = np.uint32(b & ~(mask_32 - 1))
    a0 = np.uint32(a & ~(mask_32 - 1))
    s0 = np.uint32((a0 & c0) | (a0 & ~b0) | (~b0 & c0))

    s_high = np.uint32(a_approx_high + b_approx_high)
    s_low = np.uint32(
        (
            (a_approx_low & b_approx_low_1)
            | (a_approx_low & ~b_approx_low)
            | (~b_approx_low & b_approx_low_1)
        )
    )

    s_low = np.uint32(s_low >> 1)
    s_low = np.uint32((s_low << 1) + s0)
    c_n = np.uint32(b_approx_low >> (approx_bits - 1))

    s_high = np.uint32(s_high + c_n << (approx_bits + 1))

    return np.uint32(s_high + s_low)


result = approx_add_B(0b011010010110100101101001, 1 << 15, 16)
print(result)


def approx_add_C(a: int, b: int, approx_bits: int) -> int:
    a = np.uint32(a)
    b = np.uint32(b)
    mask_approx = np.uint32((1 << approx_bits) - 1)

    a_approx_low = np.uint32(a & mask_approx)
    b_approx_low = np.uint32(b & mask_approx)
    a_approx_high = np.uint32(a - a_approx_low)
    b_approx_high = np.uint32(b - b_approx_low)

    s_high = np.uint32(a_approx_high + b_approx_high)
    s_low = np.uint32(a_approx_low | b_approx_low)
    s_high += np.uint32(
        (b_approx_low >> approx_bits - 1) & (a_approx_low >> approx_bits - 1)
    )

    return np.uint32(s_high + s_low)


def approx_sub(x: int, y: int, approx_bits: int, approx_add) -> int:

    if y == 0x80000000:
        y_flipped = y
    else:
        y_flipped = -y & 0xFFFFFFFF
    return approx_add(x, y_flipped, approx_bits)

