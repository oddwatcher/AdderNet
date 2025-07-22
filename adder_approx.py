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

    return s_high + s_low  # numpy will automatically expand the type of data


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
    ) << (approx_bits + 1)

    return s_high + s_low


INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min
UINT32_MAX = np.iinfo(np.uint32).max


def add_approx_signed(a_int: np.int32, b_int: np.int32, adder, approx_bit) -> np.int32:
    a_u = np.uint32(a_int)
    b_u = np.uint32(b_int)
    sum_u = adder(a_u, b_u, approx_bit)
    sum_s = np.int32(sum_u)

    a_sign = a_int > 0
    b_sign = b_int > 0

    if a_sign and b_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not b_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s


def sub_approx_signed(a_int: np.int32, b_int: np.int32, adder, approx_bit) -> np.int32:
    a_u = np.uint32(a_int)
    neg_b_u = np.uint32(-np.int32(b_int))
    sum_u = adder(a_u, neg_b_u, approx_bit)
    sum_s = np.int32(sum_u)

    c_s = np.int32(neg_b_u)
    a_sign = a_int > 0
    c_sign = c_s > 0

    if a_sign and c_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not c_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s
