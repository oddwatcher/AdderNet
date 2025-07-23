import torch
from torch.autograd import Function
import numpy as np
from numba import vectorize

INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min
UINT32_MAX = np.iinfo(np.uint32).max


@vectorize(
    ["int32(int32, int32, int32)"], target="parallel", fastmath=True, nopython=True
)
def approx_sum_B(a_int: np.int32, b_int: np.int32, approx_bits: np.int32) -> np.int32:

    a_sign = a_int > 0
    b_sign = b_int > 0
    # sum/sub detection
    mask_32 = np.uint32((1 << 32) - 1)
    mask_approx = np.uint32((1 << approx_bits) - 1)
    a = np.uint32(a_int)
    b = np.uint32(b_int)
    c0 = np.uint32(0)
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
    c_n = np.uint32(b_approx_low % 2)
    c_n_uint = np.uint32(c_n << (approx_bits + 1))
    s_high = np.uint32(s_high + c_n_uint)

    sum_s = np.int32(s_high + s_low)

    if a_sign and b_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not b_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s


@vectorize(
    ["int32(int32, int32, int32)"], target="parallel", fastmath=True, nopython=True
)
def approx_sum_B_32(
    a_int: np.int32, b_int: np.int32, approx_bits: np.int32
) -> np.int32:

    a_sign = a_int > 0
    b_sign = b_int > 0
    # sum/sub detection
    mask_approx = np.uint32((1 << 32) - 1)
    mask_32 = np.uint32((1 << 32) - 1)
    a = np.uint32(a_int)
    b = np.uint32(b_int)
    c0 = np.uint32(0)
    a_approx_low = a & mask_approx
    b_approx_low = b & mask_approx
    b_approx_low_1 = np.uint32(b_approx_low >> 1)

    b0 = np.uint32(b & ~(mask_32 - 1))
    a0 = np.uint32(a & ~(mask_32 - 1))
    s0 = np.uint32((a0 & c0) | (a0 & ~b0) | (~b0 & c0))

    s_low = np.uint32(
        (
            (a_approx_low & b_approx_low_1)
            | (a_approx_low & ~b_approx_low)
            | (~b_approx_low & b_approx_low_1)
        )
    )

    s_low = np.uint32(s_low >> 1)
    s_low = np.uint32((s_low << 1) + s0)

    sum_s = np.int32(s_low)

    if a_sign and b_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not b_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s


@vectorize(
    ["int32(int32, int32, int32)"], target="parallel", fastmath=True, nopython=True
)
def approx_sum_C(a_int: np.int32, b_int: np.int32, approx_bits: np.int32) -> np.int32:

    a = np.uint32(a_int)
    b = np.uint32(b_int)
    mask_approx = np.uint32((1 << approx_bits) - 1)

    a_approx_low = np.uint32(a & mask_approx)
    b_approx_low = np.uint32(b & mask_approx)
    a_approx_high = np.uint32(a - a_approx_low)
    b_approx_high = np.uint32(b - b_approx_low)

    s_high = np.uint32(a_approx_high + b_approx_high)
    s_low = np.uint32(a_approx_low | b_approx_low)
    s_high += np.uint32(
        ((b_approx_low >> (approx_bits - 1)) & (a_approx_low >> (approx_bits - 1)))
        << (approx_bits + 1)
    )

    sum_s = np.int32(s_high + s_low)

    a_sign = a_int > 0
    b_sign = b_int > 0

    if a_sign and b_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not b_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s


@vectorize(
    ["int32(int32, int32, int32)"], target="parallel", fastmath=True, nopython=True
)
def approx_sum_C_32(
    a_int: np.int32, b_int: np.int32, approx_bits: np.int32
) -> np.int32:

    a = np.uint32(a_int)
    b = np.uint32(b_int)
    mask_approx = np.uint32((1 << 32) - 1)

    a_approx_low = np.uint32(a & mask_approx)
    b_approx_low = np.uint32(b & mask_approx)

    s_low = np.uint32(a_approx_low | b_approx_low)

    sum_s = np.int32(s_low)

    a_sign = a_int > 0
    b_sign = b_int > 0

    if a_sign and b_sign:
        if sum_s <= 0:
            return np.int32(INT32_MAX)
    elif not a_sign and not b_sign:
        if sum_s >= 0:
            return np.int32(INT32_MIN)
    return sum_s


import random

if __name__ == "__main__":
    for i in range(1, 1 << 32 - 1):
        j = random.randint(1, 1 << 32 - 1)
        true = np.int64(i + j)
        result = approx_sum_B(i, j, 1)
        diff = np.int64(true - result)

        if diff > 10:
            if true > np.iinfo(np.int32).max or true < np.iinfo(np.int32).min:
                continue
            print(f"{i},{j},{true},{result}")
            result = approx_sum_B(i, j, 1)
