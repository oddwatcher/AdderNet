
import torch
from typing import Optional,Dict,OrderedDict
import numpy as np

def clamp_integer(value, nbits):
    max_value = (1 << nbits) - 1  # 2^nbits - 1
    if value > max_value:
        return max_value
    else:
        return value

def add_mixed_precision(
    a: int,
    b: int,
    n_bit=32,
    n_approx_bit=8,
    mode="logic",
    table: Optional[Dict[tuple, tuple]] = None,
) -> int:

    return result



def tensor_add_p(
    a: torch.Tensor, b: torch.Tensor, n_bit=8, n_approx_bit=4, mode="logic", table=None
):
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
