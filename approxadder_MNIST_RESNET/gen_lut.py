from typing import Dict, Tuple

def generate_approx_adder_table(approx_bit: int = 8) -> Dict[Tuple[int, int, int], Tuple[int, int]]:
    """
    生成近似加法器的查找表（LUT），用于查表实现。
    :param approx_bit: 控制哪些低位使用近似逻辑（前 approx_bit 位）
    :return: 查找表，格式为 {(a_bit, b_bit, carry_in): (sum_bit, carry_out)}
    """
    table = {}

    # 遍历所有可能的输入组合：a_bit ∈ {0,1}, b_bit ∈ {0,1}, carry_in ∈ {0,1}
    for a_bit in [0, 1]:
        for b_bit in [0, 1]:
            for carry_in in [0, 1]:
                # 对高位使用精确加法逻辑
                sum_exact = a_bit ^ b_bit ^ carry_in
                carry_exact = (a_bit & b_bit) | (b_bit & carry_in) | (a_bit & carry_in)

                # 对低位使用近似加法逻辑（这里可以自由设计近似规则）
                # 示例：在低位中，强制 carry_out 为 0，模拟近似行为
                if approx_bit > 0:
                    # 假设我们让前 approx_bit 位（低位）使用简单近似逻辑
                    # 这里你可以自定义更复杂的近似方式
                    sum_approx = a_bit ^ b_bit  # 忽略进位输入
                    carry_approx = (a_bit & b_bit)  # 忽略当前进位输入的影响
                else:
                    sum_approx = sum_exact
                    carry_approx = carry_exact

                # 使用一个简单的规则：前 approx_bit 位使用近似，高位使用精确
                # 在 add_mixed_precision 函数中会根据当前位 i < approx_bit 决定使用哪种逻辑
                # 所以在这里我们统一返回近似结果，让 add_mixed_precision 决定是否使用
                table[(a_bit, b_bit, carry_in)] = (sum_approx, carry_approx)

    return table