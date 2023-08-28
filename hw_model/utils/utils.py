from typing import List
from bf16 import *


def radix_4_booth_encoder(bin: str) -> List:
    """
    String binary operand to encoded integer list
    """
    if len(bin) % 2 != 0:
        bin_operand = "0" + bin
    enc_list = list()

    # process in two bits
    for i in range(len(bin_operand) - 1, 0, -2):
        two_bits = bin_operand[i - 1: i + 1]
        if two_bits == "00":
            enc_value = 0
        elif two_bits == "01":
            enc_value == 1
        elif two_bits == "10":
            enc_value = -1
        elif two_bits == "11":
            enc_value = 2

        enc_list.insert(0, enc_value)
    return enc_list

def round_to_nearest_even_bit(bit: ubit, round_width: int) -> bit:
    # bitwidth of bit should be larger than round_width + grs bits
    # xxxx_xxxx_xxxx_xxxx...
    # xxxx_xxxG_RSSS_SSSS...
    if round_width + 2 > bit.bitwidth:
        raise ValueError("Bitwidth of before truncation should be larger than one of after truncation")
    guard_bit = bit[bit.bitwidth - round_width - 1]
    round_bit = bit[bit.bitwidth - round_width - 2]
    sticky_bitsting = bit[bit.bitwidth - round_width - 3:0]
    

    sticky_bit = sticky_bitsting.reduceor()
    truncated_bit = bit[bit.bitwidth-1:bit.bitwidth - round_width]

    # G = 0
    if guard_bit == ubit(1, '0'):
        rounded_bit = truncated_bit
    else:
        # GR = 11
        if round_bit == ubit(1, '1'):
            rounded_bit = truncated_bit + ubit(1, '1')
        else:
            # GRS = 101
            if sticky_bit != ubit(1, '0'):
                rounded_bit = truncated_bit + ubit(1, '1')
            # GRS = 100
            else:
                if truncated_bit[0] == ubit(1, '0'):
                    rounded_bit = truncated_bit
                else:
                    rounded_bit = truncated_bit + ubit(1, '1')
                    
    #print('trunc: ', truncated_bit)
    #print('round: ', rounded_bit)
    #print('guard', guard_bit)
    #print('round', round_bit)
    #print('sticky', sticky_bit)
    return rounded_bit

def leading_zero_count(bit: ubit) -> int:
    # If there's no 1 in bit, return zero indicator: 256
    if '1' not in bit.bin:
        return (1 << bf16_obj.exponent_bits)
    return (bit.bin+'1').index('1')


