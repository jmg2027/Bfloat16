from typing import List

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
