from ..utils.commonimport import *
from typing import Generic, TypeVar, Tuple
from ..utils import utils as hwutil
#from ..utils.utils import isnan, isinf, iszero

#FloatBaseT = TypeVar('FloatBaseT', bound='fc.FloatBase')


class FloatMultiplication:
    def __init__(self, a: FPBitT, b: FPBitT) -> None:
        self.a = a
        self.b = b

    def excute(self) -> FPBitT:
        # Decomposed bits
        a_sign, a_exp, a_mant_nohidden = self.a
        b_sign, b_exp, b_mant_nohidden = self.b
        a_mant = bit(fp32_config.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant = bit(fp32_config.mantissa_bits + 1, f'1{b_mant_nohidden}')

        a_exp_signed = sbit(fp32_config.exponent_bits + 2, f'0{a_exp.bin}')
        b_exp_signed = sbit(fp32_config.exponent_bits + 2, f'0{b_exp.bin}')
        bias_signed =  sbit(fp32_config.exponent_bits + 2, bin(fp32_config.bias))

        ret_sign_0 : bit = bit(1, '0')
        ret_exp_0 : bit = bit(1, '0')
        ret_mant_0: bit = bit(1, '0')

        a_mant_us = ubit(a_mant.bitwidth, f'{a_mant.bin}')
        b_mant_us = ubit(b_mant.bitwidth, f'{b_mant.bin}')
        
        # sign
        ret_sign_0 = a_sign ^ b_sign

        # Define precision bitwidth
        precision_bit = fp32_config.mantissa_bits + 1

        # Special cases
        #input
        isnormal = False
        # nan * ? -> nan
        if isnan(self.a) or isnan(self.b):
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max))
            ret_mant_0 = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # inf * 0 = nan
        elif (isinf(self.a) and iszero(self.b)) or (iszero(self.a) and isinf(self.b)):
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max))
            ret_mant_0 = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # inf * !0 = inf
        elif (isinf(self.a) and (not iszero(self.b))) or ((not iszero(self.a)) and isinf(self.b)):
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max))
            ret_mant_0 = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # zero * x = zero
        elif ((not isinf(self.a)) and iszero(self.b)) or (iszero(self.a) and (not isinf(self.b))):
            ret_exp_0 = bit(fp32_config.exponent_bits, fp32_config.exponent_bits * '0')
            ret_mant_0 = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # normal case
        else:
            isnormal = True

        if isnormal:
            # exponent
            ret_exp_0 = a_exp_signed + b_exp_signed - bias_signed
            # mantissa
            ret_mant_0 = (a_mant_us * b_mant_us)

            # define product bitwidth
            product_bit = 2 * precision_bit

            # normalize & rounding
            # In case of 11.11_1111_1111_1111, rounded value would be 100.0_0000_0000_0000 then after rounding and postnormalization:: exp = exp + 2, mant = 7'b0
            if ret_mant_0 == ubit(ret_mant_0.bitwidth, f'{"1" * product_bit}'):
                ret_exp_1 = ret_exp_0 + sbit(len(ret_exp_0) + 2 ,'010')
                # mant[15:0]
                ret_mant_1 = ret_mant_0
                # mant[15:8]
                ret_mant_2 = ubit(fp32_config.mantissa_bits + 1, f'1{"0" * (fp32_config.mantissa_bits)}')
            # In case of 10.11_1111_1111_1111, rounded value would be 11.00_0000_0000_0000 
            elif ret_mant_0 == ubit(ret_mant_0.bitwidth, f'10{"1" * (product_bit-2)}'):
                ret_exp_1 = ret_exp_0 + sbit(ret_exp_0.bitwidth + 2 ,'1')
                # mant[15:8]
                ret_mant_1 = ret_mant_0
                # rounding
                # mant[15:8]
                ret_mant_2 = ubit(fp32_config.mantissa_bits + 1, f'11{"0" * (fp32_config.mantissa_bits - 1)}')
            # In case of 01.11_1111_1111_1111, rounded value would be 10.00_0000_0000_0000
            elif ret_mant_0 == ubit(ret_mant_0.bitwidth, f'01{"1" * (product_bit-2)}'):
                ret_exp_1 = ret_exp_0 + sbit(ret_exp_0.bitwidth + 2 ,'1')
                # mant[15:8]
                ret_mant_1 = ret_mant_0
                # rounding
                # mant[15:8]
                ret_mant_2 = ubit(fp32_config.mantissa_bits + 1, f'1{"0" * (fp32_config.mantissa_bits)}')
            # mant[15] == 1
            elif ret_mant_0[len(ret_mant_0) - 1] == bit(1, '1'):
                ret_exp_1 = ret_exp_0 + sbit(ret_exp_0.bitwidth + 2 ,'1')
                # mant[15:8]
                ret_mant_1 = ret_mant_0
                # rounding
                # mant[15:8]
                ret_mant_2 = hwutil.round_to_nearest_even_bit(ret_mant_1, precision_bit)
            # mant[15] == 0
            else:
                ret_exp_1 = ret_exp_0
                # mant[14:0]
                ret_mant_1 = ret_mant_0[len(ret_mant_0) - 2:0]
                # mant[14:7]
                ret_mant_2 = hwutil.round_to_nearest_even_bit(ret_mant_1, precision_bit)
            

            # remove hidden bit
            ret_mant_3 = ret_mant_2[fp32_config.mantissa_bits - 1:0]

            # Overflow case: make inf
            if ret_exp_1 > sbit(ret_exp_1.bitwidth + 2 , bin((1 << fp32_config.exponent_bits) - 1)):
                ret_exp_1 = sbit(fp32_config.exponent_bits + 2, bin((1 << fp32_config.exponent_bits) - 1))
                ret_mant_3 = bit(fp32_config.mantissa_bits, '0')
            # Underflow case: make zero
            elif ret_exp_1 < sbit(ret_exp_1.bitwidth + 2, bin(0)):
                ret_exp_1 = sbit(fp32_config.exponent_bits, '0')
                ret_mant_3 = bit(fp32_config.mantissa_bits, '0')
        # Special case
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_3 = ret_mant_0


        # Remove sign bit from exponent
        ret_exp_bit_1 = bit(fp32_config.exponent_bits, ret_exp_1.bin)

        ret_sign = ret_sign_0
        ret_exp = ret_exp_bit_1
        ret_mant = ret_mant_3

        return ret_sign, ret_exp, ret_mant

        # Compose BF16
        mul = fp32_int.compose(ret_sign, ret_exp_bit_1, ret_mant_3)
        # If input is Bfloat16, fp32_to_bf16
        if bf16_input:
            mul = mul.fp32_to_bf16()
        return mul
