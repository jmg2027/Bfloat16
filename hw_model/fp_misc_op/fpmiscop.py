import bf16.bf16 as bf16
from bf16.bf16 import sbit
from bf16.bf16 import bit

class FloatPowerofTwo:
    
    """
    FloatPowerofTwo
    For given a, return n power of 2
    a: Bfloat16, n: signed integer
    """
    def __init__(self, a:'bf16.Bfloat16', n: int) -> None:
        self.a = a
        self.n = n

    def power(self) -> 'bf16.Bfloat16':
        # Decompose Bfloat16 to BitString
        a_sign, a_exp, a_mant = self.a.decompose_bf16()

        # Exponent to signed bitstring for n to be negative
        # +1 for sign bit
        signed_a_exp = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        signed_bias = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))
        signed_n = bf16.sbit(a_exp.bitwidth + 2, bin(self.n))

        #Check for special cases
        # input
        # zero
        a_isnormal = False
        if self.a.iszero():
            ret_exp_0 = 0
        # inf
        elif self.a.isinf():
            ret_exp_0 = a_exp
        # nan
        elif self.a.isnan():
            ret_exp_0 = a_exp
        # normal case
        else:
            a_isnormal = True
            ret_exp_0 = signed_a_exp + signed_n

        
        # output
        if a_isnormal:
            # Overflow case: make inf
            if ret_exp_0 > bf16.sbit(ret_exp_0.bitwidth + 2 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_1 = bf16.sbit(a_exp.bitwidth + 2, bin((1 << self.a.exponent_bits) - 1))
                ret_mant_1 = 0
            # Underflow case: make zero
            elif ret_exp_0 < bf16.sbit(ret_exp_0.bitwidth + 2, bin(0)):
                ret_exp_1 = bf16.sbit(a_exp.bitwidth + 2, '0')
                ret_mant_1 = 0
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = a_mant
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_1 = a_mant
        ret_sign_1 = a_sign

        # Remove sign bit from exponent
        ret_exp_bit_1 = bf16.bit(a_exp.bitwidth, ret_exp_1.bin)

        # Compose BF16
        pow = bf16.Bfloat16.compose_bf16(ret_sign_1, ret_exp_bit_1, ret_mant_1)
        return pow


class FloatNegative:
    
    """
    FloatNegative
    For given a, return negative a
    a: Bfloat16
    """
    def __init__(self, a:'bf16.Bfloat16') -> None:
        self.a = a

    def negative(self) -> 'bf16.Bfloat16':
        # Decompose Bfloat16 to BitString class
        a_sign, a_exp, a_mant = self.a.decompose_bf16()

        # neg -> pos
        ret_exp = a_exp
        ret_mant = a_mant
        if a_sign == bf16.bit(1, '1'):
            ret_sign = '0'
        elif a_sign == bf16.bit(1, '0'):
            ret_sign = '1'
        else:
            raise ValueError('Sign bit must be 0 or 1')

        # Compose BF16
        neg = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
        return neg
