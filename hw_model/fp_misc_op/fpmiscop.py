import bf16.bf16 as bf16

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
            # needs improvement... to support bitstring class subtraction
#            ret_exp_0 = bf16.bit(bin(int(a_exp) + self.n))
            ret_exp_0 = a_exp + bf16.bit(bin(self.n))
        
        # output
        if a_isnormal:
            # Overflow case: make inf
            if ret_exp_0 > bf16.bit(bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_1 = (1 << self.a.exponent_bits) - 1
                ret_mant_1 = 0
            # Underflow case: make zero
            elif ret_exp_0 < bf16.bit(bin(0)):
                ret_exp_1 = 0
                ret_mant_1 = 0
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = a_mant
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_1 = a_mant
        ret_sign_1 = a_sign

        # Compose BF16
        pow = bf16.Bfloat16.compose_bf16(ret_sign_1, ret_exp_1, ret_mant_1)
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
        print(a_sign)
        if a_sign == bf16.bit('1'):
            ret_sign = '0'
        elif a_sign == bf16.bit('0'):
            ret_sign = '1'
        else:
            raise ValueError('Sign bit must be 0 or 1')

        # Compose BF16
        neg = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
        return neg