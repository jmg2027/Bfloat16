class FloatMultiplication:
    def __init__(self, a:'bf16.Bfloat16', b:'bf16.Bfloat16') -> None:
        self.a = a
        self.b = b

    def multiply(self):
        # Decompose Bfloat16 to bitstring class
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        b_sign, b_exp, b_mant_nohidden = self.b.decompose_bf16()
        a_mant = bf16.bit(self.a.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant = bf16.bit(self.b.mantissa_bits + 1, f'1{b_mant_nohidden}')

        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{b_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))
        
        # sign
        ret_sign = a_sign ^ b_sign

        # Special cases
        #input
        a_isnormal = False
        # nan * ? -> nan
        if self.a.isnan() or self.b.isnan():
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf * 0 = nan
        elif (self.a.isinf() and self.b.iszero()) or (self.a.iszero() and self.b.isinf()):
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf * !0 = inf
        elif (self.a.isinf() and (not self.b.iszero())) or ((not self.a.iszero()) and self.b.isinf()):
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # zero * x = zero
        elif ((not self.a.isinf()) and self.b.iszero()) or (self.a.iszero() and (not self.b.isinf())):
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits * '0', bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits * '0', bf16.Bfloat16.mantissa_bits * '0')
        # normal case
        else:
            a_isnormal = True

        if a_isnormal:
            # exponent
            ret_exp_0 = a_exp_signed + b_exp_signed - bias_signed
            # mantissa
            print('a_mant: ', a_mant)
            print('b_mant: ', b_mant)
            ret_mant_0 = (a_mant * b_mant)
            print('ret_mant_0: ', ret_mant_0, len(ret_mant_0))

            #normalize & rounding
            # now: truncate
            if ret_mant_0[((bf16.Bfloat16.mantissa_bits + 1) * 2 - 1)] == bf16.bit(1, '1'):
                ret_exp_1 = ret_exp_0 + 1
                ret_mant_1 = (ret_mant_0 >> 1)[(bf16.Bfloat16.mantissa_bits) * 2:0]
                ret_mant_2 = ret_mant_1[(bf16.Bfloat16.mantissa_bits) * 2: bf16.Bfloat16.mantissa_bits]
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = ret_mant_0
                ret_mant_2 = ret_mant_1[(bf16.Bfloat16.mantissa_bits + 1) * 2 - 1: bf16.Bfloat16.mantissa_bits + 1]

            #postnormalize
            ret_mant_3 = ret_mant_2[bf16.Bfloat16.mantissa_bits * 2 - 1:bf16.Bfloat16.mantissa_bits]


        # Remove sign bit from exponent
        ret_exp_bit_1 = bf16.bit(a_exp.bitwidth, ret_exp_1.bin)

        # Compose BF16
        mul = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp_bit_1, ret_mant_3)
        return mul
