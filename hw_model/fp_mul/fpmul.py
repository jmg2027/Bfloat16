import bf16.bf16 as bf16

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

        a_mant_us = bf16.ubit(a_mant.bitwidth, f'{a_mant.bin}')
        b_mant_us = bf16.ubit(b_mant.bitwidth, f'{b_mant.bin}')
        
        # sign
        ret_sign = a_sign ^ b_sign

        # Special cases
        #input
        isnormal = False
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
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bf16.Bfloat16.exponent_bits * '0')
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # normal case
        else:
            isnormal = True

        if isnormal:
            # exponent
            ret_exp_0 = a_exp_signed + b_exp_signed - bias_signed
            # mantissa
            ret_mant_0 = (a_mant_us * b_mant_us)

            #normalize & rounding
            # if mant = 16b'11.11_1111_1111_1111 then after rounding and postnormalization:: exp = exp + 2, mant = 7'b0
            # 16b'10.1111_..., 16b01.1111.... postnormalization check 
            if ret_mant_0 == bf16.ubit(ret_mant_0.bitwidth, '1111111111111111'):
                ret_exp_1 = ret_exp_0 + bf16.sbit(len(ret_exp_0) + 2 ,'010')
                # mant[15:0]
                ret_mant_1 = ret_mant_0
                # mant[15:8]
                ret_mant_2 = bf16.ubit(8, '10000000')
            # mant[15] == 1
            elif ret_mant_0[len(ret_mant_0) - 1] == bf16.bit(1, '1'):
                ret_exp_1 = ret_exp_0 + bf16.sbit(ret_exp_0.bitwidth + 2 ,'1')
                # mant[15:8]
                ret_mant_1 = ret_mant_0
                # rounding
                # mant[15:8]
                ret_mant_2 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_1, 8)
            # mant[15] == 0
            else:
                ret_exp_1 = ret_exp_0
                # mant[14:0]
                ret_mant_1 = ret_mant_0[len(ret_mant_0) - 2:0]
                # mant[14:7]
                ret_mant_2 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_1, 8)

            # remove hidden bit
            ret_mant_3 = ret_mant_2[bf16.Bfloat16.mantissa_bits - 1:0]

            # Overflow case: make inf
            if ret_exp_1 > bf16.sbit(ret_exp_1.bitwidth + 2 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_1 = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, bin((1 << self.a.exponent_bits) - 1))
                ret_mant_3 = 0
            # Underflow case: make zero
            elif ret_exp_1 < bf16.sbit(ret_exp_1.bitwidth + 2, bin(0)):
                ret_exp_1 = bf16.sbit(bf16.Bfloat16.exponent_bits, '0')
                ret_mant_3 = 0
        # Special case
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_3 = ret_mant_0


        # Remove sign bit from exponent
        ret_exp_bit_1 = bf16.bit(bf16.Bfloat16.exponent_bits, ret_exp_1.bin)

        # Compose BF16
        mul = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp_bit_1, ret_mant_3)
        return mul
