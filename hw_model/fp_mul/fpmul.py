import bf16.bf16 as bf16

class FloatMultiplication:
    def __init__(self, a:'bf16.Bfloat16', b:'bf16.Bfloat16') -> None:
        self.a = a
        self.b = b

    def multiply(self):
        # Decompose Bfloat16 to bitstring class
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        b_sign, b_exp, b_mant_nohidden = self.b.decompose_bf16()
        a_mant = bf16.bit(f'1{a_mant_nohidden}')
        b_mant = bf16.bit(f'1{b_mant_nohidden}')

        # sign
        res_sign = a_sign ^ b_sign

        # Special cases
        #input
        a_isnormal = False
        # nan * ? -> nan
        if self.a.isnan() or self.b.isnan():
            ret_exp_0 = bf16.bit(bin(bf16.Bfloat16.exp_max), bf16.Bfloat16.exponent_bits)
            ret_mant_0 = bf16.bit(bin(bf16.Bfloat16.mant_max), bf16.Bfloat16.mantissa_bits)
        # inf * 0 = nan
        elif (self.a.isinf() and self.b.iszero()) or (self.a.iszero() and self.b.isinf()):
            ret_exp_0 = bf16.bit(bin(bf16.Bfloat16.exp_max), bf16.Bfloat16.exponent_bits)
            ret_mant_0 = bf16.bit(bin(bf16.Bfloat16.mant_max), bf16.Bfloat16.mantissa_bits)
        # inf * !0 = inf
        elif (self.a.isinf() and (not self.b.iszero())) or ((not self.a.iszero()) and self.b.isinf()):
            ret_exp_0 = bf16.bit(bin(bf16.Bfloat16.exp_max), bf16.Bfloat16.exponent_bits)
            ret_mant_0 = bf16.bit('0', bf16.Bfloat16.mantissa_bits)
        # zero * x = zero
        elif ((not self.a.isinf()) and self.b.iszero()) or (self.a.iszero() and (not self.b.isinf())):
            ret_exp_0 = bf16.bit('0', bf16.Bfloat16.exponent_bits)
            ret_mant_0 = bf16.bit('0', bf16.Bfloat16.mantissa_bits)
        # normal case
        else:
            a_isnormal = True

        if a_isnormal:
            # exponent
            res_exp_0 = a_exp + b_exp - bf16.Bfloat16.bias
            # mantissa
            res_mant_0 = a_mant * b_mant

        # Handle rounding and normalization
        while res_mant_0 >= (1 << (self.a.mantissa_bits + 1)):
            res_mant_0 >>= 1
            result_exponent += 1
        # rounding function

        result_mantissa &= (1 << self.a.mantissa_bits) - 1

        print(bin(result_mantissa))
        result = self.a.compose_float(result_sign, result_exponent, result_mantissa)
        return bf16.Bfloat16(result, self.a.mantissa_bits, self.a.exponent_bits)
