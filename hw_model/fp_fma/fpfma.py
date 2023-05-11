import bf16.bf16 as bf16

class FloatFMA:
    # https://repositories.lib.utexas.edu/bitstream/handle/2152/13269/quinnelle60861.pdf?sequence=2
    def __init__(self, a: 'bf16.Bfloat16', b: 'bf16.Bfloat16', c: 'bf16.Bfloat16'):
        self.a = a
        self.b = b
        self.c = c

    def fma(self) -> 'bf16.Bfloat16':
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        b_sign, b_exp, b_mant_nohidden = self.b.decompose_bf16()
        c_sign, c_exp, c_mant_nohidden = self.c.decompose_bf16()
        a_mant_us = bf16.ubit(self.a.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant_us = bf16.ubit(self.b.mantissa_bits + 1, f'1{b_mant_nohidden}')
        c_mant_us = bf16.ubit(self.c.mantissa_bits + 1, f'1{c_mant_nohidden}')

        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = bf16.sbit(b_exp.bitwidth + 2, f'0{b_exp.bin}')
        c_exp_signed = bf16.sbit(c_exp.bitwidth + 2, f'0{c_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        # Special cases
        #input
        # fma(nan,?,?) -> nan
        # nan inputs clearance
        isnormal = False
        if self.a.isnan() or self.b.isnan() or self.c.isnan():
            ret_sign_0 = a_sign ^ b_sign ^ c_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        else:
            isnormal = True

        if isnormal:
            # Simple way
            exp_diff = a_exp_signed - b_exp_signed
            exp_diff_abs = abs(int(exp_diff))

            product = self.a * self.b
            print(product)
            result = product + self.c
            print(self.c)
            print(result)
            ret_sign = 0
            ret_exp_signed = 0
            ret_mant = 0
        else:
            ret_sign = ret_sign_0
            ret_exp_signed = ret_exp_0
            ret_mant = ret_mant_0

        """
        Define separate paths for each case: Adder Anchor Path, Product Anchor Path, and Close Path.
        Implement exponent and sign logic for the FMA architecture.

        For the Adder Anchor Path:
        Select this path when the exponent difference indicates that the addend is larger.
        Align the product terms over a 57-bit range and invert them for subtractions.
        Combine all operands using 3:2 Carry Save Adders (CSAs) or Half Adders (HAs).

        For the Product Anchor Path:
        Select this path when the exponent difference indicates that the product is larger.
        Align and invert the addend against the position of the product.
        Combine the operands using a 3:2 Carry Save Adder (CSA).

        For the Close Path:
        Select this path when the exponent difference between the addend and product is too close.
        Handle fused multiply-add subtraction operations specifically for massive cancellation.
        Use significand swapping to remove the need for a complementation stage.
        For Exponent and Sign Logic:

        Use four initial exponent calculations: three for alignment, comparison, and multiplier exponent values, and a fourth for the close path alignment calculation.
        Remove the classic fused multiply-add normalization adjustment adders.
        """

        # Remove sign bit from exponent
        #ret_exp = bf16.bit(bf16.Bfloat16.exponent_bits, ret_exp_signed.bin)

        # Compose BF16
        #fma = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
        fma = result
        return fma