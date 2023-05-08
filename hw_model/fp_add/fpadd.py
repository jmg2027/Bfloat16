import bf16.bf16 as bf16

class FloatAddition:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        # Decompose Bfloat16 to bitstring class
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        b_sign, b_exp, b_mant_nohidden = self.b.decompose_bf16()
        a_mant_us = bf16.ubit(self.a.mantissa_bits, f'1{a_mant_nohidden}')
        b_mant_us = bf16.ubit(self.b.mantissa_bits, f'1{b_mant_nohidden}')
        a_mant_signed = bf16.sbit(self.a.mantissa_bits + 2, f'01{a_mant_nohidden}')
        b_mant_signed = bf16.sbit(self.b.mantissa_bits + 2, f'01{b_mant_nohidden}')

        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{b_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        # if exponent difference is greater than mantissa bits + GRSSS (7 + 5), lesser number is considered as zero
        sticky_bit_width = bf16.ubit(2, bin(3))

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
            # Calculate Exponent differences
            exp_diff = a_exp_signed - b_exp_signed
            exp_diff_abs = abs(int(exp_diff))

            # Set flags
            a_exp_gt_b = exp_diff > bf16.sbit(1, '0')
            a_exp_eq_b = exp_diff == bf16.sbit(1, '0')
            a_mant_gt_b = a_mant_us > b_mant_us
            subtract_mant = a_sign ^ b_sign
            # if exponent difference is greater than mantissa bits + GRSSS (7 + 5), lesser number is considered as zero?

            if a_exp_gt_b:
                mant_unshift = a_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = b_mant_us.concat(bf16.ubit(2, '0'))
            else:
                mant_unshift = b_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = a_mant_us.concat(bf16.ubit(2, '0'))

            # Sticky bit
            if exp_diff_abs >= bf16.Bfloat16.mantissa_bits +  2 + int(sticky_bit_width):
                mant_sticky = mant_shift_in.reduceor()
            else:
                mant_sticky = mant_shift_in << bf16.Bfloat16.mantissa_bits +  2 + int(sticky_bit_width) - exp_diff_abs
            
            mant_shift = (mant_shift_in >> exp_diff_abs).concat(mant_sticky)

            if subtract_mant == bf16.bit(1, '1'):
                if a_exp_gt_b:
                    mant_inv_a = True
                elif a_exp_eq_b:
                    if a_mant_gt_b:
                        mant_inv_b = True
                    else:
                        mant_inv_a = True
                else:
                    mant_inv_a = True
            else:
                mant_inv_a = False
                mant_inv_b = False

            mant_add_carry = subtract_mant
            if mant_inv_a:
                mant_add_in_a = ~mant_shift
                mant_add_in_b = mant_unshift
            elif mant_inv_b:
                mant_add_in_a = mant_shift
                mant_add_in_b = ~mant_unshift
            else:
                mant_add_in_a = mant_shift
                mant_add_in_b = mant_unshift

            mant_add = mant_add_in_a + mant_add_in_b + mant_add_carry

            ret_mant_0 = (mant_add[mant_add.bitwidth-1] & subtract_mant).concat(mant_add[mant_add.bitwidth-2:0])

            # Sign Caculation
            if a_exp_gt_b:
                ret_sign_0 = a_sign
            elif a_exp_eq_b:
                if a_mant_gt_b:
                    ret_sign_0 = a_sign
                else: 
                    ret_sign_0 = b_sign
            else:
                ret_sign_0 = b_sign

            # Prenormalized Exponent
            if a_exp_gt_b:
                ret_exp_0 = a_exp_signed
            else:
                ret_exp_0 = b_exp_signed

            # Normalization
            if ret_mant_0[ret_mant_0.bitwidth-1] == bf16.ubit(1, '1'):
                ret_exp_1 = ret_exp_0 + bf16.ubit(ret_exp_0.bitwidth, '01')
                ret_mant_1 = ret_mant_0
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = ret_mant_0
            # Round
            # Need to check if ret_mant_2 if carry discards when rounds up 
            # TBD: What's wrong with sticky bits??
            ret_mant_2 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_1, 8, 0)

            # Post normalization
            if ret_mant_2[ret_mant_2.bitwidth-1] == bf16.ubit(1, '1'):
                ret_exp_2 = ret_exp_1 + bf16.ubit(ret_exp_1.bitwidth, '01')
                ret_mant_3 = ret_mant_2[ret_mant_2.bitwidth-1:ret_mant_2.bitwidth-9]
            else:
                ret_exp_2 = ret_exp_1
                ret_mant_3 = ret_mant_2
            
            ret_mant_4 = ret_mant_3[bf16.Bfloat16.mantissa_bits - 1:0]
            
        # Remove sign bit from exponent
        ret_exp_bit_2 = bf16.bit(a_exp.bitwidth, ret_exp_2.bin)

        # Compose BF16
        add = bf16.Bfloat16.compose_bf16(ret_sign_0, ret_exp_bit_2, ret_mant_4)
        return add
