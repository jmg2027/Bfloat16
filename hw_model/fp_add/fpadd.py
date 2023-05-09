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

        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{b_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        # if exponent difference is greater than mantissa bits + GRSSS (7 + 5), lesser number is considered as zero
#        sticky_bit_width = bf16.ubit(2, bin(3))

        # Special cases
        #input
        isnormal = False
        # nan + ? -> nan
        if self.a.isnan() or self.b.isnan():
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf + 0 = nan ?
        # tbd
        elif (self.a.isinf() and self.b.iszero()) or (self.a.iszero() and self.b.isinf()):
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf + !0 = inf
        elif (self.a.isinf() and (not self.b.iszero())) or ((not self.a.iszero()) and self.b.isinf()):
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # zero + x = x
#        elif ((not self.a.isinf()) and self.b.iszero()) or (self.a.iszero() and (not self.b.isinf())):
#            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bf16.Bfloat16.exponent_bits * '0')
#            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # normal case
        else:
            isnormal = True

        
        if isnormal:
            # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6489167
            # Exponent calculation
            # Calculate Exponent differences
            exp_diff = a_exp_signed - b_exp_signed
            exp_diff_abs = abs(int(exp_diff))
            # Set flags
            a_exp_gt_b = exp_diff > bf16.sbit(1, '0')
            a_exp_eq_b = exp_diff == bf16.sbit(1, '0')
            a_mant_gt_b = a_mant_us > b_mant_us
            subtract_mant = a_sign ^ b_sign
            # Prenormalized Exponent
            if a_exp_gt_b:
                ret_exp_0 = a_exp_signed
            else:
                ret_exp_0 = b_exp_signed

            # Mantissa swap and shift
            # a_mant_us = [7:0]
            # mant_unshift = [7+3:0] -> grs
            # mant_shift = [7+3:0] -> grs
            # mant_shift_in = [7+2:0] -> grs, sticky bit will concat later
            if a_exp_gt_b:
                mant_unshift = a_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = b_mant_us.concat(bf16.ubit(2, '0'))
            else:
                mant_unshift = b_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = a_mant_us.concat(bf16.ubit(3, '0'))
            # Sticky bit:
            # if exp_diff is larger than mantissa bits + hidden bit + GRS then all shift mantissa is sticky
            # else sticky is exp_diff - 3 ~ 0
            if exp_diff_abs >= bf16.Bfloat16.mantissa_bits + 1 + 3:
                mant_sticky = mant_shift_in.reduceor()
            else:
                mant_sticky = (mant_shift_in << (bf16.Bfloat16.mantissa_bits + 1 + 3 - exp_diff_abs)).reduceor()
            
            mant_shift = (mant_shift_in >> exp_diff_abs).concat(mant_sticky)

            # Add/Sub Shifted mantissa /w LZA
            # Invert flag: mantissa of lesser exponent for Sub
            if subtract_mant == bf16.bit(1, '1'):
                if a_exp_gt_b:
                    mant_inv_b = True
                # With equal exponent, invert lesser mantissa
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
            
            # Invert mantissa
            # mant_unshift with grs= [7+3:0] 
            # mant_shift with grs= [7+3:0]
            if mant_inv_a:
                mant_add_in_a = ~mant_shift
                mant_add_in_b = mant_unshift
            elif mant_inv_b:
                mant_add_in_a = mant_shift
                mant_add_in_b = ~mant_unshift
            else:
                mant_add_in_a = mant_shift
                mant_add_in_b = mant_unshift
            
            # Add mantissa (Including sub)
            # mant_add[11:0] (Including carry)
            mant_add = mant_add_in_a + mant_add_in_b
            ret_mant_0 = mant_add
            # Normalize with LZA shift amount when Sub
            # apply lza later
            # Sign calculation with Sub result
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
            # Exponent adjust with normalize/Add
            # Sub: in case of 0x.xxxxxxx shift amount = lzc - 1
            if subtract_mant:
                exp_adj = bf16.hwutil.leading_zero_count(ret_mant_0) - 1
            # Add, carry occured
            elif ret_mant_0[ret_mant_0.bitwidth-1] == bf16.ubit(1, '1'):
                exp_adj = 1
            # Add, carry not occured
            else:
                exp_adj = 0
                
            ret_exp_1 = ret_exp_0 + bf16.sbit(ret_exp_0.bitwidth + 2, bin(exp_adj))

            # Round and Postnormalization
            if ret_mant_0 == bf16.ubit(12, '1111111111111111'):
                # Postnormalize
                ret_exp_2 = ret_exp_1 + bf16.sbit(ret_exp_0.bitwidth + 2, 1)
                ret_mant_1 = bf16.ubit(8, '0')
            else:
                # round
                ret_exp_2 = ret_exp_1
                ret_mant_1 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_0, 8, 1)
            
            # Check for overflow/underflow

            # remove hidden bit
            ret_mant_2 = ret_mant_1[bf16.Bfloat16.mantissa_bits - 1:0]
        else:
            ret_mant_2 = ret_mant_0
            ret_exp_2 = ret_exp_0
            
        # Remove sign bit from exponent
        ret_exp_bit_2 = bf16.bit(a_exp.bitwidth, ret_exp_2.bin)
        print(ret_mant_0)
        print(ret_mant_1)
        print(ret_mant_2)

        # Compose BF16
        add = bf16.Bfloat16.compose_bf16(ret_sign_0, ret_exp_bit_2, ret_mant_2)
        return add
