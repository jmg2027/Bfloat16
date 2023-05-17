import bf16.bf16 as bf16

class FloatFMA:
    # https://repositories.lib.utexas.edu/bitstream/handle/2152/13269/quinnelle60861.pdf?sequence=2
    # a * b + c
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
        isnormal = False
        # fma(nan,?,?) -> nan
        # nan inputs clearance
        if self.a.isnan() or self.b.isnan() or self.c.isnan():
            ret_sign_0 = a_sign ^ b_sign ^ c_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # (inf * 0) + ? -> nan
        elif (self.a.isinf() and self.b.iszero()) or (self.b.isinf() and self.a.iszero()):
            ret_sign_0 = a_sign ^ b_sign ^ c_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # (inf * ?) + -inf, (-inf * ?) + inf -> nan
        elif (self.a.isinf() or self.b.isinf()) and self.c.isinf() and (a_sign ^ b_sign ^ c_sign):
            ret_sign_0 = a_sign ^ b_sign ^ c_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # (inf * ?) + ? -> inf, (-inf * ?) + ? -> -inf
        elif (self.a.isinf() or self.b.isinf()):
            ret_sign_0 = a_sign ^ b_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # (? * ?) + inf -> inf, (? * ?) - inf -> inf
        elif self.c.isinf():
            ret_sign_0 = c_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        else:
            isnormal = True


        if isnormal:
            # Define precision bitwidth
            precision_bit = bf16.Bfloat16.mantissa_bits + 1

            # Sign precalculation
            subtract_mant = c_sign ^ p_sign
            p_sign = a_sign ^ b_sign

            # Exponent calculation
            # If product is zero
            if self.a.iszero() or self.b.iszero():
                p_exp_signed = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, '0')
            # Normal case
            else:
                p_exp_signed = a_exp_signed + b_exp_signed - bias_signed

            # c + product
            # Exponent difference
            exp_diff = c_exp_signed - p_exp_signed
            exp_diff_abs = abs(int(exp_diff))
            
            # Set flags
            c_exp_gt_p = exp_diff > bf16.sbit(1, '0')
            c_exp_eq_p = exp_diff == bf16.sbit(1, '0')
            c_mant_gt_p = c_mant_us >= p_mant_us

            exp_larger = c_exp_signed if c_exp_gt_p else p_exp_signed

            # This machine assumes no denormalized number (All subnormal num = zero)
            # Addend anchor
            # All product bits are sticky
            # C1.xxx_xxxx_GRS | xx.xx_xxxx_xxxx_xxxx
            # < 1+h+m+3 = p+4 >
            # <            1+p+3+2p                >
            # -1 for product point
            addend_anchor_case = int(exp_diff) >= precision_bit + 2
            # Product anchor
            # All addend bits are sticky
            # Cxx.xx_xxxx_xxxx_xxxx_GRS | 1.xxx_xxxx
            # <   1+(h+m)*2+3 = 2p+4    >
            # <              1+2p+3+p              >
            product_anchor_case = int(exp_diff) <= -(2 * precision_bit + 1)
                
            if addend_anchor_case:
                # All product sticky
                addend_shift_amt = 0
            elif product_anchor_case:
                # All addened sticky
                addend_shift_amt = 3 * precision_bit + 4
            else:
                # Normal case
                addend_shift_amt = precision_bit + 3 - int(exp_diff)
            
            # Product
            # Product mantissa: (1+7) * 2 = 16 bits
            p_mant_us = (a_mant_us * b_mant_us)

            # Place product into 3p+4 bitwidth
            # ??
            p_mant_shift = p_mant_us << 3

            # Addened
            # 0....0 | 1.xxx_xxxx | 0.....00GR| Sticky bits
            # <shift><      p     ><2p+4-shift><    rest   >
            # <                3p+4+rest                   >

            addened_shift_with_sticky = (c_mant_us.concat(bf16.bit(3 * precision_bit + 4, '0'))) >> addend_shift_amt
            addened_shift = addened_shift_with_sticky[:precision_bit]
            addened_sticky = addened_shift_with_sticky[precision_bit-1:0].reduceor()


            

            # 2.8 Multiple Path Fused Multiplier-Adder
            # ref) http://i.stanford.edu/pub/cstr/reports/csl/tr/96/711/CSL-TR-96-711.pdf
            """
             By swapping the operands such that the smaller operand is always
            subtracted from the larger operand, the conversion in step 4 is eliminated in all
            cases except for equal exponents.
            In the case of equal exponents, it is possible
            that the result of step 3 may be negative.
            the conversion addition in step 4 and the rounding addition in step 7 become mutually exclusive
            by appropriately swapping the operands.
            In the case of effective addition, there is never any cancellation of the results.
Accordingly, only one full-length shift, an initial aligning shift, can ever be
needed.
For subtraction, when the
exponent difference d > 1, a full-length aligning shift may be needed.
 However, the result never requires more than a 1 bit left shift.
 Similarly if d <= 1, no full-length aligning shift is necessary, but a full-length normalizing shift may be
required in the case of subtraction.
 In this case, the 1 bit aligning shift and the conditional swap can be predicted from the low-order two bits of the exponents,
reducing the latency of this path.
CLOSE
for d <= 1, and FAR for d > 1
predict the number of leading zeros in the result directly from the input operands
            """
            
            
            # Prenormalized Exponent
            if c_exp_gt_p:
                ret_exp_0 = c_exp_signed
            else:
                ret_exp_0 = p_exp_signed

            # Mantissa swap and shift
            # Shift mantissa of smaller exponent
            # c_mant_us = [7:0]
            # p_mant_us = [15:0]
            # mant_unshift = [n+3:0] -> grs
            # mant_shift = [n+3:0] -> grs
            # mant_shift_in = [n+2:0] -> grs, sticky bit will concat later
            
            if c_exp_gt_p:
                mant_unshift = c_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = p_mant_us.concat(bf16.ubit(2, '0'))
            else:
                mant_unshift = c_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = p_mant_us.concat(bf16.ubit(2, '0'))

            

            # Add/Sub Shifted mantissa /w LZA
            # Invert flag: mantissa of lesser exponent for Sub
            mant_inv_c = False
            mant_inv_p = False
            if subtract_mant == bf16.bit(1, '1'):
                if c_exp_gt_p:
                    mant_inv_p = True
                # With equal exponent, invert lesser mantissa
                elif c_exp_eq_p:
                    if c_mant_gt_p:
                        mant_inv_p = True
                    else:
                        mant_inv_c = True
                else:
                    mant_inv_c = True
            else:
                mant_inv_c = False
                mant_inv_p = False

            # Invert mantissa
            # mant_unshift with grs= [7+3:0] 
            # mant_shift with grs= [7+3:0]
            if mant_inv_c:
                mant_add_in_a = -mant_shift
                mant_add_in_b = mant_unshift
            elif mant_inv_p:
                mant_add_in_a = mant_shift
                mant_add_in_b = -mant_unshift
            else:
                mant_add_in_a = mant_shift
                mant_add_in_b = mant_unshift
            
            # Add mantissa (Including sub)
            # mant_add[11:0] (Including carry)
            # Not to discard carry
            mant_add = bf16.ubit.add_bitstring(mant_add_in_a, mant_add_in_b)
            ret_mant_0 = bf16.ubit(bf16.Bfloat16.mantissa_bits + 5, mant_add.bin)
            # Normalize with LZA shift amount when Sub
            # apply lza later
            # Sign calculation with Sub result
            # Sign Caculation
            if c_exp_gt_p:
                ret_sign_0 = c_sign
            elif c_exp_eq_p:
                if c_mant_gt_p:
                    ret_sign_0 = c_sign
                else: 
                    ret_sign_0 = p_sign
            else:
                ret_sign_0 = p_sign

            # Exponent adjust with normalize/Add
            # Sub: in case of 00.0...0xx shift amount = lzc + 1
            if subtract_mant == bf16.bit(1, '1'):
                shift_amt = bf16.hwutil.leading_zero_count(ret_mant_0[ret_mant_0.bitwidth-2:0]) 
                ret_mant_1 = ret_mant_0 << shift_amt
                exp_adj = -shift_amt
            # Add, carry occured
            elif ret_mant_0[ret_mant_0.bitwidth-1] == bf16.ubit(1, '1'):
                shift_amt = 1
                exp_adj = shift_amt
                ret_mant_1 = ret_mant_0 >> shift_amt
            # Add, carry not occured
            else:
                exp_adj = 0
                ret_mant_1 = ret_mant_0
            ret_exp_1 = ret_exp_0 + bf16.sbit(ret_exp_0.bitwidth + 2, bin(exp_adj))

            # adjusted exponent: remove mantissa carry
            ret_mant_2 = ret_mant_1[ret_mant_0.bitwidth-2:0]

            # Round and Postnormalization
            if ret_mant_2 == bf16.ubit(bf16.Bfloat16.mantissa_bits + 4, f'{1*(bf16.Bfloat16.mantissa_bits + 4)}'):
                # Postnormalize
                ret_exp_2 = ret_exp_1 + bf16.sbit(ret_exp_0.bitwidth + 2, 1)
                ret_mant_3 = bf16.ubit(8, '0')
            else:
                # round
                ret_exp_2 = ret_exp_1
                ret_mant_3 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_2, 8, 1)
            
            # Overflow case: make inf
            if ret_exp_2 > bf16.sbit(ret_exp_2.bitwidth + 2 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_2 = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, bin((1 << self.a.exponent_bits) - 1))
                ret_mant_3 = 0
            # Underflow case: make zero
            elif ret_exp_2 < bf16.sbit(ret_exp_2.bitwidth + 2, bin(0)):
                ret_exp_2 = bf16.sbit(bf16.Bfloat16.exponent_bits, '0')
                ret_mant_3 = 0

            # remove hidden bit
            ret_mant_4 = ret_mant_3[bf16.Bfloat16.mantissa_bits - 1:0]

            ret_sign = ret_sign_0
            ret_exp_signed = ret_exp_2
            ret_mant = ret_mant_4
        else:
            ret_sign = ret_sign_0
            ret_exp_signed = ret_exp_0
            ret_mant = ret_mant_0

        # Remove sign bit from exponent
        ret_exp = bf16.bit(bf16.Bfloat16.exponent_bits, ret_exp_signed.bin)

        # Compose BF16
        fma = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
        return fma
