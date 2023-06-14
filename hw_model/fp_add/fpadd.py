import bf16.bf16 as bf16

class FloatAddition:
    def __init__(self, a: 'bf16.Bfloat16', b: 'bf16.Bfloat16'):
        self.a = a
        self.b = b

    def add(self):
        # Decompose Bfloat16 to bitstring class
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        b_sign, b_exp, b_mant_nohidden = self.b.decompose_bf16()
        a_mant_us = bf16.ubit(self.a.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant_us = bf16.ubit(self.b.mantissa_bits + 1, f'1{b_mant_nohidden}')

        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = bf16.sbit(b_exp.bitwidth + 2, f'0{b_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        # Special cases
        #input
        isnormal = False
        # nan + ? -> nan
        # nan inputs clearance
        if self.a.isnan() or self.b.isnan():
            ret_sign_0 = bf16.bit(1, '0')
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max + bf16.Bfloat16.bias))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf + inf = inf
        # -inf + -inf = -inf
        elif (self.a.isinf() and self.b.isinf() and (self.a.sign == self.b.sign)):
            ret_sign_0 = a_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max + bf16.Bfloat16.bias))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # -inf + +inf = nan
        elif (self.a.isinf() and self.b.isinf() and (self.a.sign != self.b.sign)):
            ret_sign_0 = bf16.bit(1, '0')
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max + bf16.Bfloat16.bias))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bin(bf16.Bfloat16.mant_max))
        # inf + not inf = inf
        # -inf + not inf = -inf
        elif self.a.isinf():
            ret_sign_0 = a_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max + bf16.Bfloat16.bias))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        elif self.b.isinf():
            ret_sign_0 = b_sign
            ret_exp_0 = bf16.bit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.exp_max + bf16.Bfloat16.bias))
            ret_mant_0 = bf16.bit(bf16.Bfloat16.mantissa_bits, bf16.Bfloat16.mantissa_bits * '0')
        # zero + x = x
        elif self.a.iszero():
            ret_sign_0 = b_sign
            ret_exp_0 = b_exp
            ret_mant_0 = b_mant_nohidden
        elif self.b.iszero():
            ret_sign_0 = a_sign
            ret_exp_0 = a_exp
            ret_mant_0 = a_mant_nohidden
        # normal case
        else:
            isnormal = True

        
        if isnormal:
            # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6489167
            # Exponent calculation
            # Calculate Exponent differences
            exp_diff = a_exp_signed - b_exp_signed
            #print('a_exp_signed:',a_exp_signed)
            #print('b_exp_signed:',b_exp_signed)
            exp_diff_abs = abs(int(exp_diff))
            #print('sum exp diff', exp_diff)
            # Set flags
            a_exp_gt_b = exp_diff > bf16.sbit(1, '0')
            a_exp_eq_b = exp_diff == bf16.sbit(1, '0')
            a_mant_gt_b = a_mant_us >= b_mant_us
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
            # Shift mantissa of lesser exponent
            if a_exp_gt_b:
                mant_unshift = a_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = b_mant_us.concat(bf16.ubit(2, '0'))
            elif a_exp_eq_b:
                # In case of equal exponent, shift lesser mantissa
                if a_mant_gt_b:
                    mant_unshift = a_mant_us.concat(bf16.ubit(3, '0'))
                    mant_shift_in = b_mant_us.concat(bf16.ubit(2, '0'))
                else:
                    mant_unshift = b_mant_us.concat(bf16.ubit(3, '0'))
                    mant_shift_in = a_mant_us.concat(bf16.ubit(2, '0'))
            else:
                mant_unshift = b_mant_us.concat(bf16.ubit(3, '0'))
                mant_shift_in = a_mant_us.concat(bf16.ubit(2, '0'))
            
            # Sticky bit:
            # if exp_diff is larger than mantissa bits + hidden bit + GR then all shift mantissa is stick
            # else sticky is exp_diff - 3 ~ 0
            if exp_diff_abs >= bf16.Bfloat16.mantissa_bits + 1 + 2:
                mant_sticky = mant_shift_in.reduceor()
            else:
                mant_sticky = (mant_shift_in << (bf16.Bfloat16.mantissa_bits + 1 + 2 - exp_diff_abs)).reduceor()
            
            mant_shift = (mant_shift_in >> exp_diff_abs).concat(mant_sticky)

            # Invert flag: mantissa of lesser exponent -> shifted mantissa
            
            # Invert mantissa
            # mant_unshift with grs= [7+3:0] 
            # mant_shift with grs= [7+3:0]
            if subtract_mant == bf16.bit(1, '1'):
                mant_add_in_a = -mant_shift
                mant_add_in_b = mant_unshift
            else:
                mant_add_in_a = mant_shift
                mant_add_in_b = mant_unshift
            #print('exp_a: ', int(a_exp))
            #print('exp_b: ', int(b_exp))
            #print('mant_shift_in', repr(mant_shift_in))
            #print('mant_shift', repr(mant_shift))
            #print('mant_a: ', a_mant_us)
            #print('mant_b: ', b_mant_us)
            #print(repr(mant_shift))
            #print(repr(mant_unshift))
            #print('mant_shift: ', mant_add_in_a)
            #print('mant_unshift: ', mant_add_in_b)
            # Add mantissa (Including sub)
            # mant_add[11:0] (Including carry)
            # Not to discard carry
            mant_add = bf16.ubit.add_bitstring(mant_add_in_a, mant_add_in_b)
            ret_mant_0 = bf16.ubit(bf16.Bfloat16.mantissa_bits + 5, mant_add.bin)
            #print('sum', repr(mant_add))
            #print('ret_mant_0', repr(ret_mant_0[ret_mant_0.bitwidth-2:0]))

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
            
            # if mant: 11.11_1111_1R, then after normalize: 1.111_1111_1 -> 10.00_0000_00
            # if mant: 10.11_1111_1R, then after normalize: 1.011_1111_1 -> x
            # if mant: 01.11_1111_1R, then after normalize: 01.11_1111_1R -> 10.00_0000_00 (when R = 1)

            ret_exp_1 = bf16.sbit.add_bitstring(ret_exp_0, bf16.sbit(ret_exp_0.bitwidth, bin(exp_adj)))

            # adjusted exponent: remove mantissa carry
            ret_mant_2 = ret_mant_1[ret_mant_0.bitwidth-2:0]

            # Round and Postnormalization
            # Postnormalize: when mant = 1.111_1111_R (R = 1)
            # 1.111_1111
            if ret_mant_2[ret_mant_2.bitwidth-1:2] == bf16.ubit(bf16.Bfloat16.mantissa_bits + 2, '1' * (bf16.Bfloat16.mantissa_bits + 2)):
                ret_exp_2 = ret_exp_1 + bf16.sbit(ret_exp_0.bitwidth + 2, '01')
                ret_mant_3 = bf16.ubit(8, '10000000')
            else:
                # round
                ret_exp_2 = ret_exp_1
                ret_mant_3 = bf16.hwutil.round_to_nearest_even_bit(ret_mant_2, 8)
            
            #print('exp0',ret_exp_0)
            #print('exp1',ret_exp_1)
            #print('exp2',ret_exp_2)
            #print('mant0',ret_mant_0)
            #print('mant1',ret_mant_1)
            #print('mant2',ret_mant_2)
            #print('mant3',ret_mant_3)
            
            # Overflow case: make inf
            # ret_exp_2: 01_0000_0000 ~
            if ret_exp_2[bf16.Bfloat16.exponent_bits + 1:0] >= bf16.sbit(bf16.Bfloat16.exponent_bits + 1 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_2 = bf16.sbit(bf16.Bfloat16.exponent_bits, bin((bf16.Bfloat16.exp_max + bf16.Bfloat16.bias)))
                ret_mant_3 = bf16.ubit(bf16.Bfloat16.mantissa_bits, '0')
            # BF16 assumes denormalized number as zero
            # There is no denormalized output in addition
            # Underflow case: make zero
            # ret_exp_2: 11_0000_0000 ~
            elif ret_exp_2[bf16.Bfloat16.exponent_bits + 1:0] <= bf16.sbit(bf16.Bfloat16.exponent_bits + 1, bin(0)):
                ret_exp_2 = bf16.sbit(bf16.Bfloat16.exponent_bits, '0')
                ret_mant_3 = bf16.ubit(bf16.Bfloat16.mantissa_bits, '0')

            # remove hidden bit
            ret_mant_4 = ret_mant_3[bf16.Bfloat16.mantissa_bits - 1:0]
        # Special case
        else:
            ret_mant_4 = ret_mant_0
            ret_exp_2 = ret_exp_0
            
        # Remove sign bit from exponent
        ret_exp_bit_2 = bf16.bit(bf16.Bfloat16.exponent_bits, ret_exp_2.bin)

        # Compose BF16
        add = bf16.Bfloat16.compose_bf16(ret_sign_0, ret_exp_bit_2, ret_mant_4)
        return add
