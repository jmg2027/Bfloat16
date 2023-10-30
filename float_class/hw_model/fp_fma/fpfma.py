from ..utils.commonimport import *
#from ..utils.utils import isnan, isinf, iszero


class FloatFMA:
    # https://repositories.lib.utexas.edu/bitstream/handle/2152/13269/quinnelle60861.pdf?sequence=2
    # a * b + c
    def __init__(self, a: FPBitT, b: FPBitT, c: FPBitT) -> None:
        self.a = a
        self.b = b
        self.c = c

    def excute(self) -> FPBitT:
        a_sign, a_exp, a_mant_nohidden = self.a
        b_sign, b_exp, b_mant_nohidden = self.b
        c_sign, c_exp, c_mant_nohidden = self.c
        a_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{b_mant_nohidden}')
        c_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{c_mant_nohidden}')

        a_exp_signed = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        b_exp_signed = sbit(b_exp.bitwidth + 2, f'0{b_exp.bin}')
        c_exp_signed = sbit(c_exp.bitwidth + 2, f'0{c_exp.bin}')
        bias_signed = sbit(a_exp.bitwidth + 2, bin(fp32_config.bias))

        ret_sign_0 : bit = bit(1, '0')
        ret_exp_0 : bit = bit(1, '0')
        ret_mant_0: bit = bit(1, '0')

        # Special cases
        #input
        isnormal = False
        # fma(nan,?,?) -> nan
        # nan inputs clearance
        if isnan(self.a) or isnan(self.b) or isnan(self.c):
            ret_sign_0 = bit(1, '0')
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_0 = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * 0) + ? -> nan
        elif (isinf(self.a) and iszero(self.b)) or (isinf(self.b) and iszero(self.a)):
            ret_sign_0 = bit(1, '0')
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_0 = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * ?) + -inf, (-inf * ?) + inf -> nan
        elif (isinf(self.a) or isinf(self.b)) and isinf(self.c) and (a_sign ^ b_sign ^ c_sign):
            ret_sign_0 = bit(1, '0')
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_0 = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * ?) + ? -> inf, (-inf * ?) + ? -> -inf
        elif (isinf(self.a) or isinf(self.b)):
            ret_sign_0 = a_sign ^ b_sign
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_0 = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # (? * ?) + inf -> inf, (? * ?) - inf -> inf
        elif isinf(self.c):
            ret_sign_0 = c_sign
            ret_exp_0 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_0 = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # FIX: add zero adddend/product cases
        # Zero product case
        elif iszero(self.a) or iszero(self.b):
            # result is c
            ret_sign_0 = c_sign
            ret_exp_0 = c_exp
            ret_mant_0 = c_mant_nohidden
        else:
            isnormal = True

        if isnormal:
            # Define precision bitwidth
            precision_bit = fp32_config.mantissa_bits + 1

            # Zero addend case
            if iszero(self.c):
                # set flag for normal calculation
                addend_is_zero = True
            else:
                addend_is_zero = False

            # Sign precalculation
            p_sign = a_sign ^ b_sign
            subtract_mant = c_sign ^ p_sign

            # Product mantissa
            p_mant_us = a_mant_us * b_mant_us

            # Product is 1x.xxx...
            # Product exponent calculation and normalize
            # FIX: addend to 01.xxx_xxxx
            if p_mant_us[2*precision_bit-1] == ubit(1, '1'):
                # 1x.xxx...
                # exp + 1
                p_exp_signed = a_exp_signed + b_exp_signed - bias_signed + sbit(2, '01')
            else:
                # 0x.xxx...
                # left shift mantissa
                p_exp_signed = a_exp_signed + b_exp_signed - bias_signed
                p_mant_us.__ilshift__(1)

                
            #print('a_exp + b_exp', int(a_exp_signed + b_exp_signed - bias_signed))
            #print('c_exp_signed:',int(c_exp_signed))
            #print('p_exp_signed:',int(p_exp_signed))
            # Exponent difference
            exp_diff = c_exp_signed - p_exp_signed
            exp_diff_abs = abs(int(exp_diff))

            # Set flags
            #print('exp_diff', int(exp_diff))
            c_exp_gt_p = exp_diff > sbit(1, '0')
            c_exp_eq_p = exp_diff == sbit(1, '0')
            c_mant_gt_p = (c_mant_us.concat(ubit(precision_bit, '0'))) >= p_mant_us
            #print(c_mant_us)
            #print('a_mant_us', hex(int(a_mant_us)))
            #print('b_mant_us', hex(int(b_mant_us)))
            #print('p_mant_us', hex(int(p_mant_us)))

            # Prenormalized Exponent
            if c_exp_gt_p:
                ret_exp_0 = c_exp_signed
            else:
                ret_exp_0 = p_exp_signed

            # Mantissa swap and shift
            # c_mant_us = [7:0]
            # p_mant_us = [15:0]
            # mant_unshift = [15+3:0] -> grs
            # mant_shift = [15+3:0] -> grs
            # mant_shift_in = [15+2:0] -> grs, sticky bit will concat later
            # sum = cx.xxx_xxxx_xxxx_xxxx_xxxx_xxxx_GRS (3p+4 bits)

            # This machine assumes no denormalized number (All subnormal num = zero)
            # Addend anchor
            # All product bits are sticky
            # Cx.xxx_xxxx_00S
            # ANCHOR CASE DOES NOT MATCH TO RESULT: NEED DEBUG MORE
#            addend_anchor_case = int(exp_diff) >= precision_bit + 3
            addend_anchor_case = False
            # Product anchor
            # All addend bits are sticky
            # Cx.xxx_xxxx_xxxx_xxxx_00S
#            product_anchor_case = int(exp_diff) <= -(2 * precision_bit + 2)
            product_anchor_case = False

#            print('addend_anchor: ', addend_anchor_case)
#            print('product_anchor: ', product_anchor_case)
            if addend_anchor_case:
                # add will be just 1+p+3 bits
                mant_sticky = p_mant_us.reduceor()
                c_mant_shft = c_mant_us.concat(ubit(2, '0'))
                p_mant_shft = ubit(precision_bit+2, '0').concat(mant_sticky)
                if subtract_mant == ubit(1, '1'):
                    p_mant_shft = -p_mant_shft
                # match to 1+2p+3 bits
                # FIX: add subtract cases: addend: 01.xxx....
                # FIX: not need to "add"
                sum = ubit.add_bitstring(c_mant_shft, p_mant_shft).concat(ubit(precision_bit, '0'))
            elif product_anchor_case:
                # sum will be just 1+2p+3 bits
                mant_sticky = p_mant_us.reduceor()
                c_mant_shft = ubit(2*precision_bit+2, '0').concat(mant_sticky)
                p_mant_shft = p_mant_us.concat(bit(2, '0'))
                if subtract_mant == ubit(1, '1'):
                    c_mant_shft = -c_mant_shft
                sum = ubit.add_bitstring(c_mant_shft, p_mant_shft)
            else:
                mant_shift_amt = exp_diff_abs
                # match to 2p+grs bits
                # make sum mantissa to '0'01.xxx...
                # FIX: make sum just 11bits (not now)
                if c_exp_gt_p:
                    mant_unshift = c_mant_us.concat(ubit(3 + precision_bit, '0'))
                    mant_shift_in = p_mant_us.concat(ubit(3, '0'))
                elif c_exp_eq_p:
                    if c_mant_gt_p:
                        # In case of equal exponent, shift lesser mantissa
                        mant_unshift = c_mant_us.concat(ubit(3 + precision_bit, '0'))
                        mant_shift_in = p_mant_us.concat(ubit(3, '0'))
                    else:
                        mant_unshift = p_mant_us.concat(ubit(3, '0'))
                        mant_shift_in = c_mant_us.concat(ubit(precision_bit + 3, '0'))
                else:
                    mant_unshift = p_mant_us.concat(ubit(3, '0'))
                    mant_shift_in = c_mant_us.concat(ubit(precision_bit + 3, '0'))
                mant_shift = mant_shift_in >> mant_shift_amt

                # Invert mantissa
                if subtract_mant == ubit(1, '1'):
                    mant_add_in_c = -mant_shift
                    mant_add_in_p = mant_unshift
                else:
                    mant_add_in_c = mant_shift
                    mant_add_in_p = mant_unshift

                #print(hex(int(mant_unshift)))
                #print(hex(int(mant_shift_in)))
                #print(hex(int(mant_shift)))
                #print(mant_shift_amt)

                #print('c>p', c_exp_gt_p)
                #print(repr(mant_shift))
                #print(repr(mant_unshift))
                #print('mant_c_shft: ', mant_add_in_c)
                #print('mant_p_shft: ', mant_add_in_p)
                #print('mant_shift_amt', mant_shift_amt)
                #print('mant_c_shft: ', hex(int(mant_add_in_c)))
                #print('mant_p_shft: ', hex(int(mant_add_in_p)))

                # Add mantissa (Including sub)
                # mant_add[1+2p+3:0] (Including carry)
                # Not to discard carry
                mant_add = ubit.add_bitstring(mant_add_in_c, mant_add_in_p)
                sum = ubit(2 * precision_bit + 4, mant_add.bin)

            ret_mant_0 = sum
            #print('fma_sum', repr(ret_mant_0))
            # CHECK: for mantissa carry bit (such as 1.xx... -> 01.xx... before addition)

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
            # Sum format: cx.xxx....xx (2p+4 bits)
            # Sub: in case of 00.0...0xx shift amount = lzc + 1
            if subtract_mant == bit(1, '1'):
                shift_amt = hwutil.leading_zero_count(ret_mant_0[ret_mant_0.bitwidth-2:0]) 
                ret_mant_1 = ret_mant_0 << shift_amt
                exp_adj = -shift_amt
            # Add, carry occured
            elif ret_mant_0[ret_mant_0.bitwidth-1] == ubit(1, '1'):
                shift_amt = 1
                exp_adj = shift_amt
                ret_mant_1 = ret_mant_0 >> shift_amt
            # Add, carry not occured
            else:
                exp_adj = 0
                ret_mant_1 = ret_mant_0
            ret_exp_1 = ret_exp_0 + sbit(ret_exp_0.bitwidth + 2, bin(exp_adj))

            # adjusted exponent: remove mantissa carry
            ret_mant_2 = ret_mant_1[ret_mant_0.bitwidth-2:0]

            # Round and Postnormalization
            # Postnormalize: when mant = 1.111_1111_R (R = 1)
            postnorm_mant = ret_mant_2[ret_mant_2.bitwidth-1:ret_mant_2.bitwidth-fp32_config.mantissa_bits-2]
            postnorm_cond = ubit(fp32_config.mantissa_bits + 2, '1' * (fp32_config.mantissa_bits + 2))
            if postnorm_mant == postnorm_cond:
                ret_exp_2 = ret_exp_1 + sbit(ret_exp_0.bitwidth + 2, '01')
                ret_mant_3 = ubit(fp32_config.mantissa_bits + 1, '0')
            else:
                # round
                ret_exp_2 = ret_exp_1
                ret_mant_3 = hwutil.round_to_nearest_even_bit(ret_mant_2, fp32_config.mantissa_bits + 1)
            
            # Overflow case: make inf
            # ret_exp_2: 01_0000_0000 ~
            if ret_exp_2[fp32_config.exponent_bits + 1:0] >= sbit(fp32_config.exponent_bits + 1 , bin((1 << fp32_config.exponent_bits) - 1)):
                ret_exp_2 = sbit(fp32_config.exponent_bits, bin((fp32_config.exp_max + fp32_config.bias)))
                ret_mant_3 = ubit(fp32_config.mantissa_bits, '0')
            # Underflow case: make zero
            # ret_exp_2: 11_0000_0000 ~
            elif ret_exp_2[fp32_config.exponent_bits + 1:0] <= sbit(fp32_config.exponent_bits + 1, bin(0)):
                ret_exp_2 = sbit(fp32_config.exponent_bits, '0')
                ret_mant_3 = ubit(fp32_config.mantissa_bits, '0')

            # remove hidden bit
            ret_mant_4 = ret_mant_3[fp32_config.mantissa_bits - 1:0]

            ret_sign = ret_sign_0
            ret_exp_signed = ret_exp_2
            ret_mant = ret_mant_4

            #print('exp 0', repr(ret_exp_0))
            #print('exp 1', repr(ret_exp_1))
            #print('exp 2', repr(ret_exp_2))

            #print('exp 0', (int(ret_exp_0)))
            #print('exp 1', (int(ret_exp_1)))
            #print('exp 2', (int(ret_exp_2)))

            #print('mant 0', repr(ret_mant_0))
            #print('mant 1', repr(ret_mant_1))
            #print('mant 2', repr(ret_mant_2))
            #print('mant 3', repr(ret_mant_3))
            #print('mant 4', repr(ret_mant_4))

            #print('mant 0', hex(int(ret_mant_0)))
            #print('mant 1', hex(int(ret_mant_1)))
            #print('mant 2', hex(int(ret_mant_2)))
            #print('mant 3', hex(int(ret_mant_3)))
            #print('mant 4', hex(int(ret_mant_4)))

        else:
            ret_sign = ret_sign_0
            ret_exp_signed = ret_exp_0
            ret_mant = ret_mant_0

        # Remove sign bit from exponent
        ret_exp = bit(fp32_config.exponent_bits, ret_exp_signed.bin)

        return ret_sign, ret_exp, ret_mant
