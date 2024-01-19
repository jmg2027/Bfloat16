from ..utils.commonimport import *
#from ..utils.utils import isnan, isinf, iszero

class FloatFMA:
    '''
    Large ulp error in FMA resulted from treating denormalized product as zero inside calculation logic
    '''
    # https://repositories.lib.utexas.edu/bitstream/handle/2152/13269/quinnelle60861.pdf?sequence=2
    # a * b + c
    def __init__(self, a: FPBitT, b: FPBitT, c: FPBitT) -> None:
        self.a = a
        self.b = b
        self.c = c
    
    def execute(self, algorithm = "SINGLE_PATH") -> FPBitT:
        if algorithm == "SINGLE_PATH":
            ret_sign, ret_exp, ret_mant = self.execute_single_path()
        elif algorithm == "MULTI_PATH":
            ret_sign, ret_exp, ret_mant = self.execute_multi_path()
        else:
            raise TypeError(f"FMA supports SINGLE_PATH, MULTI_PATH algorithm, not {algorithm}")
        return ret_sign, ret_exp, ret_mant
    
    def execute_single_path(self) -> FPBitT:
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
                p_exp_signed = a_exp_signed + b_exp_signed - bias_signed + sbit(a_exp.bitwidth + 2, '01')
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

                #print("mant_unshift", hex(int(mant_unshift)))
                #print("mant_shift_in", hex(int(mant_shift_in)))
                #print("mant_shift", hex(int(mant_shift)))
                #print("mant_shift_amt", mant_shift_amt)

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
        

    def execute_multi_path(self) -> FPBitT:
        a_sign, a_exp, a_mant_nohidden = self.a
        b_sign, b_exp, b_mant_nohidden = self.b
        c_sign, c_exp, c_mant_nohidden = self.c
        p_sign = a_sign ^ b_sign

        a_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{a_mant_nohidden}')
        b_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{b_mant_nohidden}')
        if iszero(self.c):
            c_mant_us = ubit(fp32_config.mantissa_bits + 1, f'0')
        else:
            c_mant_us = ubit(fp32_config.mantissa_bits + 1, f'1{c_mant_nohidden}')

        #a_exp_signed = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        #b_exp_signed = sbit(b_exp.bitwidth + 2, f'0{b_exp.bin}')
        #c_exp_signed = sbit(c_exp.bitwidth + 2, f'0{c_exp.bin}')
        #bias_signed = sbit(a_exp.bitwidth + 2, bin(fp32_config.bias))
        a_exp_signed = a_exp
        b_exp_signed = b_exp
        c_exp_signed = c_exp
        bias_signed = sbit(a_exp.bitwidth, bin(fp32_config.bias))

        p_exp_signed_overflow = sbit(10, a_exp_signed.bin) + sbit(10, b_exp_signed.bin) - bias_signed
        p_inf = int(p_exp_signed_overflow) > 254
        p_zero = int(p_exp_signed_overflow) < 1

        ret_sign_sp = bit(1, '0')
        ret_exp_sp = bit(1, '0')
        ret_mant_sp = bit(1, '0')

        NAN_MANT = 0x7E_0000

        # Special cases
        #input
        isnormal = False
        # fma(nan,?,?) -> nan
        # nan inputs clearance
        if isnan(self.a) or isnan(self.b) or isnan(self.c):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(NAN_MANT))
        # (inf * 0) + ? -> nan
        elif (isinf(self.a) and iszero(self.b)) or (isinf(self.b) and iszero(self.a)):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(NAN_MANT))
        # (inf * ?) + -inf, (-inf * ?) + inf -> nan
        elif p_inf and isinf(self.c) and (a_sign ^ b_sign ^ c_sign):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(NAN_MANT))
        # (inf * ?) + ? -> inf, (-inf * ?) + ? -> -inf
        elif p_inf:
            ret_sign_sp = a_sign ^ b_sign
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # (? * ?) + inf -> inf, (? * ?) - inf -> inf
        elif isinf(self.c):
            ret_sign_sp = c_sign
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # FIX: add zero adddend/product cases
        # Zero product case
        elif iszero(self.a) or iszero(self.b):
            # result is c
            ret_sign_sp= c_sign
            ret_exp_sp = c_exp
            ret_mant_sp = c_mant_nohidden
        else:
            isnormal = True

        # Normal case
        # Define precision bitwidth
        precision_bit = fp32_config.mantissa_bits + 1

        # Sign precalculation
        
        fma_sign = c_sign ^ p_sign

        # Exponent calculation
        # p_exp_signed ranges from 0 to 255 because of denormalized number is handled as zero
        # p_exp < 1
        p_exp_signed = bit(8, '0') if p_zero else bit(8, p_exp_signed_overflow[7:0].bin)
        exp_diff = sbit(9, f'0{p_exp_signed[7:0]}') - sbit(9, f'0{c_exp_signed}')
        #print(sbit(9, f'0{p_exp_signed[7:0]}'))
        #print(sbit(9, f'0{c_exp_signed}'))
        #print(exp_diff)

        # Product mantissa
        p_mant_us = bit(48, '0') if p_zero else (a_mant_us * b_mant_us)

        # Define 5 cases of FMA
        # Case 1) P_exp – C_exp ≥ 24, 49bit adder, C_man => sticky
        # Case 2) 2 ≤ P_exp – C_exp < 24, 49bit adder
        # Case 3) -2 < P_exp – C_exp < 2, considering cancellation, use 27bit adder
        # Case 4) -24 ≤ P_exp – C_exp ≤ -2, 25bit adder, R sticky
        # Case 5) P_exp – C_exp < -24, R, sticky

        # for special case when c is zero, case_5 is selected
        case_1 = int(exp_diff) >=24
        case_2 = 24 > int(exp_diff) >= 2
        #case_3 = 2 > int(exp_diff) > -2
        #case_4 = -2 >= int(exp_diff) >= -24
        case_3 = 2 > int(exp_diff) >= -2 and not iszero(self.c)
        case_4 = -2 > int(exp_diff) >= -24
        case_5 = (-24 > int(exp_diff)) or iszero(self.c)

        # 3 path design
        # Case 5
        # Case 1, 2, 4
        # Case 3

        # Case 5
        # all p_mant goes round/sticky
        # round condition: exp_diff = -25, p_mant[47] = 1
        # else c_mant is answer
        lsb_5 = c_mant_us[0]
        round_5 = p_mant_us[47]
        sticky_5 = bit(1, bin(p_mant_us[46:0] != bit(1, '0')))
        if int(exp_diff) == -25:
            round_up_5 = (lsb_5 & round_5) | (round_5 & sticky_5)
        else:
            round_up_5 = bit(1, '0')
        
        # postnormalize
        ret_mant_rounded_5 = ubit(c_mant_us.bitwidth, bin(int(c_mant_us) + (-int(round_up_5) if fma_sign else int(round_up_5))))
        if (ret_mant_rounded_5 == bit(c_mant_us.bitwidth, bin((1 << c_mant_us.bitwidth) - 1))) & bool(int(round_up_5.bin)):
            ret_mant_case_5 = bit(precision_bit - 1, '0')
            ret_exp_case_5 = (c_exp_signed + bit(1, '1'))
            # check overflow
            if int(c_exp_signed) >= 254:
                ret_exp_case_5 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        else:
            ret_mant_case_5 = ret_mant_rounded_5[precision_bit-2:0]
            ret_exp_case_5 = c_exp_signed
        ret_sign_case_5 = c_sign

        # result: ret_mant_case_5, ret_exp_case_5, ret_sign_case_5

        # Case 1, 2, 4
        if case_1:
            #shift_amt_124 = 23
            #shift_amt_124 = 24
            #shift_amt_124 = int(exp_diff)
            shift_amt_124 = 23 if int(exp_diff) > 47 else int(exp_diff)-24
        elif case_2:
            shift_amt_124 = int(exp_diff)
        elif case_4:
            shift_amt_124 = -int(exp_diff)
        else:
            # not used
            shift_amt_124 = 0

        p_mant_signed_49_pre_124 = sbit(p_mant_us.bitwidth + 1, f'0{p_mant_us.bin}')
        c_mant_signed_26_pre_124 = sbit(c_mant_us.bitwidth + 2, f'00{c_mant_us.bin}')

        # Shift lesser mantissa
        # case 1, 2: p > c
        # 1x.xxx -> exp + 1
        # case 4: c > p
        # 1.xxx -> exp

        if case_1:
            # p > c
            #c_mant_signed_49_pre_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_pre_124.bin}')
            #c_mant_signed_49_124 = -c_mant_signed_49_pre_124 if fma_sign.bin == '1' else c_mant_signed_49_pre_124
            c_mant_signed_26_124 = -c_mant_signed_26_pre_124 if fma_sign.bin == '1' else c_mant_signed_26_pre_124
            c_mant_signed_49_124 = c_mant_signed_26_124.sign_extend(p_mant_us.bitwidth + 1)
            #c_mant_signed_26_124 = -c_mant_signed_26_pre_124 if fma_sign.bin == '1' else c_mant_signed_26_pre_124
            #c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.concat(sbit(23, "0")).bin}')
            #c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.bin}')
            # JMG: fix here
            #shifted_input_124 = sbit(p_mant_us.bitwidth + 1, '0')
            #shift_in_124 = c_mant_signed_49_124.arith_rshift(shift_amt_124)
            shift_in_124 = c_mant_signed_49_124
            unshifted_input_124 = p_mant_signed_49_pre_124
            temp_exp_124 = p_exp_signed
            ret_sign_case_124 = p_sign
        elif case_2:
            # p > c
            c_mant_signed_26_124 = -c_mant_signed_26_pre_124 if fma_sign.bin == '1' else c_mant_signed_26_pre_124
            c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.concat(sbit(23, "0")).bin}')
            shift_in_124 = c_mant_signed_49_124
            unshifted_input_124 = p_mant_signed_49_pre_124
            temp_exp_124 = p_exp_signed
            ret_sign_case_124 = p_sign
        else:
            # c > p
            p_mant_signed_49_124 = -p_mant_signed_49_pre_124 if fma_sign.bin == '1' else p_mant_signed_49_pre_124
            c_mant_signed_26_124 = c_mant_signed_26_pre_124
            c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.concat(sbit(23, "0")).bin}')
            shift_in_124 = p_mant_signed_49_124
            unshifted_input_124 = c_mant_signed_49_124
            temp_exp_124 = c_exp_signed
            ret_sign_case_124 = c_sign
        
        shifted_input_124 = shift_in_124.arith_rshift(shift_amt_124)
        mant_add_before_norm_124 = shifted_input_124 + unshifted_input_124

        # normalize
        if mant_add_before_norm_124[47].bin == '1':
            # 1xx.xxx...
            mant_add_124 = mant_add_before_norm_124
            temp_norm_exp_124 = temp_exp_124 + bit(1, '1') 
            # check overflow
            if int(temp_exp_124) >= 254:
                mant_add_124 = sbit(mant_add_124.bitwidth, '00')
                temp_norm_exp_124 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        elif mant_add_before_norm_124[46].bin == '1':
            # 01x.xxx...
            mant_add_124 = sbit(mant_add_before_norm_124.bitwidth, f'{mant_add_before_norm_124[47:0].bin}0')
            temp_norm_exp_124 = temp_exp_124
        else:
            # 001.xxx...
            # no overflow case: if c_exp is inf then alredy inf
            mant_add_124 = sbit(mant_add_before_norm_124.bitwidth, f'{mant_add_before_norm_124[46:0].bin}00')
            temp_norm_exp_124 = temp_exp_124 - bit(1, '1')
        
        #print("mant_add_before_norm_124[47].bin == '1'", mant_add_before_norm_124[47].bin == '1')
        #print("mant_add_before_norm_124[46].bin == '1'", mant_add_before_norm_124[46].bin == '1')

        #print('p_mant_signed_49_pre_124', repr(p_mant_signed_49_pre_124))
        #print('c_mant_signed_25_pre_124', repr(c_mant_signed_26_pre_124))
        #print('shifted_input_124       ', repr(shifted_input_124       ))
        #print('unshifted_input_124     ', repr(unshifted_input_124     ))
        #print('temp_exp_124            ',  int(temp_exp_124            ))
        #print('mant_add_before_norm_124', repr(mant_add_before_norm_124[48:0]))
        #print('mant_add_124            ', repr(mant_add_124[48:0]            ))

        # round
        lsb_124 = mant_add_124[24]
        round_124 = mant_add_124[23]
        sticky_124 = sbit(1, bin(mant_add_124[22:0] != sbit(1, '0')))
        round_up_124 = (lsb_124 & round_124) | (round_124 & sticky_124)
        mant_rounded_124 = mant_add_124[48:24] + sbit(2, f'0{round_up_124.bin}')

        # postnormalize
        if (mant_rounded_124 == sbit(mant_rounded_124.bitwidth, bin((1 << mant_rounded_124.bitwidth) - 1))) & bool(int(round_up_124.bin)):
            ret_mant_case_124 = bit(precision_bit - 1, '0')
            ret_exp_case_124 = (temp_norm_exp_124 + bit(1, '1'))
            # check overflow
            if int(temp_norm_exp_124) >= 254:
                ret_exp_case_124 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        else:
            ret_mant_case_124 = bit(precision_bit-1, mant_rounded_124[precision_bit-2:0].bin)
            ret_exp_case_124 = temp_norm_exp_124

        #print(round_up_124)
        #print('mant_add_124[48:24]  ', repr(mant_add_124[48:24]))
        #print('mant_rounded_124     ', repr(mant_rounded_124))
        #print('ret_mant_case_124    ', repr(ret_mant_case_124))
        

        # Case 3
        # 2 + 4
        # exp_diff = -1
        # 001.0000000000000000000000000000000000000000000000 exp = 1 (49bits)
        # 001.00000000000000000000000RS exp = 2 (28bits)
        # 000.1000000000000000000000000000000000000000000000 exp = 2
        # 0001.10000000000000000000000RS exp = 2 + 2 - 2
        # 11.1
        # 01.1
        # 11.1 + 01.1 = 100.1

        # 2 + 2
        # exp_diff = 0
        # 001.0000000000000000000000000000000000000000000000 exp = 1
        # 001.00000000000000000000000RS exp = 1 (28bits)
        # 0010.00000000000000000000000RS exp = 1 + 2 - 1
         
        # 4 + 2
        # exp_diff = 1
        # 001.0000000000000000000000000000000000000000000000 exp = 2 (49bits)
        # 001.00000000000000000000000RS exp = 1 (28bits)
        # 000.10000000000000000000000RS exp = 2 (28bits)
        # 0011.00000000000000000000000RS exp = 2 + 2 - 1 (28bits)
        # xxx., exp = 2
        # scxx.xx... (50bits)
        c_mant_inv = -sbit(27, f'000{c_mant_us.bin}') if c_sign.bin == '1' else sbit(27, f'000{c_mant_us.bin}')
        #v = bit(25, f'0{c_mant_us.bin}')
        #mask = bit(25, f'{c_sign.bin}{c_exp.bin}{c_mant_nohidden.bin}')
        #ttt = sbit(25, f'{((v^mask) - mask).bin}')
        #c_mant_inv = sbit(27, f'{ttt.bin}')

        p_mant_inv = -sbit(50, f'00{p_mant_us.bin}') if p_sign.bin == '1' else sbit(50, f'00{p_mant_us.bin}')
        c_mant_signed_29_3 = sbit(29, f'{c_mant_inv.bin}00')
        p_mant_signed_50_3 = p_mant_inv
        c_mant_signed_shifted_29_3 = c_mant_signed_29_3.arith_rshift(1) if int(exp_diff) == 1 else c_mant_signed_29_3
        if int(exp_diff) == -1:
            p_mant_signed_shifted_50_3 = p_mant_signed_50_3.arith_rshift(1)
        elif int(exp_diff) == -2:
            p_mant_signed_shifted_50_3 = p_mant_signed_50_3.arith_rshift(2) if int(exp_diff) == -2 else p_mant_signed_50_3
        else:
            p_mant_signed_shifted_50_3 = p_mant_signed_50_3
        

        mant_add_29_3 = p_mant_signed_shifted_50_3[49:21] + c_mant_signed_shifted_29_3
        mant_add_50_3 = mant_add_29_3.concat(sbit(21, p_mant_signed_shifted_50_3[20:0].bin))
        mant_add_signed_50_3 = -mant_add_50_3 if mant_add_29_3[28].bin == '1' else mant_add_50_3
        
        
        
        # JMG TEST
        #shift_amt_3 = hwutil.leading_zero_count(mant_add_signed_50_3[48:0])
        shift_amt_3 = hwutil.leading_zero_count(mant_add_signed_50_3[48:23])
        if int(exp_diff) == 1:
            # p > c
            temp_exp_3 = bit(9, (p_exp_signed + bit(2, bin(2))).bin)
        else:
            # c >= p
            temp_exp_3 = bit(9, (c_exp_signed + bit(2, bin(2))).bin)

        # how to deal with underflow?
        # FIXED
        shifted_exp_3 = temp_exp_3 - bit(fp32_config.exponent_bits, bin(shift_amt_3)) if int(temp_exp_3) > shift_amt_3 else bit(fp32_config.exponent_bits, '0')
        mant_add_shifted_50_3 = ubit(mant_add_signed_50_3.bitwidth, (mant_add_signed_50_3 << shift_amt_3).bin)

        # round
        lsb_3 = mant_add_shifted_50_3[25]
        round_3 = mant_add_shifted_50_3[24]
        sticky_3 = ubit(1, bin(mant_add_shifted_50_3[23:0] != ubit(1, '0')))
        round_up_3 = (lsb_3 & round_3) | (round_3 & sticky_3)
        mant_rounded_3 = mant_add_shifted_50_3[49:25] + round_up_3

        # postnormalize
        if mant_rounded_3 == ubit(mant_rounded_3.bitwidth, bin((1 << mant_rounded_3.bitwidth) - 1)):
            ret_mant_case_3 = bit(precision_bit - 1, '0')
            ret_exp_case_3 = (shifted_exp_3 + bit(1, '1'))
            # check overflow
        #    if int(shifted_exp_3) >= 254:
        #        ret_exp_case_3 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        else:
            ret_mant_case_3 = mant_rounded_3[precision_bit-2:0]
            ret_exp_case_3 = shifted_exp_3

        # FIXED
        # check overflow
        if int(shifted_exp_3) >= 255:
            ret_exp_case_3 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_case_3 = bit(precision_bit - 1, '0')
        ret_sign_case_3 = bit(1, mant_add_29_3[28].bin)

        #print('c_mant_us ', repr(c_mant_us))
        #print('p_mant_us ', repr(p_mant_us))
        #print('c_mant_inv', repr(c_mant_inv))
        #print('p_mant_inv', repr(p_mant_inv))
        #print('c_mant_signed_shifted_29_3', repr(c_mant_signed_shifted_29_3))
        #print('p_mant_signed_shifted_50_3', repr(p_mant_signed_shifted_50_3[49:21]))
        #print('mant_add_29_3', repr(mant_add_29_3))
        #print('mant_add_50_3', repr(mant_add_50_3))
        #print('mant_add_signed_50_3', repr(mant_add_signed_50_3))

        #print('mant_add_signed_50_3[47:0]', mant_add_signed_50_3[47:0])

        #print('shift_amt_3', shift_amt_3)
        #print('temp_exp_3', int(temp_exp_3))
        #print('shifted_exp_3', int(shifted_exp_3))
        #print('mant_add_signed_50_3', mant_add_signed_50_3)
        #print('mant_add_shifted_50_3', mant_add_shifted_50_3)
        #print('mant_add_shifted_50_3[48:24]', mant_add_shifted_50_3[48:24])
        #print('round_up_3', round_up_3)
        #print('mant_rounded_3', repr(mant_rounded_3))

        # final output mux
        if case_1 or case_2 or case_4:
            ret_sign_normal = ret_sign_case_124
            ret_exp_normal = ret_exp_case_124
            ret_mant_normal = ret_mant_case_124
        elif case_3:
            ret_sign_normal = ret_sign_case_3
            ret_exp_normal = ret_exp_case_3
            ret_mant_normal = ret_mant_case_3
        else:
            ret_sign_normal = ret_sign_case_5
            ret_exp_normal = ret_exp_case_5
            ret_mant_normal = ret_mant_case_5

        if isnormal:
            ret_sign = ret_sign_normal
            ret_exp_signed = ret_exp_normal
            ret_mant = ret_mant_normal
        else:
            ret_sign = ret_sign_sp
            ret_exp_signed = ret_exp_sp
            ret_mant = ret_mant_sp

        #print('exp_diff', int(exp_diff))

        #print('case_1', case_1)
        #print('case_2', case_2)
        #print('case_3', case_3)
        #print('case_4', case_4)
        #print('case_5', case_5)

        #print('ret_exp_case_5', int(ret_exp_case_5))
        #print('ret_exp_case_124', int(ret_exp_case_124))
        #print('ret_exp_case_3', int(ret_exp_case_3))

        #print('ret_mant_case_5', repr(ret_mant_case_5))
        #print('ret_mant_case_124', repr(ret_mant_case_124))
        #print('ret_mant_case_3', repr(ret_mant_case_3))

        #print('ret_mant_case_5', int(ret_mant_case_5))
        #print('ret_mant_case_124', int(ret_mant_case_124))
        #print('ret_mant_case_3', int(ret_mant_case_3))

        #print('ret_mant_case_5', hex(int(ret_mant_case_5)))
        #print('ret_mant_case_124', hex(int(ret_mant_case_124)))
        #print('ret_mant_case_3', hex(int(ret_mant_case_3)))

        #print('ret_sign_case_5', ret_sign_case_5)
        #print('ret_sign_case_124', ret_sign_case_124)
        #print('ret_sign_case_3', ret_sign_case_3)

        # Remove sign bit from exponent
        ret_exp = bit(fp32_config.exponent_bits, ret_exp_signed.bin)

        #print('ret_sign', ret_sign)
        #print('ret_exp ', ret_exp )
        #print('ret_mant', ret_mant)

        return ret_sign, ret_exp, ret_mant
