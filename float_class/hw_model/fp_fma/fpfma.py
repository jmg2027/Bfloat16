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
        if ret_mant_rounded_5 == bit(c_mant_us.bitwidth, bin((1 << c_mant_us.bitwidth) - 1)):
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
            shift_amt_124 = 23
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
            shifted_input_124 = sbit(p_mant_us.bitwidth + 1, '0')
            unshifted_input_124 = p_mant_signed_49_pre_124
            temp_exp_124 = p_exp_signed
            ret_sign_case_124 = p_sign
        elif case_2:
            # p > c
            c_mant_signed_26_124 = -c_mant_signed_26_pre_124 if fma_sign.bin == '1' else c_mant_signed_26_pre_124
            c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.concat(sbit(23, "0")).bin}')
            shifted_input_124 = c_mant_signed_49_124.arith_rshift(shift_amt_124)
            unshifted_input_124 = p_mant_signed_49_pre_124
            temp_exp_124 = p_exp_signed
            ret_sign_case_124 = p_sign
        else:
            # c > p
            p_mant_signed_49_124 = -p_mant_signed_49_pre_124 if fma_sign.bin == '1' else p_mant_signed_49_pre_124
            c_mant_signed_26_124 = c_mant_signed_26_pre_124
            c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_26_124.concat(sbit(23, "0")).bin}')
            shifted_input_124 = p_mant_signed_49_124.arith_rshift(shift_amt_124)
            unshifted_input_124 = c_mant_signed_49_124
            temp_exp_124 = c_exp_signed
            ret_sign_case_124 = c_sign
        
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
        #print('mant_add_before_norm_124', repr(mant_add_before_norm_124))
        #print('mant_add_124            ', repr(mant_add_124            ))

        # round
        lsb_124 = mant_add_124[24]
        round_124 = mant_add_124[23]
        sticky_124 = sbit(1, bin(mant_add_124[22:0] != sbit(1, '0')))
        round_up_124 = (lsb_124 & round_124) | (round_124 & sticky_124)
        mant_rounded_124 = mant_add_124[48:24] + round_up_124

        # postnormalize
        if mant_rounded_124 == sbit(mant_rounded_124.bitwidth, bin((1 << mant_rounded_124.bitwidth) - 1)):
            ret_mant_case_124 = bit(precision_bit - 1, '0')
            ret_exp_case_124 = (temp_norm_exp_124 + bit(1, '1'))
            # check overflow
            if int(temp_norm_exp_124) >= 254:
                ret_exp_case_124 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        else:
            ret_mant_case_124 = bit(precision_bit-1, mant_rounded_124[precision_bit-2:0].bin)
            ret_exp_case_124 = temp_norm_exp_124
        

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
        
        
        
        shift_amt_3 = hwutil.leading_zero_count(mant_add_signed_50_3[48:0])

        if int(exp_diff) == 1:
            # p > c
            temp_exp_3 = bit(9, (p_exp_signed + bit(2, bin(2))).bin)
        else:
            # c >= p
            temp_exp_3 = bit(9, (c_exp_signed + bit(2, bin(2))).bin)

        shifted_exp_3 = temp_exp_3 - bit(fp32_config.exponent_bits, bin(shift_amt_3))
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
            if int(shifted_exp_3) >= 254:
                ret_exp_case_3 = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
        else:
            ret_mant_case_3 = mant_rounded_3[precision_bit-2:0]
            ret_exp_case_3 = shifted_exp_3
        ret_sign_case_3 = bit(1, mant_add_29_3[28].bin)

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
