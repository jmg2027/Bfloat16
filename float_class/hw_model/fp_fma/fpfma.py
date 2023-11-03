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

        ret_sign_sp = bit(1, '0')
        ret_exp_sp = bit(1, '0')
        ret_mant_sp = bit(1, '0')

        # Special cases
        #input
        isnormal = False
        # fma(nan,?,?) -> nan
        # nan inputs clearance
        if isnan(self.a) or isnan(self.b) or isnan(self.c):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * 0) + ? -> nan
        elif (isinf(self.a) and iszero(self.b)) or (isinf(self.b) and iszero(self.a)):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * ?) + -inf, (-inf * ?) + inf -> nan
        elif (isinf(self.a) or isinf(self.b)) and isinf(self.c) and (a_sign ^ b_sign ^ c_sign):
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # (inf * ?) + ? -> inf, (-inf * ?) + ? -> -inf
        elif (isinf(self.a) or isinf(self.b)):
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
        p_sign = a_sign ^ b_sign
        fma_sign = c_sign ^ p_sign

        # Exponent calculation
        p_exp_signed = a_exp_signed + b_exp_signed - bias_signed
        exp_diff = p_exp_signed - c_exp_signed

        # Product mantissa
        p_mant_us = a_mant_us * b_mant_us

        # Define 5 cases of FMA
        # Case 1) P_exp – C_exp ≥ 24, 49bit adder, C_man => sticky
        # Case 2) 2 ≤ P_exp – C_exp < 24, 49bit adder
        # Case 3) -2 < P_exp – C_exp < 2, considering cancellation, use 27bit adder
        # Case 4) -24 ≤ P_exp – C_exp ≤ -2, 25bit adder, R sticky
        # Case 5) P_exp – C_exp < -24, R, sticky

        case_1 = int(exp_diff) >=24
        case_2 = 24 > int(exp_diff) >= 2
        case_3 = 2 > int(exp_diff) > -2
        case_4 = -2 >= int(exp_diff) >= -24
        case_5 = -24 > int(exp_diff)

        if case_1:
            shift_amt_124 = 23
        elif case_2:
            shift_amt_124 = int(exp_diff)
        elif case_4:
            shift_amt_124 = -int(exp_diff)
        else:
            # not used
            shift_amt_124 = 0
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
        if c_mant_us == bit(c_mant_us.bitwidth, bin((1 << c_mant_us.bitwidth) - 1)):
            ret_mant_case_5 = bit(precision_bit - 1, '0')
            ret_exp_case_5 = (c_exp_signed + sbit(2, '01'))[fp32_config.exponent_bits-1:0]
        else:
            ret_mant_case_5 = ret_mant_rounded_5[precision_bit-2:0]
            ret_exp_case_5 = c_exp_signed[fp32_config.exponent_bits-1:0]
        ret_sign_case_5 = c_sign

        # result: ret_mant_case_5, ret_exp_case_5, ret_sign_case_5

        # Case 1, 2, 4
        p_mant_signed_49_pre_124 = sbit(p_mant_us.bitwidth + 1, f'0{p_mant_us.bin}')
        c_mant_signed_25_pre_124 = sbit(c_mant_us.bitwidth + 1, f'0{c_mant_us.bin}')
        p_mant_signed_49_124 = -p_mant_signed_49_pre_124 if fma_sign else p_mant_signed_49_pre_124
        c_mant_signed_25_124 = -c_mant_signed_25_pre_124 if fma_sign else c_mant_signed_25_pre_124
        # Sign extension
        c_mant_signed_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_signed_25_124.bin}')
        c_mant_49_124 = sbit(p_mant_us.bitwidth + 1, f'{c_mant_us.concat(bit(24, "0")).bin}')
        

        # Shift lesser mantissa
        # case 1, 2: p > c
        # case 4: c > p
        # 1x.xxx -> exp + 1, 0x.xxx -> exp
        if case_1 or case_2:
            shifted_input_124 = c_mant_signed_49_124 >> shift_amt_124
            unshifted_input_124 = p_mant_signed_49_pre_124
            temp_exp_124 = p_exp_signed + sbit(2, '01') if p_mant_signed_49_pre_124[47] else p_exp_signed
        else:
            shifted_input_124 = p_mant_signed_49_124 >> shift_amt_124
            unshifted_input_124 = c_mant_49_124
            temp_exp_124 = c_exp_signed
        
        mant_add_124 = shifted_input_124 + unshifted_input_124

        # round
        lsb_124 = mant_add_124[24]
        round_124 = mant_add_124[23]
        sticky_124 = sbit(1, bin(mant_add_124[22:0] != sbit(1, '0')))
        round_up_124 = (lsb_124 & round_124) | (round_124 & sticky_124)
        mant_rounded_124 = mant_add_124[48:24] + round_up_124

        # postnormalize
        if mant_rounded_124 == sbit(mant_rounded_124.bitwidth, bin((1 << mant_rounded_124.bitwidth) - 1)):
            ret_mant_case_124 = bit(precision_bit - 1, '0')
            ret_exp_case_124 = (temp_exp_124 + sbit(2, '01'))[fp32_config.exponent_bits-1:0]
        else:
            ret_mant_case_124 = mant_rounded_124[precision_bit-2:0]
            ret_exp_case_124 = temp_exp_124[fp32_config.exponent_bits-1:0]
        ret_sign_case_124 = fma_sign

        # Case 3
        c_mant_inv = -sbit(25, f'0{c_mant_us.bin}') if c_sign.bin == '1' else sbit(25, f'0{c_mant_us.bin}')
        p_mant_inv = -sbit(49, f'0{p_mant_us.bin}') if p_sign.bin == '1' else sbit(49, f'0{p_mant_us.bin}')
        c_mant_signed_27_3 = sbit(27, f'{c_mant_inv.bin}00')
        p_mant_signed_49_3 = p_mant_inv
        c_mant_signed_shifted_27_3 = c_mant_signed_27_3 >> 1 if int(exp_diff) == 1 else c_mant_signed_27_3
        #p_mant_signed_shifted_49_3 = p_mant_signed_49_3 >> 1 if int(exp_diff) == -1 else p_mant_signed_49_3
        p_mant_signed_shifted_49_3 = p_mant_signed_49_3 if int(exp_diff) == 1 else p_mant_signed_49_3 >> 1

        mant_add_49_3 = p_mant_signed_shifted_49_3[48:22] + c_mant_signed_shifted_27_3
        mant_add_signed_27_3 = -mant_add_49_3 if mant_add_49_3 == sbit(2, '01') else mant_add_49_3
        mant_add_signed_49_3 = mant_add_signed_27_3.concat(p_mant_signed_shifted_49_3[21:0])

        #shift_amt_3 = hwutil.leading_zero_count(mant_add_signed_49_3)
        shift_amt_3 = hwutil.leading_zero_count(mant_add_signed_49_3[47:0])
        
        print('p_mant_us', repr(p_mant_us))
        print('p_mant_inv', repr(p_mant_inv))

        if int(exp_diff) == 1:
            # p > c
            if p_mant_us[47] == bit(1, '1'):
                temp_exp_3 = p_exp_signed + sbit(2, '01')
            else:
                temp_exp_3 = p_exp_signed
        else:
            # c >= p
            temp_exp_3 = c_exp_signed

        shifted_exp_3 = temp_exp_3 - sbit(fp32_config.exponent_bits + 2, bin(shift_amt_3))
        mant_add_shifted_49_3 = ubit(mant_add_signed_49_3.bitwidth, (mant_add_signed_49_3 << shift_amt_3).bin)

        # round
        lsb_3 = mant_add_shifted_49_3[24]
        round_3 = mant_add_shifted_49_3[23]
        sticky_3 = ubit(1, bin(mant_add_shifted_49_3[22:0] != ubit(1, '0')))
        round_up_3 = (lsb_3 & round_3) | (round_3 & sticky_3)
        mant_rounded_3 = mant_add_shifted_49_3[48:24] + round_up_3

        # postnormalize
        if mant_rounded_3 == ubit(mant_rounded_3.bitwidth, bin((1 << mant_rounded_3.bitwidth) - 1)):
            ret_mant_case_3 = bit(precision_bit - 1, '0')
            ret_exp_case_3 = (shifted_exp_3 + sbit(2, '01'))[fp32_config.exponent_bits-1:0]
        else:
            #ret_mant_case_3 = mant_rounded_3[precision_bit-2:0]
            ret_mant_case_3 = mant_rounded_3[precision_bit-2:0]
            ret_exp_case_3 = shifted_exp_3[fp32_config.exponent_bits-1:0]
        ret_sign_case_3 = fma_sign

        print('shift_amt_3', shift_amt_3)
        print('c_mant_signed_shifted_27_3', repr(c_mant_signed_shifted_27_3))
        print('p_mant_signed_shifted_49_3', repr(p_mant_signed_shifted_49_3))
        print('mant_add_signed_49_3', mant_add_signed_49_3)
        print('mant_add_shifted_49_3', mant_add_shifted_49_3)
        print('mant_add_shifted_49_3[48:24]', mant_add_shifted_49_3[48:24])
        print('round_up_3', round_up_3)
        print('mant_rounded_3', repr(mant_rounded_3))

        # final output mux
        if case_1 or case_2 or case_4:
            ret_sign = ret_sign_case_124
            ret_exp_signed = ret_exp_case_124
            ret_mant = ret_mant_case_124
        elif case_3:
            ret_sign = ret_sign_case_3
            ret_exp_signed = ret_exp_case_3
            ret_mant = ret_mant_case_3
        else:
            ret_sign = ret_sign_case_5
            ret_exp_signed = ret_exp_case_5
            ret_mant = ret_mant_case_5

        if isnormal:
            ret_sign = ret_sign
            ret_exp_signed = ret_exp_signed
            ret_mant = ret_mant
        else:
            ret_sign = ret_sign_sp
            ret_exp_signed = ret_exp_sp
            ret_mant = ret_mant_sp

        print('exp_diff', int(exp_diff))

        print('case_1', case_1)
        print('case_2', case_2)
        print('case_3', case_3)
        print('case_4', case_4)
        print('case_5', case_5)

        print('ret_exp_case_5', int(ret_exp_case_5))
        print('ret_exp_case_124', int(ret_exp_case_124))
        print('ret_exp_case_3', int(ret_exp_case_3))

        print('ret_mant_case_5', repr(ret_mant_case_5))
        print('ret_mant_case_124', repr(ret_mant_case_124))
        print('ret_mant_case_3', repr(ret_mant_case_3))

        print('ret_sign_case_5', ret_sign_case_5)
        print('ret_sign_case_124', ret_sign_case_124)
        print('ret_sign_case_3', ret_sign_case_3)

        # Remove sign bit from exponent
        ret_exp = bit(fp32_config.exponent_bits, ret_exp_signed.bin)

        return ret_sign, ret_exp, ret_mant
