from ..utils.commonimport import *

from typing import Union, Generic, Literal, TypeVar
from math import log2

#FloatBaseT = TypeVar('FloatBaseT', bound='FloatBase')

class FloatPowerofTwo:
    
    """
    FloatPowerofTwo
    For given a, return n power of 2
    a: Bfloat16, n: signed integer
    """
    def __init__(self, a: FPBitT, n: int) -> None:
        self.a = a
        self.n: int = n

    def excute(self) -> FPBitT:
        # Decompose Bfloat16 to BitString
        a_sign, a_exp, a_mant = self.a

        # Exponent to signed bitstring for n to be negative
        # +1 for sign bit
        signed_a_exp = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        signed_n = sbit(a_exp.bitwidth + 2, bin(self.n))

        #Check for special cases
        # input
        # zero
        a_isnormal = False
        if iszero(self.a):
            ret_exp_0: sbit = sbit(a_exp.bitwidth + 2, '0')
        # inf
        elif isinf(self.a):
            ret_exp_0: sbit = signed_a_exp
        # nan
        elif isnan(self.a):
            ret_exp_0: sbit = signed_a_exp
        # normal case
        else:
            a_isnormal = True
            ret_exp_0: sbit = signed_a_exp + signed_n
        
        # output
        if a_isnormal:
            # Overflow case: make inf
            if ret_exp_0 > sbit(ret_exp_0.bitwidth + 2 , bin((1 << fp32_config.exponent_bits) - 1)):
                ret_exp_1 = sbit(a_exp.bitwidth + 2, bin((1 << fp32_config.exponent_bits) - 1))
                ret_mant_1 = bit(a_mant.bitwidth, '0')
            # Underflow case: make zero
            elif ret_exp_0 < sbit(ret_exp_0.bitwidth + 2, bin(0)):
                ret_exp_1 = sbit(a_exp.bitwidth + 2, '0')
                ret_mant_1 = bit(a_mant.bitwidth, '0')
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = a_mant
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_1 = a_mant
        ret_sign_1 = a_sign

        # Remove sign bit from exponent
        ret_exp_bit_1 = bit(a_exp.bitwidth, ret_exp_1.bin)

        return ret_sign_1, ret_exp_bit_1, ret_mant_1


class FloatNegative:
    
    """
    FloatNegative
    For given a, return negative a
    a: Bfloat16
    """
    def __init__(self, a: FPBitT) -> None:
        self.a = a

    def excute(self) -> FPBitT:
        # Decompose Bfloat16 to BitString class
        a_sign, a_exp, a_mant = self.a

        # neg -> pos
        ret_exp = a_exp
        ret_mant = a_mant
        if a_sign == bit(1, '1'):
            ret_sign = bit(1, '0')
        elif a_sign == bit(1, '0'):
            ret_sign = bit(1, '1')
        else:
            raise ValueError('Sign bit must be 0 or 1')

        # Compose BF16
        return ret_sign, ret_exp, ret_mant


class FloatFPtoInt:
    """
    FloatFPtoInt
    Bfloat16 -> Signed Integer (16bits output)
    -32768~32767(0x8000 ~ 0x7FFF)
    - Algorithm
    exponent > 14: overflow -> positive: 0x7FFF, negative: 0x8000
    else: shift mantissa as exponent
    15 bits for integer without sign
    7 bits for mantissa under point
    1xx....xx   .   xxxxxxx
    < 15bits><point>< mant>
    // mod              rnd_mod
    // 0: fp32          0: round to nearest even
    // 1: bf16          1: floor => truncation
    //                  2: ceil  => always rounding, round to infinity
    """
    def __init__(self, a: FPBitT, mod: int = 0, rnd_mod: int = 0) -> None:
        self.a = a
        self.mod = mod
        self.rnd_mod = rnd_mod
        pass

    def excute(self) -> int:
        output_bitwidth = 1 + fp32_config.exponent_bits + fp32_config.mantissa_bits
        a_sign, a_exp, a_mant_nohidden = self.a 
        sig = ubit(output_bitwidth + fp32_config.mantissa_bits, f'1{a_mant_nohidden}')
        sh_sig = ubit(output_bitwidth + fp32_config.mantissa_bits, '0')
        ans = ubit(output_bitwidth, '0')
        shamt = ubit(int(log2(output_bitwidth)), f'{bin(int(a_exp) - fp32_config.bias)}')
        ulp = ubit(1, '0')
        rnd = ubit(1, '0')
        stk = ubit(1, '0')
        rndup = ubit(1, '0')

        if (int(a_exp) >= fp32_config.bias + output_bitwidth - 1): # 127 + 31 -> max or overflow -> saturation
            if (a_sign == ubit(1, '1')): # negative
                ans.set_bin(bin(1 << 32))
            else:
                ans.set_bin(bin((1 << 32) - 1))
        elif (int(a_exp) >= fp32_config.bias): # normal conversion, shift and round
            sh_sig = sig << int(shamt)
            ulp.set_bin(sh_sig[fp32_config.mantissa_bits].bin)
            rnd.set_bin(sh_sig[fp32_config.mantissa_bits-1].bin)
            stk.set_bin(bin(int(sh_sig[fp32_config.mantissa_bits-2:0] != ubit(1, '0'))))
            rndup.set_bin(((ulp & rnd) | (rnd & stk)).bin)
            if (self.rnd_mod == 0):
                sh_sig = sh_sig + rndup
            elif (self.rnd_mod == 1):
                sh_sig = sh_sig
            else:
                sh_sig = sh_sig + ubit(1, '1')
            ans.set_bin(sh_sig[output_bitwidth+fp32_config.mantissa_bits:fp32_config.mantissa_bits].bin)
        else:
            ans.set_bin('0')

        
        int_result = ans
        if (a_sign == ubit(1, '1')): # negative
            return -int(int_result)
        else:
            return int(int_result)

    def fptoint2(self):
        # If input is Bfloat16, bf16_to_fp32
        # Make flag of bf16 input
        bf16_input = isinstance(self.a, bf16)
        if bf16_input:
            a_fp32 = self.a.bf16_to_fp32()
        elif isinstance(self.a, fp32):
            a_fp32= self.a
        else:
            raise FloatTypeError('not allowed in fptoint')

        output_bitwidth = 1 + fp32_config.exponent_bits + fp32_config.mantissa_bits
        a_sign, a_exp, a_mant_nohidden = a_fp32.decompose()
        a_mant_us = bit(fp32_config.mantissa_bits + 1, f'{a_exp.reduceor()}{a_mant_nohidden}')
        a_exp_signed = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        bias_signed = sbit(a_exp.bitwidth + 2, bin(a_fp32.bias))

        exp_unbiased = int(a_exp_signed - bias_signed)
        mant_unshifted = ubit(output_bitwidth + fp32_config.mantissa_bits - 1, a_mant_us.bin)

        # without sign bit
        # int(no sign) 15 + mant 7
        mant_shifted = mant_unshifted << exp_unbiased
        int_trunc = mant_shifted[output_bitwidth + fp32_config.mantissa_bits - 2:fp32_config.mantissa_bits]
        int_before_sign = sbit(output_bitwidth, f'0{str(int_trunc)}')

        if exp_unbiased > output_bitwidth - 2:
            if a_sign == bit(1, '1'):
                int_result = sbit(output_bitwidth, bin(1 << (fp32_config.sign_bitpos-1)))
            else:
                int_result = sbit(output_bitwidth, bin((1 << (fp32_config.sign_bitpos-1)) - 1))
        elif exp_unbiased < 0:
            int_result = sbit(output_bitwidth, '0')
        else:
            twos_comp = -int_before_sign
            if a_sign == bit(1, '1'):
                int_result = twos_comp
            else:
                int_result = int_before_sign
        return int(int_result)


class FloatInttoFP:
    """
    FloatInttoFP
    Mod == 0: Int -> Bfloat16 Mod == 1: Int -> Float32 
    - Algorithm
    save sign and make two's complement
    31 bits for integer without sign
    23 bits for mantissa under point
    23 bits for round/sticky
    1xx....xx   .   xxxx...xxx rsss...sss
    < 31bits><point>< mant> <  r/s >
    find leading one and shift mantissa
    remains are mantissa
    shift amount is exponent
    add ceiling and flooring
    """
    sticky_bits = 22
    def __init__(self, a: int, mod: Union[Literal[0], Literal[1]] = 0) -> None:
        self.a = a
        self.mod = mod
        pass

    def set_sticky_bits(self, n: int) -> None:
        self.sticky_bits = n
        return
    
    #def inttofp(self) -> Union[bf16, fp32]:
    def excute(self) -> FPBitT:
        #fp_out = 0
        if (self.a == 0):
            if (self.mod == 1):
                sign, final_exp, frac = bit(1, '0'), bit(8, '0'), bit(23, '0')
            else:
                sign, final_exp, frac = bit(1, '0'), bit(8, '0'), bit(7, '0')
        elif (self.a == 0x8000_0000):
            # 0xCF00_0000
            if (self.mod == 1):
                sign, final_exp, frac = bit(1, '1'), bit(8, '10011110'), bit(23, '0')
            # 0xCF00
            else:
                sign, final_exp, frac = bit(1, '1'), bit(8, '10011110'), bit(7, '0')
        else:
            # check for negative integer
            a: ubit = ubit.from_int(32, self.a)

            sign= ubit(1, '0')
            lsb= ubit(1, '0')
            rnd= ubit(1, '0')
            sticky= ubit(1, '0')
            roundup= ubit(1, '0')
            
            mag= ubit(32, '0')
            significant= ubit(32, '0')

            exp= ubit(8, '0')
            new_exp= ubit(8, '0')
            final_exp= ubit(8, '0')

            num_zeros= int

            sig25= ubit(25, '0')
            

            sign = a[31]
            mag = a if (sign == ubit(1, '0')) else -a
            # is this 158?
            #exp.set_bin(bin(157)) # to compensate binary point of integer 31bit data
                      # 127 + 30
            exp.set_bin(bin(158)) # to compensate binary point of integer 31bit data
                      # 127 + 31
            num_zeros = hwutil.leading_zero_count(mag)
            new_exp = exp - ubit(8, bin(num_zeros))
            significant.set_bin((mag << num_zeros).bin)
            if (self.mod == 1):
                frac = ubit(23, '0')
                # how does this work if bitwidth is different?
                lsb = significant[7]
                #lsb.set_bin(significant[7].bin)
                rnd.set_bin(significant[6].bin)
                sticky.set_bin(bin(significant[5:0] != ubit(1, '0')))

                roundup.set_bin(((rnd & lsb) | (rnd & sticky)).bin)

                sig25.set_bin((significant[30:7] + roundup).bin)

                frac.set_bin((sig25[23:1] if (sig25[24] == ubit(1, '1')) else sig25[22:0]).bin)
                final_exp.set_bin((new_exp + ubit(1, '1') if (sig25[24] == ubit(1, '1')) else new_exp).bin)

                return sign, final_exp, frac
            else:
                frac = ubit(7, '0')
                lsb.set_bin(significant[23].bin)
                rnd.set_bin(significant[22].bin)
                sticky.set_bin(bin(significant[21:0] != ubit(1, '0')))
                roundup.set_bin(((rnd & lsb) | (rnd & sticky)).bin)

                sig25.set_bin((significant[30:23] + roundup).bin)
                frac.set_bin((sig25[7:1] if (sig25[8] == ubit(1, '1')) else sig25[6:0]).bin)
                final_exp.set_bin((new_exp + ubit(1, '1') if (sig25[8] == ubit(1, '1')) else new_exp).bin)

                return sign, final_exp, frac
        return sign, final_exp, frac

    def inttofp2(self):
        output_bitwidth = 1 + fp32_config.exponent_bits + fp32_config.mantissa_bits
        round_bits = self.sticky_bits + 1
        a = self.a
        a_bitstring = sbit(output_bitwidth, bin(a))
        a_sign = ubit(1, a_bitstring[output_bitwidth-1].bin)

        if a_sign == ubit(1, '1'):
            a_unsigned = -a_bitstring
        else:
            a_unsigned = a_bitstring

        # remove sign bit
        int_unsigned = ubit(output_bitwidth-1, a_unsigned[output_bitwidth-2:0].bin)
        # extend shift register
        int_extended = int_unsigned.concat(ubit(fp32_config.mantissa_bits + round_bits, '0'))
        a_exp_preshift = ubit(fp32_config.exponent_bits, bin(fp32_config.bias))
        shift_amount = 0
        int_shift_bitwidth = (output_bitwidth-1) + fp32_config.mantissa_bits + round_bits

        # find leading one in [int_shift_bitwidth - 1:int_shift_bitwidth - (output_bitwidth-2)] 
        # [28:15] -> 14
        # [76:47] -> 30
        #for i in range(1, 31):
        #    if int_extended[int_shift_bitwidth - i] == bit(1, '1'):
        #        shift_amount = output_bitwidth - (i + 1)
        #        break
        #    else:
        #        shift_amount = 0
        if int_extended[int_shift_bitwidth-1] == bit(1, '1'):
            shift_amount = output_bitwidth - 2
        elif int_extended[int_shift_bitwidth-2] == bit(1, '1'):
            shift_amount = output_bitwidth - 3
        elif int_extended[int_shift_bitwidth-3] == bit(1, '1'):
            shift_amount = output_bitwidth - 4
        elif int_extended[int_shift_bitwidth-4] == bit(1, '1'):
            shift_amount = output_bitwidth - 5
        elif int_extended[int_shift_bitwidth-5] == bit(1, '1'):
            shift_amount = output_bitwidth - 6
        elif int_extended[int_shift_bitwidth-6] == bit(1, '1'):
            shift_amount = output_bitwidth - 7
        elif int_extended[int_shift_bitwidth-7] == bit(1, '1'):
            shift_amount = output_bitwidth - 8
        elif int_extended[int_shift_bitwidth-8] == bit(1, '1'):
            shift_amount = output_bitwidth - 9
        elif int_extended[int_shift_bitwidth-9] == bit(1, '1'):
            shift_amount = output_bitwidth - 10
        elif int_extended[int_shift_bitwidth-10] == bit(1, '1'):
            shift_amount = output_bitwidth - 11
        elif int_extended[int_shift_bitwidth-11] == bit(1, '1'):
            shift_amount = output_bitwidth - 12
        elif int_extended[int_shift_bitwidth-12] == bit(1, '1'):
            shift_amount = output_bitwidth - 13
        elif int_extended[int_shift_bitwidth-13] == bit(1, '1'):
            shift_amount = output_bitwidth - 14
        elif int_extended[int_shift_bitwidth-14] == bit(1, '1'):
            shift_amount = output_bitwidth - 15
        elif int_extended[int_shift_bitwidth-15] == bit(1, '1'):
            shift_amount = output_bitwidth - 16
        elif int_extended[int_shift_bitwidth-16] == bit(1, '1'):
            shift_amount = output_bitwidth - 17
        elif int_extended[int_shift_bitwidth-17] == bit(1, '1'):
            shift_amount = output_bitwidth - 18
        elif int_extended[int_shift_bitwidth-18] == bit(1, '1'):
            shift_amount = output_bitwidth - 19
        elif int_extended[int_shift_bitwidth-19] == bit(1, '1'):
            shift_amount = output_bitwidth - 20
        elif int_extended[int_shift_bitwidth-20] == bit(1, '1'):
            shift_amount = output_bitwidth - 21
        elif int_extended[int_shift_bitwidth-21] == bit(1, '1'):
            shift_amount = output_bitwidth - 22
        elif int_extended[int_shift_bitwidth-22] == bit(1, '1'):
            shift_amount = output_bitwidth - 23
        elif int_extended[int_shift_bitwidth-23] == bit(1, '1'):
            shift_amount = output_bitwidth - 24
        elif int_extended[int_shift_bitwidth-24] == bit(1, '1'):
            shift_amount = output_bitwidth - 25
        elif int_extended[int_shift_bitwidth-25] == bit(1, '1'):
            shift_amount = output_bitwidth - 26
        elif int_extended[int_shift_bitwidth-26] == bit(1, '1'):
            shift_amount = output_bitwidth - 27
        elif int_extended[int_shift_bitwidth-27] == bit(1, '1'):
            shift_amount = output_bitwidth - 28
        elif int_extended[int_shift_bitwidth-28] == bit(1, '1'):
            shift_amount = output_bitwidth - 29
        elif int_extended[int_shift_bitwidth-29] == bit(1, '1'):
            shift_amount = output_bitwidth - 30
        elif int_extended[int_shift_bitwidth-30] == bit(1, '1'):
            shift_amount = output_bitwidth - 31
        else:
            shift_amount = 0


        int_shifted = int_extended >> shift_amount
        
        a_exp_preround = a_exp_preshift + ubit(fp32_config.exponent_bits, bin(shift_amount))

        int_mant_part = int_shifted[fp32_config.mantissa_bits + round_bits:0]
        #print(repr(int_mant_part))

        # round and postnormalize
        if int_mant_part[fp32_config.mantissa_bits + round_bits:round_bits-1].bin == '1' * (fp32_config.mantissa_bits + 2):
            a_exp = a_exp_preround + ubit(fp32_config.exponent_bits, '1')
            int_shifted_rounded = ubit(fp32_config.mantissa_bits + 1, f'1{"0" * (fp32_config.mantissa_bits + 1)}')
        else:
            a_exp = a_exp_preround
            int_shifted_rounded = hwutil.round_to_nearest_even_bit(int_mant_part, Float32.mantissa_bits+1)
        
        a_mant = int_shifted_rounded[fp32_config.mantissa_bits-1:0]

        # check for zero input
        a_is_zero = (a_bitstring.reduceor() == bit(1, '0'))

        if a_is_zero:
            ret_sign = ubit(1, '0')
            ret_exp = ubit(fp32_config.exponent_bits, '0')
            ret_mant = ubit(fp32_config.mantissa_bits, '0')
        else:
            ret_sign = a_sign
            ret_exp = a_exp
            ret_mant = a_mant

        #print(ret_sign)
        #print(ret_exp)
        #print(ret_mant)

        inttofp = fp32.compose(ret_sign, ret_exp, ret_mant)
        return inttofp


class FloatBfloat16toFloat32:
    def __init__(self, a: FPBitT) -> None:
        self.a = a
        pass

    def excute(self) -> FPBitT:
        # Decompose Bfloat16 to BitString
        mant_diff_bit = fp32_config.mantissa_bits - bf16_config.mantissa_bits # 16
        a_sign, a_exp, a_mant = self.a

        ret_sign = a_sign
        ret_exp = a_exp
        ret_mant = a_mant.concat(ubit(mant_diff_bit, '0'))

        return ret_sign, ret_exp, ret_mant


class FloatFloat32toBfloat16:
    def __init__(self, a: FPBitT) -> None:
        self.a = a
        pass

    def excute(self) -> FPBitT:
        # Decompose Float32 to BitString
        mant_diff_bit = fp32_config.mantissa_bits - bf16_config.mantissa_bits
        a_sign, a_exp, a_mant = self.a

        #rnd = a_mant[15]
        #stk = bit(1, f'{int(a_mant[14:0] != bit(15, bin(0)))}')
        #rndup = bit(1, f'{(a_mant[16] & rnd) | (rnd & stk)}')

        #ret_sign = a_sign
        #ret_exp = a_exp

        #if ((a_mant[22:16] == bit(7, bin(0x7F)))):
        #    ret_mant = bit(1, '0')
        #    if (int(a_exp) != 255):
        #        ret_exp = a_exp + bit(1, '1')
        #else:
        #    ret_mant = (a_mant + rndup.concat(bit(mant_diff_bit, '0')))[22:16]


        
        ret_sign = a_sign
        ret_exp = a_exp
        ret_mant = hwutil.round_to_nearest_even_bit(a_mant, 7)
        

        return ret_sign, ret_exp, ret_mant
