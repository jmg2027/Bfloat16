from bf16 import bf16, fp32, bit, sbit, ubit, FloatBaseT, fp32_obj, bf16_obj, bit_zero, bit_one, Type, Generic
from hw_model import hwutil

class FloatPowerofTwo:
    
    """
    FloatPowerofTwo
    For given a, return n power of 2
    a: Bfloat16, n: signed integer
    """
    def __init__(self, a: FloatBaseT, n: int) -> None:
        self.a: FloatBaseT = a
        self.n: int = n

    def power(self) -> fp32:
        # If input is Bfloat16, bf16_to_fp32
        # Make flag of bf16 input
        bf16_input: bool = isinstance(self.a, bf16)
        if bf16_input:
            self.a = self.a.bf16_to_fp32()
        # Decompose Bfloat16 to BitString
        a_sign, a_exp, a_mant = self.a.decompose()

        # Exponent to signed bitstring for n to be negative
        # +1 for sign bit
        signed_a_exp = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        signed_bias = sbit(a_exp.bitwidth + 2, bin(self.a.bias))
        signed_n = sbit(a_exp.bitwidth + 2, bin(self.n))

        #Check for special cases
        # input
        # zero
        a_isnormal = False
        if self.a.iszero():
            ret_exp_0 = 0
        # inf
        elif self.a.isinf():
            ret_exp_0: sbit = signed_a_exp
        # nan
        elif self.a.isnan():
            ret_exp_0: sbit = signed_a_exp
        # normal case
        else:
            a_isnormal = True
            ret_exp_0: sbit = signed_a_exp + signed_n

        
        # output
        if a_isnormal:
            # Overflow case: make inf
            if ret_exp_0 > sbit(ret_exp_0.bitwidth + 2 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_1 = sbit(a_exp.bitwidth + 2, bin((1 << self.a.exponent_bits) - 1))
                ret_mant_1 = 0
            # Underflow case: make zero
            elif ret_exp_0 < sbit(ret_exp_0.bitwidth + 2, bin(0)):
                ret_exp_1 = sbit(a_exp.bitwidth + 2, '0')
                ret_mant_1 = 0
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = a_mant
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_1 = a_mant
        ret_sign_1 = a_sign

        # Remove sign bit from exponent
        ret_exp_bit_1 = bit(a_exp.bitwidth, ret_exp_1.bin)

        # Compose BF16
        pow = fp32.compose(ret_sign_1, ret_exp_bit_1, ret_mant_1)
        if bf16_input:
            pow = pow.fp32_to_bf16()
        return pow


class FloatNegative:
    
    """
    FloatNegative
    For given a, return negative a
    a: Bfloat16
    """
    def __init__(self, a: bf16) -> None:
        self.a = a

    def negative(self) -> bf16:
        # If input is Bfloat16, bf16_to_fp32
        # Make flag of bf16 input
        bf16_input = isinstance(self.a, bf16)
        if bf16_input:
            self.a = self.a.bf16_to_fp32()
        # Decompose Bfloat16 to BitString class
        a_sign, a_exp, a_mant = self.a.decompose()

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
        neg = fp32.compose(ret_sign, ret_exp, ret_mant)
        if bf16_input:
            neg = neg.fp32_to_bf16()
        return neg


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
    """
    def __init__(self, a: fp32) -> None:
        self.a = a
        pass

    def fptoint(self) -> int:
        # If input is Bfloat16, bf16_to_fp32
        # Make flag of bf16 input
        bf16_input = isinstance(self.a, bf16)
        if bf16_input:
            self.a = self.a.bf16_to_fp32()

        output_bitwidth = 1 + Float32.exponent_bits + Float32.mantissa_bits
        a_sign, a_exp, a_mant_nohidden = self.a.decompose()
        a_mant_us = ubit(self.a.mantissa_bits + 1, f'{a_exp.reduceor()}{a_mant_nohidden}')
        a_exp_signed = sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        bias_signed = sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        exp_unbiased = int(a_exp_signed - bias_signed)
        mant_unshifted = ubit(output_bitwidth + Float32.mantissa_bits - 1, a_mant_us.bin)

        # without sign bit
        # int(no sign) 15 + mant 7
        mant_shifted = mant_unshifted << exp_unbiased
        int_trunc = mant_shifted[output_bitwidth + Float32.mantissa_bits - 2:Float32.mantissa_bits]
        int_before_sign = sbit(output_bitwidth, f'0{str(int_trunc)}')

        if exp_unbiased > output_bitwidth - 2:
            if a_sign == bit(1, '1'):
                int_result = sbit(output_bitwidth, bin(1 << (Float32.sign_bitpos-1)))
            else:
                int_result = sbit(output_bitwidth, bin((1 << (Float32.sign_bitpos-1)) - 1))
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
    Int -> Float32
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
    def __init__(self, a: int) -> None:
        self.a = a
        pass

    def set_sticky_bits(self, n: int) -> None:
        self.sticky_bits = n
        return

    def inttofp(self):
        output_bitwidth = 1 + Float32.exponent_bits + Float32.mantissa_bits
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
        int_extended = int_unsigned.concat(ubit(Float32.mantissa_bits + round_bits, '0'))
        a_exp_preshift = ubit(Float32.exponent_bits, bin(Float32.bias))
        shift_amount = 0
        int_shift_bitwidth = (output_bitwidth-1) + Float32.mantissa_bits + round_bits

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
        
        a_exp_preround = a_exp_preshift + ubit(Float32.exponent_bits, bin(shift_amount))

        int_mant_part = int_shifted[Float32.mantissa_bits + round_bits:0]
        #print(repr(int_mant_part))

        # round and postnormalize
        if int_mant_part[Float32.mantissa_bits + round_bits:round_bits-1].bin == '1' * (Float32.mantissa_bits + 2):
            a_exp = a_exp_preround + ubit(Float32.exponent_bits, '1')
            int_shifted_rounded = ubit(Float32.mantissa_bits + 1, f'1{"0" * (Float32.mantissa_bits + 1)}')
        else:
            a_exp = a_exp_preround
            int_shifted_rounded = hwutil.round_to_nearest_even_bit(int_mant_part, Float32.mantissa_bits+1)
        
        a_mant = int_shifted_rounded[Float32.mantissa_bits-1:0]

        # check for zero input
        a_is_zero = (a_bitstring.reduceor() == bit(1, '0'))

        if a_is_zero:
            ret_sign = ubit(1, '0')
            ret_exp = ubit(Float32.exponent_bits, '0')
            ret_mant = ubit(Float32.mantissa_bits, '0')
        else:
            ret_sign = a_sign
            ret_exp = a_exp
            ret_mant = a_mant

        #print(ret_sign)
        #print(ret_exp)
        #print(ret_mant)

        inttofp = Float32.compose(ret_sign, ret_exp, ret_mant)
        return inttofp


class FloatBfloat16toFloat32:
    def __init__(self, a: bf16):
        self.a = a
        pass

    def bf16_to_fp32(self):
        # Decompose Bfloat16 to BitString
        a_sign, a_exp, a_mant = self.a.decompose()
        mant_diff_bit = Float32.mantissa_bits - Bfloat16.mantissa_bits

        ret_sign = a_sign
        ret_exp = a_exp
        ret_mant = a_mant.concat(ubit(mant_diff_bit, '0'))

        return Float32.compose(ret_sign, ret_exp, ret_mant)


class FloatFloat32toBfloat16:
    def __init__(self, a: fp32):
        self.a = a
        pass

    def fp32_to_bf16(self):
        # Decompose Float32 to BitString
        a_sign, a_exp, a_mant = self.a.decompose()
        mant_diff_bit = Float32.mantissa_bits - Bfloat16.mantissa_bits

        ret_sign = a_sign
        ret_exp = a_exp
        ret_mant = hwutil.round_to_nearest_even_bit(a_mant, 7)

        return bf16.compose(ret_sign, ret_exp, ret_mant)
