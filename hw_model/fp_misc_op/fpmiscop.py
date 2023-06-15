import bf16.bf16 as bf16

class FloatPowerofTwo:
    
    """
    FloatPowerofTwo
    For given a, return n power of 2
    a: Bfloat16, n: signed integer
    """
    def __init__(self, a:'bf16.Bfloat16', n: int) -> None:
        self.a = a
        self.n = n

    def power(self) -> 'bf16.Bfloat16':
        # Decompose Bfloat16 to BitString
        a_sign, a_exp, a_mant = self.a.decompose_bf16()

        # Exponent to signed bitstring for n to be negative
        # +1 for sign bit
        signed_a_exp = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        signed_bias = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))
        signed_n = bf16.sbit(a_exp.bitwidth + 2, bin(self.n))

        #Check for special cases
        # input
        # zero
        a_isnormal = False
        if self.a.iszero():
            ret_exp_0 = 0
        # inf
        elif self.a.isinf():
            ret_exp_0 = a_exp
        # nan
        elif self.a.isnan():
            ret_exp_0 = a_exp
        # normal case
        else:
            a_isnormal = True
            ret_exp_0 = signed_a_exp + signed_n

        
        # output
        if a_isnormal:
            # Overflow case: make inf
            if ret_exp_0 > bf16.sbit(ret_exp_0.bitwidth + 2 , bin((1 << self.a.exponent_bits) - 1)):
                ret_exp_1 = bf16.sbit(a_exp.bitwidth + 2, bin((1 << self.a.exponent_bits) - 1))
                ret_mant_1 = 0
            # Underflow case: make zero
            elif ret_exp_0 < bf16.sbit(ret_exp_0.bitwidth + 2, bin(0)):
                ret_exp_1 = bf16.sbit(a_exp.bitwidth + 2, '0')
                ret_mant_1 = 0
            else:
                ret_exp_1 = ret_exp_0
                ret_mant_1 = a_mant
        else:
            ret_exp_1 = ret_exp_0
            ret_mant_1 = a_mant
        ret_sign_1 = a_sign

        # Remove sign bit from exponent
        ret_exp_bit_1 = bf16.bit(a_exp.bitwidth, ret_exp_1.bin)

        # Compose BF16
        pow = bf16.Bfloat16.compose_bf16(ret_sign_1, ret_exp_bit_1, ret_mant_1)
        return pow


class FloatNegative:
    
    """
    FloatNegative
    For given a, return negative a
    a: Bfloat16
    """
    def __init__(self, a:'bf16.Bfloat16') -> None:
        self.a = a

    def negative(self) -> 'bf16.Bfloat16':
        # Decompose Bfloat16 to BitString class
        a_sign, a_exp, a_mant = self.a.decompose_bf16()

        # neg -> pos
        ret_exp = a_exp
        ret_mant = a_mant
        if a_sign == bf16.bit(1, '1'):
            ret_sign = '0'
        elif a_sign == bf16.bit(1, '0'):
            ret_sign = '1'
        else:
            raise ValueError('Sign bit must be 0 or 1')

        # Compose BF16
        neg = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
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
    def __init__(self, a: 'bf16.Bfloat16') -> None:
        self.a = a
        pass

    def fptoint(self) -> int:
        output_bitwidth = 1 + bf16.Bfloat16.exponent_bits + bf16.Bfloat16.mantissa_bits
        a_sign, a_exp, a_mant_nohidden = self.a.decompose_bf16()
        a_mant_us = bf16.ubit(self.a.mantissa_bits + 1, f'{a_exp.reduceor()}{a_mant_nohidden}')
        a_exp_signed = bf16.sbit(a_exp.bitwidth + 2, f'0{a_exp.bin}')
        bias_signed = bf16.sbit(a_exp.bitwidth + 2, bin(self.a.bias))

        exp_unbiased = int(a_exp_signed - bias_signed)
        mant_unshifted = bf16.ubit(output_bitwidth + bf16.Bfloat16.mantissa_bits - 1, a_mant_us.bin)

        # without sign bit
        # int(no sign) 15 + mant 7
        mant_shifted = mant_unshifted << exp_unbiased
        int_trunc = mant_shifted[output_bitwidth + bf16.Bfloat16.mantissa_bits - 2:bf16.Bfloat16.mantissa_bits]
        int_before_sign = bf16.sbit(output_bitwidth, f'0{str(int_trunc)}')

        if exp_unbiased > output_bitwidth - 2:
            if a_sign == bf16.bit(1, '1'):
                int_result = bf16.sbit(output_bitwidth, bin(0x8000))
            else:
                int_result = bf16.sbit(output_bitwidth, bin(0x7FFF))
        elif exp_unbiased < 0:
            int_result = bf16.sbit(output_bitwidth, '0')
        else:
            twos_comp = -int_before_sign
            if a_sign == bf16.bit(1, '1'):
                int_result = twos_comp
            else:
                int_result = int_before_sign
        return int(int_result)


class FloatInttoFP:
    """
    FloatInttoFP
    Int -> Bfloat16
    - Algorithm
    save sign and make two's complement
    15 bits for integer without sign
    7 bits for mantissa under point
    7 bits for round/sticky
    1xx....xx   .   xxxxxxx rssssss
    < 15bits><point>< mant> <  r/s >
    find leading one and shift mantissa
    remains are mantissa
    shift amount is exponent
    """
    sticky_bits = 6
    def __init__(self, a: int) -> None:
        self.a = a
        pass

    def set_sticky_bits(self, n: int) -> None:
        self.sticky_bits = n
        return

    def inttofp(self) -> 'bf16.Bfloat16':
        output_bitwidth = 1 + bf16.Bfloat16.exponent_bits + bf16.Bfloat16.mantissa_bits
        round_bits = self.sticky_bits + 1
        a = self.a
        a_bitstring = bf16.sbit(output_bitwidth, bin(a))
        a_sign = bf16.ubit(1, a_bitstring[output_bitwidth-1].bin)

        if a_sign == bf16.ubit(1, '1'):
            a_unsigned = -a_bitstring
        else:
            a_unsigned = a_bitstring

        # remove sign bit
        int_unsigned = bf16.ubit(output_bitwidth-1, a_unsigned[output_bitwidth-2:0].bin)
        # extend shift register
        int_extended = int_unsigned.concat(bf16.ubit(bf16.Bfloat16.mantissa_bits + round_bits, '0'))
        a_exp_preshift = bf16.ubit(bf16.Bfloat16.exponent_bits, bin(bf16.Bfloat16.bias))
        shift_amount = 0
        int_shift_bitwidth = (output_bitwidth-1) + bf16.Bfloat16.mantissa_bits + round_bits

        # find leading one in [int_shift_bitwidth - 1:int_shift_bitwidth - (output_bitwidth-2)] 
        # [28:15]
        if int_extended[int_shift_bitwidth-1] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 2
        elif int_extended[int_shift_bitwidth-2] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 3
        elif int_extended[int_shift_bitwidth-3] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 4
        elif int_extended[int_shift_bitwidth-4] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 5
        elif int_extended[int_shift_bitwidth-5] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 6
        elif int_extended[int_shift_bitwidth-6] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 7
        elif int_extended[int_shift_bitwidth-7] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 8
        elif int_extended[int_shift_bitwidth-8] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 9
        elif int_extended[int_shift_bitwidth-9] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 10
        elif int_extended[int_shift_bitwidth-10] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 11
        elif int_extended[int_shift_bitwidth-11] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 12
        elif int_extended[int_shift_bitwidth-12] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 13
        elif int_extended[int_shift_bitwidth-13] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 14
        elif int_extended[int_shift_bitwidth-14] == bf16.bit(1, '1'):
            shift_amount = output_bitwidth - 15
        else:
            shift_amount = 0

        int_shifted = int_extended >> shift_amount

#        while(1):
#            zero_detect = int_shifted[int_shift_bitwidth-1:bf16.Bfloat16.mantissa_bits + 1 + round_bits].reduceor()
#            if zero_detect.bin == '0':
#                break
#            else:
#                int_shifted = int_shifted >> 1
#                shift_amount = shift_amount + 1
        
        a_exp_preround = a_exp_preshift + bf16.ubit(bf16.Bfloat16.exponent_bits, bin(shift_amount))

        int_mant_part = int_shifted[bf16.Bfloat16.mantissa_bits + round_bits:0]
        #print(repr(int_mant_part))

        # round and postnormalize
        if int_mant_part[bf16.Bfloat16.mantissa_bits + round_bits:round_bits-1].bin == '111111111':
            a_exp = a_exp_preround + bf16.ubit(bf16.Bfloat16.exponent_bits, '1')
            int_shifted_rounded = bf16.ubit(bf16.Bfloat16.mantissa_bits+1, '10000000')
        else:
            a_exp = a_exp_preround
            int_shifted_rounded = bf16.hwutil.round_to_nearest_even_bit(int_mant_part, bf16.Bfloat16.mantissa_bits+1)
        
        a_mant = int_shifted_rounded[bf16.Bfloat16.mantissa_bits-1:0]

        # check for zero input
        a_is_zero = (a_bitstring.reduceor() == bf16.bit(1, '0'))

        if a_is_zero:
            ret_sign = bf16.ubit(1, '0')
            ret_exp = bf16.ubit(bf16.Bfloat16.exponent_bits, '0')
            ret_mant = bf16.ubit(bf16.Bfloat16.mantissa_bits, '0')
        else:
            ret_sign = a_sign
            ret_exp = a_exp
            ret_mant = a_mant

        #print(ret_sign)
        #print(ret_exp)
        #print(ret_mant)

        inttofp = bf16.Bfloat16.compose_bf16(ret_sign, ret_exp, ret_mant)
        return inttofp
