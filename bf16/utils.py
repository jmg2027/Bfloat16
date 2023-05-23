from typing import Tuple
import numpy as np
import tensorflow as tf
import struct

def float_to_hex(f):    # f: float number, outout: string hex code
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def float_to_hex64(f):    # f: float number, outout: string hex code
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

def float16_to_hex(f):
    return hex(struct.unpack('<H', np.float16('f').tobytes())[0])

def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])
    
def double_to_hex64(d):
    return hex(struct.unpack('<Q', struct.pack('<d', d))[0])

def hex_to_float(f):    # input: string hex w/o 0x, output: float number
    return struct.unpack('!f', bytes.fromhex(f))[0]

def hex_to_double(f):
    return struct.unpack('!d', bytes.fromhex(f))[0]

def hex64_to_double(f):
    return struct.unpack('!d', bytes.fromhex(f))[0]

def int_to_hex(f):      # input: unsigned integer, output: string hex code
    packed = struct.pack("I", f)    
    hex_str = ''.join(f'{x:02x}' for x in packed[::-1])    
    return hex_str

def int64_to_hex(f):      # input: unsigned integer, output: string hex code
    packed = struct.pack("Q", f)
    hex_str = ''.join(f'{x:02x}' for x in packed[::-1])    
    return hex_str

def hex_to_int(f):      # input: string hex code, output: integer
    return int(f, 16)

def fp32_cast_int(f):
    return hex_to_int(float_to_hex(f))

def int_cast_fp32(f):
    return hex_to_float(int_to_hex(f)[2:])

def ext_exp(f, b = 8):
    return (hex_to_int(float_to_hex(f)) & 0x7F800000) >> 23

def ext_sign(f):
    return (hex_to_int(float_to_hex(f)) & 0x80000000) >> 31

def ext_mantissa(f, b = 23):
    return hex_to_int(float_to_hex(f)) & 0x7FFFFF

def decomp_fp32(f):
    s = ext_sign(f)
    e = ext_exp(f)
    f = ext_mantissa(f)
    return s, e, f

def comp_fp32(s, e, f):
    return np.float32(hex_to_float(int_to_hex(s<<31|e<<23|f)))

def convert_float_int(s: int, e: int, f: int, sign_bitpos: int = 63, exp_bits: int = 11, mant_msb: int = 52, mant_bits: int = 7) -> int:
    """
    sign_bitpos, exp_bits, mant_msb: Sign bit posigion, exponent bits, mantissa MSB of the format to convert from
    mant_bits: Mantissa bits of the format to be converted
    """
    bias = (1 << (exp_bits - 1)) - 1
    mant_shft = mant_msb - mant_bits
    return (s << sign_bitpos | (e + bias) << mant_msb | f << mant_shft)

    
def bf16_ulp_dist(a, b):
    return abs((fp32_cast_int(a)>>16) - (fp32_cast_int(b)>>16))

def float_to_bf16(x):
    """_summary_
    convert fp32 to bf16 using tensorflow type casting

    Args:
        x (_type_): fp32

    Returns:
        _type_: bf16
    """
    return tf.cast(x, tf.bfloat16)

"""
# This function is pseudo-code
def from_lower_fp_to_higher_fp(sign, exponent, mantissa):
    #higher_fp: a = a_sign_bits, a_exponent_bits, a_mantissa_bits, a_bias
    #lower_fp: b = b_exponent_bits, b_mantissa_bits
    return sign << a_sign_bits | exponent + a_bias << exponent_bits | mantissa << (mantissa_bits - mantissa_bits)
"""

def get_hidden_bit(exponent: int) -> int:
    return 0 if exponent == 0 else 1

def round_to_nearest_even_mantissa(mant: int, n = 23, m = 7):
    """
    This method does not care about carry comes out from round up carry from all 1s
    To deal with it, use another method
    """
    if m > n:
        raise ValueError("Bitwidth of before truncation n should be larger than one of after truncation m")

    guard_bit = 0 if (n - 1) == m else (mant >> (n - m - 1)) & 1
    round_bit = 0 if (n - 2) == m else (mant >> (n - m - 2)) & 1
    sticky_bit = 0 if (n - 3) == m else (mant << (m + 3))
    truncated_number = mant >> (n-m)

    if guard_bit == 0:
        rounded_number = truncated_number
    else:
        if round_bit == 1:
            rounded_number = truncated_number + 1
        else:
            if sticky_bit != 0:
                rounded_number = truncated_number + 1
            else:
                if truncated_number % 2 == 0:
                    rounded_number = truncated_number
                else:
                    rounded_number = truncated_number + 1
    return rounded_number

def post_normalize_mantissa(exp: int, mant: int, n: int = 7) -> Tuple[int, int]:
    """
    Post normalization
    This method finds if mantissa has overflown
    n means bitwidth of mantissa
    """
    return (exp + 1, 0) if (1 << (n + 1) - 1) == mant else (exp, mant)

def round_and_postnormalize(exp: int, mant: int, n: int = 23, m: int = 7,) -> Tuple[int, int]:
    """
    n: bitwidth of round number, m: bitwidth of rounded result
    """
    rnd_mant = round_to_nearest_even_mantissa(mant, n, m)
    post_exp, post_mant = post_normalize_mantissa(exp, rnd_mant, m)
    return post_exp, post_mant

