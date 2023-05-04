import numpy as np
import tensorflow as tf

from typing import Tuple

from bf16 import utils as util
from bf16.bitstring import BitString as bit
from hw_model.fp_misc_op.fpmiscop import FloatPowerofTwo as Pow
from hw_model.fp_misc_op.fpmiscop import FloatNegative as Neg


class Bfloat16:
    """
     - Bfloat16
    16-bit Floating point representation
    s e e e e e e e e m m m m m m m

    Contains integer form of sign, exponent, mantissa
    Exponent is not biased
    Initial value is NaN
    Bfloat16 -> bf16.hex(), fp32.hex(), fp64.hex()
    
    Use decompose_bf16 method in hw model
    DEN is treated as ZERO -> ref) https://arxiv.org/pdf/1905.12322.pdf

     - HW units
    +, x, FMA, negative, 2^n(n is integer, such as times 2, 1/2, 1/4 ...), summation(get n Bfloat16 and adder tree. variable align shift)
    Conversion(BF16toInt, InttoBF16) -> floor, ceiling

     - Need improve
    all raise statements can go to Bfloat16Error class
    """

    sign_bitpos = 16
    exponent_bits = 8
    mantissa_bits = 7
    bias = (1 << (exponent_bits - 1)) - 1
    exp_max = (1 << exponent_bits) - 1
    mant_max = (1 << mantissa_bits) - 1

    def __init__(self, sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max) -> None:
        # call setters in __init__
        self.set(sign, exponent, mantissa)

    def set(self, sign: int, exponent: int, mantissa: int) -> None:
        self.sign, self.exponent, self.mantissa = self.set_sign(sign), self.set_exponent(exponent), self.set_mantissa(mantissa)

    def set_sign(self, sign: int) -> None:
        if not (sign == 0 or sign) == 1:
            raise ValueError(f"Bfloat16 sign value must be 0 or 1")
        return sign

    def set_exponent(self, exponent: int) -> None:
        if not 0 <= exponent + self.bias < (1 << self.exponent_bits):
            raise ValueError(f"Bfloat16 exponent value must be in range of -128 ~ 127")
        return exponent

    def set_mantissa(self, mantissa: int) -> None:
        if not 0 <= mantissa < (1 << self.mantissa_bits):
            raise ValueError(f"Bfloat16 mantissa value must be in range of 0 ~ 128")
        return mantissa
    
    def isnan(self) -> bool:
        return self.exponent == self.bias and self.mantissa != 0
    
    def isden(self) -> bool:
        return self.exponent == 0 and self.mantissa != 0
    
    # den is treated as zero
    def iszero(self) -> bool:
#        return self.exponent == 0 and self.mantissa == 0
        return self.exponent == 0
    
    def isinf(self) -> bool:
        return self.exponent == self.bias and self.mantissa == 0
    
    def isoverflow(self) -> bool:
        flag = self.exponent > self.bias
        if flag:
            raise ValueError(f"Bfloat16 instance overflow occured")
        return flag

    def isunderflow(self) -> bool:
        flag = self.exponent < 0
        if flag:
            raise ValueError(f"Bfloat16 instance underflow occured")
        return flag
    
    def bin(self) -> str:
        """
        BF16 Floating point binary string
        Use this in hardware model
        """
        biased_exponent = self.exponent + self.bias
        return ''.join([format(self.sign, '01b'), format(biased_exponent, f'0{self.exponent_bits}b'), format(self.mantissa, f'0{self.mantissa_bits}b')])

    def decompose_bf16(self) -> Tuple['bit', 'bit', 'bit']:
        """
        To hardware input
        """
        binary_bf16 = self.bin()
        sign = binary_bf16[0]
        exponent = binary_bf16[1:1+self.exponent_bits]
        mantissa = binary_bf16[-self.mantissa_bits:]
        return bit(sign, 1), bit(exponent, self.exponent_bits), bit(mantissa, self.mantissa_bits)

    @classmethod
    def compose_bf16(cls, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> 'Bfloat16':
        """
        From hardware output
        """
        sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
        exponent = biased_exponent - cls.bias
        return Bfloat16(sign, exponent, mantissa)

#    def decompose_bf16_int(self) -> Tuple[str, str, str]:
#        """
#        returns integer form of sign, biased exponent, mantissa
#        """
#        return tuple(map(lambda x: int(x), self.decompose_bf16()))
    
    @classmethod
    def float_to_bf16(cls, float: float) -> 'Bfloat16':
        bf16_bias = cls.bias
        bf16_sign, bf16_exp_before_bias, fp32_mant = util.decomp_fp32(float)
        bf16_exp, bf16_mant = util.round_and_postnormalize(bf16_exp_before_bias, fp32_mant, 23, 7, 3)
        bf16_exp = bf16_exp - bf16_bias
        return Bfloat16(bf16_sign, bf16_exp, bf16_mant)
    
    def bf16_to_float(self) -> float:
        float_int = util.convert_float_int(self.sign, self.exponent, self.mantissa)
        float = util.hex64_to_double(util.int64_to_hex(float_int))
        return float

    def bf16_to_tfbf16(self) -> 'tf.bfloat16':
        """
        To tensorflow.bfloat16 type
        """
        return util.float_to_bf16(float(self))
    
    def bf16_to_fp32(self) -> 'np.float32':
        """
        To numpy.float32 type
        """
        fp32_int = util.convert_float_int(self.sign, self.exponent, self.mantissa, 31, 8, 23)
        return np.float32(util.hex_to_float(util.int_to_hex(fp32_int)))
    
    # Operations in HW component
    def __add__(self, other: 'Bfloat16') -> 'Bfloat16':
        if not isinstance(other, Bfloat16):
            raise TypeError("Both operands should be Bfloat16 objects.")
        
        addition = FloatAddition(self, other)
        return FloatAddition.add()

    def __mul__(self, other: 'Bfloat16') -> 'Bfloat16':
        if not isinstance(other, Bfloat16):
            raise TypeError("Both operands should be Bfloat16 objects.")

        multiplication = FloatMultiplication(self, other)
        return multiplication.multiply()

    # from_blahblah method
    # ex) from_fp32, from_fp64
    # ex) to_fp32, to_fp64

    def __int__(self) -> int:
        # extract integer part
        # HW comp
        # first software version
        return int(float(self))
    
    def pow(self, n: int) -> 'Bfloat16':
        if not isinstance(n, int):
            raise TypeError("Operand of power of 2 should be integer number.")

        pow = Pow(self, n)
        return pow.power()
    
    def __neg__(self) -> 'Bfloat16':
        neg = Neg(self)
        return neg.negative()

    # Representation
    def __float__(self):
        float =  self.bf16_to_float()
        return float

    def __repr__(self):
        return f"Bfloat16({float(self)}, sign = {self.sign}, exponent={self.exponent}, mantissa={self.mantissa})"


class Bfloat16Error(Exception):
    """
    Error messages for Bfloat16 class
    """
    pass
