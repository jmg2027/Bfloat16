import numpy as np
import tensorflow as tf

from typing import Tuple

from bf16 import utils as util
from bf16.bitstring import BitString as bit
from bf16.bitstring import SignedBitString as sbit
from bf16.bitstring import UnsignedBitString as ubit

from hw_model.fp_misc_op.fpmiscop import FloatPowerofTwo as Pow
from hw_model.fp_misc_op.fpmiscop import FloatNegative as Neg
from hw_model.fp_misc_op.fpmiscop import FloatFPtoInt as FPtoInt
from hw_model.fp_misc_op.fpmiscop import FloatInttoFP as InttoFP
from hw_model.fp_misc_op.fpmiscop import FloatBfloat16toFloat32 as BF16toFP32
from hw_model.fp_misc_op.fpmiscop import FloatFloat32toBfloat16 as FP32toBF16
from hw_model.fp_mul.fpmul import FloatMultiplication as Mul
from hw_model.fp_add.fpadd import FloatAddition as Add
from hw_model.fp_fma.fpfma import FloatFMA as Fma
from hw_model.fp_sum.fpsum import FloatSummation as Summation
from hw_model.fp_mru.fpmru import FloatMRU as MRU

from hw_model.utils import utils as hwutil


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
    exp_max = (1 << (exponent_bits - 1))
    mant_max = (1 << mantissa_bits) - 1

    def __init__(self, sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max) -> None:
        # call setters in __init__
        self.set(sign, exponent, mantissa)

    def set(self, sign: int, exponent: int, mantissa: int) -> None:
        self.sign, self.exponent, self.mantissa = self.set_sign(sign), self.set_exponent(exponent), self.set_mantissa(mantissa)

    def set_sign(self, sign: int) -> None:
        if not (sign == 0 or sign == 1):
            raise ValueError(f"Bfloat16 sign value must be 0 or 1")
        return sign

    def set_exponent(self, exponent: int) -> None:
        if not 0 - self.bias <= exponent <= self.exp_max:
            raise ValueError(f"Bfloat16 exponent value must be in range of -127 ~ 128")
        return exponent

    def set_mantissa(self, mantissa: int) -> None:
        if not 0 <= mantissa <= self.mant_max:
            raise ValueError(f"Bfloat16 mantissa value must be in range of 0 ~ 127")
        return mantissa
    
    def isnan(self) -> bool:
        return self.exponent == self.exp_max and self.mantissa != 0
    
    def isden(self) -> bool:
        return self.exponent == 0 - self.bias and self.mantissa != 0
    
    # den is treated as zero
    def iszero(self) -> bool:
#        return self.exponent == 0 and self.mantissa == 0
        return self.exponent == 0 - self.bias
    
    def isinf(self) -> bool:
        return self.exponent == self.exp_max and self.mantissa == 0
    
    def isoverflow(self) -> bool:
        flag = self.exponent > self.exp_max
        if flag:
            raise ValueError(f"Bfloat16 instance overflow occured")
        return flag

    def isunderflow(self) -> bool:
        flag = self.exponent < 0 - self.bias
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

    def decompose(self) -> Tuple['bit', 'bit', 'bit']:
        """
        To hardware input
        """
        binary_fp = self.bin()
        sign = binary_fp[0]
        exponent = binary_fp[1:1+self.exponent_bits]
        mantissa = binary_fp[-self.mantissa_bits:]
        return bit(1, sign), bit(self.exponent_bits, exponent), bit(self.mantissa_bits, mantissa)

    @classmethod
    def compose_bf16(cls, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> 'Bfloat16':
        """
        From hardware output
        """
        sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
        exponent = biased_exponent - cls.bias
        return Bfloat16(sign, exponent, mantissa)

    @classmethod
    def float_to_bf16(cls, fp: float) -> 'Bfloat16':
        bf16_bias = cls.bias
        bf16_sign, bf16_exp_before_bias, fp32_mant = util.decomp_fp32(float(fp))
        bf16_exp, bf16_mant = util.round_and_postnormalize(bf16_exp_before_bias, fp32_mant, 23, 7)
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
        return util.float_to_tfbf16(float(self))
    
    def bf16_to_fp32(self) -> 'Float32':
        """
        FIX: Bfloat16 -> Float32
        """
        bf16tofp32 = BF16toFP32(self)
        return bf16tofp32.bf16_to_fp32()
    
    # Operations in HW component
    def __add__(self, other: 'Bfloat16') -> 'Bfloat16':
        if not isinstance(other, Bfloat16):
            raise TypeError("Both operands should be Bfloat16 objects.")
        addition = Add(self, other)
        return addition.add()

    def __mul__(self, other: 'Bfloat16') -> 'Bfloat16':
        if not isinstance(other, Bfloat16):
            raise TypeError("Both operands should be Bfloat16 objects.")
        multiplication = Mul(self, other)
        return multiplication.multiply()

    @classmethod
    def fma(cls, a: 'Bfloat16', b: 'Bfloat16', c: 'Bfloat16') -> 'Bfloat16':
        if not isinstance(a or b or c, Bfloat16):
            raise TypeError("Three of operands should be Bfloat16 objects.")
        fma = Fma(a, b, c)
        return fma.fma()
    
    @classmethod
    def mru(cls, input_vector, weight_vector):
        for v in input_vector:
            if not isinstance(v, Bfloat16):
                raise TypeError("All input vector operands should be Bfloat16 objects.")
        for v in weight_vector:
            if not isinstance(v, Bfloat16):
                raise TypeError("All weight vector operands should be Bfloat16 objects.")
        mru = MRU(input_vector, weight_vector)
        return MRU.summation()
    
    @classmethod
    def summation(cls, vector_list):
        '''
        Vector format:
        [[vector 0], [vector 1], ..., [vector n]]
        Result:
        reduce_sum(vector 0, vector 1, ..., vector n)
        '''
        for vl in vector_list:
            for v in vl:
                if not isinstance(v, Bfloat16):
                    raise TypeError("All input vector operands should be Bfloat16 objects.")
        # Convert to FP32
        fp32_vector_list = [[cls.bf16_to_fp32(v) for v in vl] for vl in vector_list]
        summation = Summation(fp32_vector_list)
        summation.set_acc(cls(0, -127, 0))
        for v in summation.vector_list:
            print('acc', summation.acc)
            summation.set_vector(v)
            acc = summation.summation()
            summation.set_acc(acc)
        return summation.acc

    # from_blahblah method
    # ex) from_fp32, from_fp64
    # ex) to_fp32, to_fp64

    def __int__(self) -> int:
        # extract integer part
        return int(float(self))
    
    def fptoint(self) -> int:
        # HW component
        # Exists in fpmiscop
        fptoint_obj = FPtoInt(self)
        return fptoint_obj.fptoint()
    
    @classmethod
    def inttofp(cls, i: int) -> 'Bfloat16':
        # HW component
        # Exists in fpmiscop
        # Assuming 32-bit integer
        if (len(bin(i))-2) > 32:
            raise TypeError("Bfloat16 inttofp accepts only 32-bit integer")
        inttofp_obj = InttoFP(i)
        return inttofp_obj.inttofp().fp32_to_bf16()
    
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
        if self.iszero():
            float_repr = 0.0
        elif self.isnan():
            float_repr = float('nan')
        elif self.isinf() and self.sign == 0:
            float_repr = float('inf')
        elif self.isinf() and self.sign == 1:
            float_repr = float('-inf')
        else:
            float_repr = self.bf16_to_float()
        return float_repr
    
    def __repr__(self):
        return f"Bfloat16({float(self)}, sign = {self.sign}, exponent={self.exponent}, mantissa={self.mantissa})"


class Bfloat16Error(Exception):
    """
    Error messages for Bfloat16 class
    """
    pass


class Float32:
    """
     - Float32
    Single precision 32-bit Floating point representation

    Contains integer form of sign, exponent, mantissa
    Exponent is not biased
    Initial value is NaN
    
    Use decompose_bf16 method in hw model
    DEN is treated as ZERO -> ref) https://arxiv.org/pdf/1905.12322.pdf

     - HW units
    +, x, FMA, negative, 2^n(n is integer, such as times 2, 1/2, 1/4 ...), summation(get n Bfloat16 and adder tree. variable align shift)
    Conversion(FP32toInt, InttoFP32) -> floor, ceiling

     - Need improve
    all raise statements can go to Bfloat16Error class
    """

    sign_bitpos = 32
    exponent_bits = 8
    mantissa_bits = 23
    bias = (1 << (exponent_bits - 1)) - 1
    exp_max = (1 << (exponent_bits - 1))
    mant_max = (1 << mantissa_bits) - 1

    def __init__(self, sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max) -> None:
        # call setters in __init__
        self.set(sign, exponent, mantissa)

    def set(self, sign: int, exponent: int, mantissa: int) -> None:
        self.sign, self.exponent, self.mantissa = self.set_sign(sign), self.set_exponent(exponent), self.set_mantissa(mantissa)

    def set_sign(self, sign: int) -> None:
        if not (sign == 0 or sign == 1):
            raise ValueError(f"Float32 sign value must be 0 or 1")
        return sign

    def set_exponent(self, exponent: int) -> None:
        if not 0 - self.bias <= exponent <= self.exp_max:
            raise ValueError(f"Float32 exponent value must be in range of -127 ~ 128")
        return exponent

    def set_mantissa(self, mantissa: int) -> None:
        if not 0 <= mantissa <= self.mant_max:
            raise ValueError(f"Float32 mantissa value must be in range of 0 ~ 2^23-1")
        return mantissa
    
    def isnan(self) -> bool:
        return self.exponent == self.exp_max and self.mantissa != 0
    
    def isden(self) -> bool:
        return self.exponent == 0 - self.bias and self.mantissa != 0
    
    # den is treated as zero
    def iszero(self) -> bool:
#        return self.exponent == 0 and self.mantissa == 0
        return self.exponent == 0 - self.bias
    
    def isinf(self) -> bool:
        return self.exponent == self.exp_max and self.mantissa == 0
    
    def isoverflow(self) -> bool:
        flag = self.exponent > self.exp_max
        if flag:
            raise ValueError(f"Bfloat16 instance overflow occured")
        return flag

    def isunderflow(self) -> bool:
        flag = self.exponent < 0 - self.bias
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

    def decompose(self) -> Tuple['bit', 'bit', 'bit']:
        """
        To hardware input
        """
        binary_fp = self.bin()
        sign = binary_fp[0]
        exponent = binary_fp[1:1+self.exponent_bits]
        mantissa = binary_fp[-self.mantissa_bits:]
        return bit(1, sign), bit(self.exponent_bits, exponent), bit(self.mantissa_bits, mantissa)

    @classmethod
    def compose_fp32(cls, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> 'Float32':
        """
        From hardware output
        """
        sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
        exponent = biased_exponent - cls.bias
        return Float32(sign, exponent, mantissa)

    @classmethod
    def float_to_fp32(cls, fp: float) -> 'Float32':
        fp32_bias = cls.bias
        fp32_sign, fp32_exp_before_bias, fp32_mant = util.decomp_fp32(float(fp))
        fp32_exp = fp32_exp_before_bias - fp32_bias
        return Float32(fp32_sign, fp32_exp, fp32_mant)
    
    def fp32_to_float(self) -> float:
        float_int = util.convert_float_int(self.sign, self.exponent, self.mantissa, 63, 11, 52, 23)
        float = util.hex64_to_double(util.int64_to_hex(float_int))
        return float

    def fp32_to_tffp32(self) -> 'tf.float32':
        """
        To tensorflow.bfloat16 type
        """
        return util.float_to_tffp32(float(self))
    
    def fp32_to_bf16(self) -> 'Bfloat16':
        """
        FIX: Float32 to Bfloat16
        """
        fp32tobf16 = FP32toBF16(self)
        return fp32tobf16.fp32_to_bf16()
    
    # Operations in HW component
    def __add__(self, other: 'Float32') -> 'Float32':
        if not isinstance(other, Float32):
            raise TypeError("Both operands should be Float32 objects.")
        addition = Add(self, other)
        return addition.add()

    def __mul__(self, other: 'Float32') -> 'Float32':
        if not isinstance(other, Float32):
            raise TypeError("Both operands should be Float32 objects.")
        multiplication = Mul(self, other)
        return multiplication.multiply()

    @classmethod
    def fma(cls, a: 'Float32', b: 'Float32', c: 'Float32') -> 'Float32':
        if not isinstance(a or b or c, Float32):
            raise TypeError("Three of operands should be Float32 objects.")
        fma = Fma(a, b, c)
        return fma.fma()
    
    @classmethod
    def mru(cls, input_vector, weight_vector):
        for v in input_vector:
            if not isinstance(v, Float32):
                raise TypeError("All input vector operands should be Float32 objects.")
        for v in weight_vector:
            if not isinstance(v, Float32):
                raise TypeError("All weight vector operands should be Float32 objects.")
        mru = MRU(input_vector, weight_vector)
        return MRU.summation()
   
    @classmethod
    def summation(cls, vector_list):
        '''
        Vector format:
        [[vector 0], [vector 1], ..., [vector n]]
        Result:
        reduce_sum(vector 0, vector 1, ..., vector n)
        '''
        for vl in vector_list:
            for v in vl:
                if not isinstance(v, Float32):
                    raise TypeError("All input vector operands should be Float32 objects.")
        summation = Summation(vector_list)
        summation.set_acc(cls(0, -127, 0))
        for v in summation.vector_list:
            summation.set_vector(v)
            acc = summation.summation()
            summation.set_acc(acc)
        return summation.acc 

    # from_blahblah method
    # ex) from_fp32, from_fp64
    # ex) to_fp32, to_fp64

    def __int__(self) -> int:
        # extract integer part
        return int(float(self))
    
    def fptoint(self) -> int:
        # HW component
        # Exists in fpmiscop
        fptoint_obj = FPtoInt(self)
        return fptoint_obj.fptoint()
    
    @classmethod
    def inttofp(cls, i: int) -> 'Float32':
        # HW component
        # Exists in fpmiscop
        # Assuming 16-bit integer
        if (len(bin(i))-2) > 32:
            raise TypeError("Float32 inttofp accepts only 32-bit integer")
        inttofp_obj = InttoFP(i)
        return inttofp_obj.inttofp()
    
    def pow(self, n: int) -> 'Float32':
        if not isinstance(n, int):
            raise TypeError("Operand of power of 2 should be integer number.")
        pow = Pow(self, n)
        return pow.power()
    
    def __neg__(self) -> 'Float32':
        neg = Neg(self)
        return neg.negative()

    # Representation
    def __float__(self):
        if self.iszero():
            float_repr = 0.0
        elif self.isnan():
            float_repr = float('nan')
        elif self.isinf() and self.sign == 0:
            float_repr = float('inf')
        elif self.isinf() and self.sign == 1:
            float_repr = float('-inf')
        else:
            float_repr = self.fp32_to_float()
        return float_repr
    
    def __repr__(self):
        return f"Float32({float(self)}, sign = {self.sign}, exponent={self.exponent}, mantissa={self.mantissa})"

