import numpy as np
import tensorflow as tf
from jaxtyping import BFloat16 as tfbfloat16
from jaxtyping import Float32 as tffloat32

from abc import ABCMeta, abstractmethod, abstractclassmethod
from typing import Tuple, Optional, ClassVar, Union, TypeVar, Type, Generic, List
from typing_extensions import Self, TypeAlias

from float_class import utils as util

from .bitstring import BitString as bit
from .bitstring import SignedBitString as sbit
from .bitstring import UnsignedBitString as ubit

#FloatBaseT = TypeVar('FloatBaseT', bound='FloatBase')
#TfFloatT = TypeVar('TfFloatT', tfbfloat16, tffloat32)
#TestInputT: TypeAlias = Union[int, float, 'Bfloat16', 'Float32']

#from hw_model import (
#    Pow,
#    Neg,
#    FPtoInt,
#    InttoFP,
#    BF16toFP32,
#    FP32toBF16,
#    Mul,
#    Add,
#    Fma,
#    Summation,
#    MRU,
#)
from hw_model import *


class FloatBase(metaclass=ABCMeta):
    """
    Base class for floating-point numbers.
    """
    _sign_bitpos: int = 64
    _exponent_bits: int = 11
    _mantissa_bits: int = 52

    @property
    def sign_bitpos(self) -> int:
        return self._sign_bitpos

    @property
    def exponent_bits(self) -> int:
        return self._exponent_bits

    @property
    def mantissa_bits(self) -> int:
        return self._mantissa_bits

    @property
    def bias(self) -> int:
        return (1 << (self._exponent_bits - 1)) - 1

    @property
    def exp_max(self) -> int:
        return (1 << (self._exponent_bits - 1))

    @property
    def mant_max(self) -> int:
        return (1 << self._mantissa_bits) - 1

    @classmethod
    def _bias(cls) -> int:
        return (1 << (cls._exponent_bits - 1)) - 1

    @classmethod
    def _exp_max(cls) -> int:
        return (1 << (cls._exponent_bits - 1))

    @classmethod
    def _mant_max(cls) -> int:
        return (1 << cls._mantissa_bits) - 1

    def __init__(self, sign: int, exponent: int, mantissa: int,     
                _sign_bitpos: int = 64, 
                _exponent_bits: int = 11, 
                _mantissa_bits: int = 52) -> None:
        # call setter in __init__
        self._sign_bitpos = _sign_bitpos
        self._exponent_bits = _exponent_bits
        self._mantissa_bits = _mantissa_bits
        self.set(sign, exponent, mantissa)

    def set(self, sign: int, exponent: int, mantissa: int) -> None:
        self.sign: int = self.set_sign(sign)
        self.exponent: int = self.set_exponent(exponent)
        self.mantissa: int = self.set_mantissa(mantissa)

    def set_sign(self, sign: int) -> int:
        if not (sign == 0 or sign == 1):
            raise FloatValueError(f"{self.__class__.__name__} sign value must be 0 or 1")
        return sign

    def set_exponent(self, exponent: int) -> int:
        if not 0 - self.bias <= exponent <= self.exp_max:
            raise FloatValueError(f"{self.__class__.__name__} exponent value must be in range of {-self._bias()} ~ {self.exp_max}")
        return exponent

    def set_mantissa(self, mantissa: int) -> int:
        if not 0 <= mantissa <= self.mant_max:
            raise FloatValueError(f"{self.__class__.__name__} mantissa value must be in range of 0 ~ {self.mant_max}")
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
            raise FloatValueError(f"Bfloat16 instance overflow occured")
        return flag

    def isunderflow(self) -> bool:
        flag = self.exponent < 0 - self.bias
        if flag:
            raise FloatValueError(f"Bfloat16 instance underflow occured")
        return flag
    
    def bin(self) -> str:
        """
        BF16 Floating point binary string
        Use this in hardware model
        """
        biased_exponent = self.exponent + self.bias
        return ''.join([format(self.sign, '01b'), format(biased_exponent, f'0{self.exponent_bits}b'), format(self.mantissa, f'0{self.mantissa_bits}b')])

    def hex(self) -> str:
        return f'0x{hex(int(self.bin(), 2))[2:].zfill(self.sign_bitpos//4)}'
    
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
    def compose(cls, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> Self:
        """
        From hardware output
        """
        sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
        exponent = biased_exponent - cls._bias()
        return cls(sign, exponent, mantissa)

    @classmethod
    def from_float(cls, fp: float) -> Self:
        bf16_bias = cls._bias()
        bf16_sign, bf16_exp_before_bias, fp32_mant = util.decomp_fp32(float(fp))
        bf16_exp, bf16_mant = util.round_and_postnormalize(bf16_exp_before_bias, fp32_mant, 23, 7)
        bf16_exp = bf16_exp - bf16_bias
        return cls(bf16_sign, bf16_exp, bf16_mant)

    @classmethod
    def from_hex(cls, h: int) -> Self:
        sign = h >> (cls._sign_bitpos - 1)
        exp_bit_msb = cls._sign_bitpos - 1
        exp_bit_lsb = cls._sign_bitpos - 1 - cls._exponent_bits
        exp_mask = ((1 << exp_bit_msb) - 1) - ((1 << exp_bit_lsb) -1)
        exponent = ((h & exp_mask) >> (cls._sign_bitpos - 1 - cls._exponent_bits)) - cls._bias()
        mant_mask = cls._mant_max()
        mantissa = h & mant_mask
        return cls(sign, exponent, mantissa)
    
    # Operations in HW component
    def __add__(self, other: Self) -> Self:
        if not isinstance(other, FloatBase):
            raise FloatTypeError("Both operands should be FloatBase objects.")
        if type(self) is not type(other):
            raise FloatTypeError("Both operands should be of the same type.")
        addition = Add(self, other)
        return addition.add()

    def __mul__(self, other: Self) -> Self:
        if not isinstance(other, FloatBase):
            raise FloatTypeError("Both operands should be FloatBase objects.")
        if type(self) is not type(other):
            raise FloatTypeError("Both operands should be of the same type.")
        multiplication = Mul(self, other)
        return multiplication.multiply()

    @classmethod
    def fma(cls, a: Self, b: Self, c: Self) -> Self:
        if not isinstance(a or b or c, FloatBase):
            raise FloatTypeError("Three of operands should be FloatBase objects.")
        if not (type(a) == type(b) == type(c)):
            raise FloatTypeError("Three of operands should be of the same type.")
        fma = Fma(a, b, c)
        return fma.fma()
    
    '''
    @classmethod
    def mru(cls, input_vector, weight_vector) -> Self:
        for v in input_vector:
            if not isinstance(v, FloatBase):
                raise FloatTypeError("All input vector operands should be FloatBase objects.")
        for v in weight_vector:
            if not isinstance(v, FloatBase):
                raise FloatTypeError("All weight vector operands should be FloatBase objects.")
        mru = MRU(input_vector, weight_vector)
        return MRU.summation()
    '''
    
    @classmethod
    #def summation(cls: Type[FloatBaseT], vector_list: List[List[FloatBaseT]]) -> 'Float32':
    def summation(cls: Type[Self], vector_list: List[List[Self]]) -> 'Float32':
        '''
        Vector format:
        [[vector 0], [vector 1], ..., [vector n]]
        Result:
        reduce_sum(vector 0, vector 1, ..., vector n)
        '''
        for vl in vector_list:
            for v in vl:
                if not isinstance(v, FloatBase):
                    raise FloatTypeError("All input vector operands should be FloatBase objects.")
        # Convert to FP32
        if cls == Bfloat16:
            fp32_vector_list = [[cls.bf16_to_fp32(v) for v in vl] for vl in vector_list]
        elif cls == Float32:
            fp32_vector_list = [[v for v in vl] for vl in vector_list]
        else:
            raise FloatTypeError('Summation should be for FloatBase')
        summation = Summation(fp32_vector_list)
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
    def inttofp(cls, i: int) -> None:
        pass
    
    def pow(self, n: int) -> Self:
        if not isinstance(n, int):
            raise FloatTypeError("Operand of power of 2 should be integer number.")
        pow = Pow(self, n)
        return pow.power()
    
    def __neg__(self) -> 'FloatBase':
        neg = Neg(self)
        return neg.negative()

    @abstractmethod
    def to_float(self) -> float:
        pass

    def bf16_to_fp32(self) -> 'Float32':
            raise NotImplementedError("This method should be implemented by subclasses if needed")

    def fp32_to_bf16(self) -> 'Bfloat16':
            raise NotImplementedError("This method should be implemented by subclasses if needed")

    #@abstractmethod
    #def to_tftype(self): # type: ignore
    #    pass

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
            float_repr = self.to_float()
        return float_repr


    def __repr__(self):
        return f"{self.__class__.__name__}(sign = {self.sign}, exponent={self.exponent}, mantissa={self.mantissa})"

class Bfloat16(FloatBase):
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

    _sign_bitpos = 16
    _exponent_bits = 8
    _mantissa_bits = 7

    def __init__(self, sign: int, exponent: int, mantissa: int) -> None:
        super().__init__(sign, exponent, mantissa, self._sign_bitpos, self._exponent_bits, self._mantissa_bits)
    
    @classmethod
    def inttofp(cls, i: int) -> Self:
        # HW component
        # Exists in fpmiscop
        # Assuming 32-bit integer
        if (len(bin(i))-2) > 32:
            raise FloatTypeError("FloatBase inttofp accepts only 32-bit integer")
        inttofp_obj = InttoFP[Bfloat16](i, 0)
        return inttofp_obj.inttofp()
    
    def to_float(self) -> float:
        f_int: int = util.convert_float_int(self.sign, self.exponent, self.mantissa)
        f: float = util.hex64_to_double(util.int64_to_hex(f_int))
        return f

    def bf16_to_tfbf16(self) -> tfbfloat16:
        return util.float_to_tfbf16(float(self))
    
    def bf16_to_fp32(self) -> 'Float32':
        """
        FIX: Bfloat16 -> Float32
        """
        bf16tofp32 = BF16toFP32(self)
        return bf16tofp32.bf16_to_fp32()
    


class Float32(FloatBase):
    """
     - Float32
    Single precision 32-bit Floating point representation

    Contains integer form of sign, exponent, mantissa
    Exponent is not biased
    No initial value
    
    Use decompose method in hw model
    DEN is treated as ZERO -> ref) https://arxiv.org/pdf/1905.12322.pdf

     - HW units
    +, x, FMA, negative, 2^n(n is integer, such as times 2, 1/2, 1/4 ...), summation(get n Bfloat16 and adder tree. variable align shift)
    Conversion(FP32toInt, InttoFP32) -> floor, ceiling
    """

    _sign_bitpos = 32
    _exponent_bits = 8
    _mantissa_bits = 23

    def __init__(self, sign: int, exponent: int, mantissa: int) -> None:
        super().__init__(sign, exponent, mantissa)
    
    @classmethod
    def inttofp(cls, i: int) -> Self:
        # HW component
        # Exists in fpmiscop
        # Assuming 32-bit integer
        if (len(bin(i))-2) > 32:
            raise FloatTypeError("FloatBase inttofp accepts only 32-bit integer")
        inttofp_obj = InttoFP[Float32](i, 1)
        return inttofp_obj.inttofp()
    
    def to_float(self) -> float:
        float_int = util.convert_float_int(self.sign, self.exponent, self.mantissa, 63, 11, 52, 23)
        float = util.hex64_to_double(util.int64_to_hex(float_int))
        return float

    def fp32_to_tffp32(self) -> tffloat32:
        return util.float_to_tffp32(float(self))
    
    def fp32_to_bf16(self) -> 'Bfloat16':
        """
        FIX: Float32 to Bfloat16
        """
        fp32tobf16 = FP32toBF16(self)
        return fp32tobf16.fp32_to_bf16()

    # from_blahblah method
    # ex) from_fp32, from_fp64
    # ex) to_fp32, to_fp64

    
class FloatTypeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class FloatValueError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
