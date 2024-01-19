import numpy as np
import tensorflow as tf
from jaxtyping import BFloat16 as tfbfloat16
from jaxtyping import Float32 as tffloat32

from abc import ABCMeta, abstractmethod, abstractclassmethod
from typing import Tuple, Optional, ClassVar, Union, TypeVar, Type, Generic, List, Dict, Callable
from typing_extensions import Self, TypeAlias

from float_class import utils as util

from .floatint import _FPConfig
from .floatint import FloatBaseInt
from .floatint import _bf16_config
from .floatint import _fp32_config
from .floatint import _fp64_config
from .floatint import FloatTypeError
from .floatint import FloatValueError

from .bitstring import BitString as bit
from .bitstring import SignedBitString as sbit
from .bitstring import UnsignedBitString as ubit

import float_class.hw_model as hw_model

FPBitT = Tuple[bit, bit, bit]

class FloatBase(metaclass=ABCMeta):
    #def __init__(self, sign: int, exponent: int, mantissa: int,     
    #            config: _FPConfig) -> None:
    #    # call setter in __init__
    #    self._config = config
    #    self.set(sign, exponent, mantissa)
    def __init__(self, sign: int, exponent: int, mantissa: int,     
                config: _FPConfig) -> None:
        # call setter in __init__
        self.fp_int = FloatBaseInt(sign, exponent, mantissa, config)

    def isnan(self) -> bool:
        return self.fp_int.exponent == self.fp_int._config.exp_max and self.fp_int.mantissa != 0
    
    def isden(self) -> bool:
        return self.fp_int.exponent == 0 - self.fp_int._config.bias and self.fp_int.mantissa != 0
    
    # den is treated as zero
    def iszero(self) -> bool:
#        return self.exponent == 0 and self.mantissa == 0
        return self.fp_int.exponent == 0 - self.fp_int._config.bias
    
    def isinf(self) -> bool:
        return self.fp_int.exponent == self.fp_int._config.exp_max and self.fp_int.mantissa == 0
    
    #def isoverflow(self) -> bool:
    #    flag = self.exponent > self._config.exp_max
    #    if flag:
    #        raise FloatValueError(f"Bfloat16 instance overflow occured")
    #    return flag
    #
    #def isunderflow(self) -> bool:
    #    flag = self.exponent < 0 - self._config.bias
    #    if flag:
    #        raise FloatValueError(f"Bfloat16 instance underflow occured")
    #    return flag
    
    def bin(self) -> str:
        """
        BF16 Floating point binary string
        Use this in hardware model
        """
        biased_exponent = self.fp_int.exponent + self.fp_int._config.bias
        return ''.join([format(self.fp_int.sign, '01b'), format(biased_exponent, f'0{self.fp_int._config.exponent_bits}b'), format(self.fp_int.mantissa, f'0{self.fp_int._config.mantissa_bits}b')])

    def hex(self) -> str:
        return f'0x{hex(int(self.bin(), 2))[2:].zfill(self.fp_int._config.sign_bitpos//4)}'

    def _from_float_pre_cal(self, fp: float) -> Tuple[int, int, int]:
        bias = self.fp_int._config.bias
        sign, exp_before_bias, mant = util.decomp_fp32(float(fp))
        return sign, exp_before_bias, mant

    @abstractmethod
    def from_float(self, fp: float) -> Self:
        pass

    def from_hex(self, h: int) -> Self:
        sign = h >> (self.fp_int._config.sign_bitpos - 1)
        exp_bit_msb = self.fp_int._config.sign_bitpos - 1
        exp_bit_lsb = self.fp_int._config.sign_bitpos - 1 - self.fp_int._config.exponent_bits
        exp_mask = ((1 << exp_bit_msb) - 1) - ((1 << exp_bit_lsb) -1)
        exponent = ((h & exp_mask) >> (self.fp_int._config.sign_bitpos - 1 - self.fp_int._config.exponent_bits)) - self.fp_int._config.bias
        mant_mask = self.fp_int._config.mant_max
        mantissa = h & mant_mask
        #return self.__class__(sign, exponent, mantissa, self.fp_int._config)
        return self.__class__(sign, exponent, mantissa)

    def decompose(self) -> Tuple['bit', 'bit', 'bit']:
        """
        To hardware input
        """
        binary_fp = self.bin()
        sign = binary_fp[0]
        exponent = binary_fp[1:1+self.fp_int._config.exponent_bits]
        mantissa = binary_fp[-self.fp_int._config.mantissa_bits:]
        return bit(1, sign), bit(self.fp_int._config.exponent_bits, exponent), bit(self.fp_int._config.mantissa_bits, mantissa)

    #@classmethod
    #def compose(cls, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> Self:
    #    """
    #    From hardware output
    #    """
    #    sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
    #    exponent = biased_exponent - cls._bias()
    #    return cls(sign, exponent, mantissa)
    def compose(self, sign_bin: 'bit', exponent_bin: 'bit', mantissa_bin: 'bit') -> Self:
        """
        From hardware output
        """
        sign, biased_exponent, mantissa = tuple(map(lambda x: int(x), (sign_bin, exponent_bin, mantissa_bin)))
        exponent = biased_exponent - self.fp_int._config.bias
        #return self.__class__(sign, exponent, mantissa, self.fp_int._config)
        return self.__class__(sign, exponent, mantissa)

    @abstractmethod
    def to_float(self) -> float:
        pass

    # Representation
    def __float__(self):
        if self.iszero():
            if self.fp_int.sign == 0:
                float_repr = 0.0
            else:
                float_repr = -0.0
        elif self.isnan():
            float_repr = float('nan')
        elif self.isinf() and self.fp_int.sign == 0:
            float_repr = float('inf')
        elif self.isinf() and self.fp_int.sign == 1:
            float_repr = float('-inf')
        else:
            float_repr = self.to_float()
        return float_repr

    def __repr__(self):
        return f"{self.__class__.__name__}({float(self)}, sign = {self.fp_int.sign}, exponent={self.fp_int.exponent}, mantissa={self.fp_int.mantissa})"

    @staticmethod
    def _validate_operands_to_mod(*ops, mod_structure: Dict[int, Tuple[Tuple[Type, ...], Type]]) -> int:
        '''
        mod_structure: {mod: ((input, ...), result), ...}
        fma_mod = {0: ((bf16, bf16, bf16), bf16), 1: ((bf16, bf16, fp32), fp32), 2: ((fp32, fp32, fp32), fp32)}
        '''
        mod: int = next((key for key, value in mod_structure.items() if tuple(map(type, ops)) == value[0]), -1)
        if mod == -1:
            raise FloatTypeError('TYPE_MISMATCH', value = (ops, tuple(map(lambda x: type(x), ops))), expected_type=mod_structure.values())
        return mod

    # Operations in HW component
    def _perform_sigle_operand_operation(self, op_class: Type) -> Self:
        a_bit: FPBitT = self.decompose()
        perform_op = op_class(a_bit)    # I don't want to annotate... This is hw_model classes
        result_bit = perform_op.execute()
        result = self.compose(result_bit[0], result_bit[1], result_bit[2])
        return result
    
    def _perform_two_operand_operation(self, other: Self, op_class: Type) -> Self:
        self._validate_operands_to_mod(self, other, mod_structure = {0: ((Bfloat16, Bfloat16), Bfloat16), 1: ((Float32, Float32), Float32)})
        bf16_input = isinstance(self, Bfloat16) & isinstance(other, Bfloat16)
        if bf16_input:
            self_fp32 = bf16_to_fp32(self)
            other_fp32 = bf16_to_fp32(other)
        else:
            self_fp32 = self
            other_fp32 = other
        a_bit: FPBitT = self_fp32.decompose()
        b_bit: FPBitT = other_fp32.decompose()
        perform_op = op_class(a_bit, b_bit)    # I don't want to annotate... This is hw_model classes
        result_bit = perform_op.execute()
        fp32 = Float32(0, 0, 0)
        result_fp32 = fp32.compose(result_bit[0], result_bit[1], result_bit[2])
        bf16_output = isinstance(self, Bfloat16) & isinstance(other, Bfloat16)
        if bf16_output:
            result = fp32_to_bf16(result_fp32)
        else:
            result = result_fp32
        return result

    def __add__(self, other: Self) -> Self:
        return self._perform_two_operand_operation(other, hw_model.Add)

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __mul__(self, other: Self) -> Self:
        return self._perform_two_operand_operation(other, hw_model.Mul)

    @abstractmethod
    def fma(cls, a, b, c, algorithm):
        pass

    @abstractmethod
    def summation(cls, vector_list, mod, algorithm):
        pass

    def __int__(self) -> int:
        # extract integer part
        return int(float(self))
    
    def fptoint(self) -> int:
        if isinstance(self, Bfloat16):
            fp32_input = bf16_to_fp32(self)
        else:
            fp32_input = self
        fp32_bit = fp32_input.decompose()
        fptoint_obj = hw_model.FPtoInt(fp32_bit)
        return fptoint_obj.execute()
    
    @classmethod
    def inttofp(cls, i: int) -> None:
        pass
    
    def pow(self, n: int) -> Self:
        if not isinstance(n, int):
            raise FloatTypeError("Operand of power of 2 should be integer number.")
        if isinstance(self, Bfloat16):
            fp32_input = bf16_to_fp32(self)
        else:
            fp32_input = self
        fp32_bit = fp32_input.decompose()
        pow = hw_model.Pow(fp32_bit, n)
        result_bit = pow.execute()
        result_fp32 = Float32(0, 0, 0)
        result_fp32 = result_fp32.compose(result_bit[0], result_bit[1], result_bit[2])
        if isinstance(self, Bfloat16):
            result = fp32_to_bf16(result_fp32)
        else:
            result = result_fp32
        return result
    
    def __neg__(self) -> 'FloatBase':
        if isinstance(self, Bfloat16):
            fp32_input = bf16_to_fp32(self)
        else:
            fp32_input = self
        fp32_bit = fp32_input.decompose()
        neg = hw_model.Neg(fp32_bit)
        result_bit = neg.execute()
        result_fp32 = Float32(0, 0, 0)
        result_fp32 = result_fp32.compose(result_bit[0], result_bit[1], result_bit[2])
        if isinstance(self, Bfloat16):
            result = fp32_to_bf16(result_fp32)
        else:
            result = result_fp32
        return result

    # Factory method
    @classmethod
    def create_instance(cls, sign: int, exponent: int, mantissa: int, config: Optional[_FPConfig] = None) -> Self:
        if cls == Bfloat16:
            return cls(sign, exponent, mantissa, _bf16_config)
        elif cls == Float32:
            return cls(sign, exponent, mantissa, _fp32_config)
        else:
            raise NotImplementedError("This method should be implemented by subclasses if needed")


    #@abstractmethod
    #def to_tftype(self): # type: ignore
    #    pass


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
    def __init__(self, sign: int, exponent: int, mantissa: int, config: _FPConfig = _bf16_config) -> None:
        super().__init__(sign, exponent, mantissa, _bf16_config)
    
    @classmethod
    def inttofp(cls, i: int) -> Self:
        # HW component
        # Exists in fpmiscop
        # Assuming 32-bit integer
        if not isinstance(i, int):
            raise FloatTypeError("InttoFP input should be integer number.")
        if (len(bin(i))-2) > 32:
            raise FloatTypeError('INVALID_INT', value = i, expected_type='32-bit integer')
        inttofp_obj = hw_model.InttoFP(i, 0)
        result_bit = inttofp_obj.execute()
        result = cls(0, 0, 0)
        result = result.compose(result_bit[0], result_bit[1], result_bit[2])
        return result

    def from_float(self, fp: float) -> Self:
        sign, exp_before_bias, mant = self._from_float_pre_cal(fp)
        bf16_exp, bf16_mant = util.round_and_postnormalize(exp_before_bias, mant, _fp32_config.mantissa_bits, _bf16_config.mantissa_bits)
        bf16_exp = bf16_exp - self.fp_int._config.bias
        return self.__class__(sign, bf16_exp, bf16_mant)

    def to_float(self) -> float:
        f_int: int = util.convert_float_int(self.fp_int.sign, self.fp_int.exponent, self.fp_int.mantissa)
        f: float = util.hex64_to_double(util.int64_to_hex(f_int))
        return f

    def bf16_to_tfbf16(self) -> tfbfloat16:
        return util.float_to_tfbf16(float(self))

    @classmethod
    #def fma(cls, a: 'Bfloat16', b: 'Bfloat16', c: 'Bfloat16', algorithm="SINGLE_PATH") -> 'Bfloat16':
    def fma(cls, a: 'Bfloat16', b: 'Bfloat16', c: 'Bfloat16', algorithm="MULTI_PATH") -> 'Bfloat16':
        # mod 0: a, b = fp32, c = fp32
        # mod 1: a, b = bf16, c = bf16
        # mod 2: a, b = bf16, c = fp32
        if not isinstance(a or b or c, cls):
            raise FloatTypeError('INVALID_OPERAND', value = (a, b, c), expected_type=type(cls))
        a_input = bf16_to_fp32(a)
        b_input = bf16_to_fp32(b)
        c_input = bf16_to_fp32(c)
        a_bit: Tuple[bit, bit, bit] = a_input.decompose()
        b_bit: Tuple[bit, bit, bit] = b_input.decompose()
        c_bit: Tuple[bit, bit, bit] = c_input.decompose()
        fma = hw_model.Fma(a_bit, b_bit, c_bit)
        result_bit = fma.execute(algorithm)
        result = Float32(0, 0, 0)
        result = result.compose(result_bit[0], result_bit[1], result_bit[2])
        result = fp32_to_bf16(result)
        return result

    @classmethod
    def summation(cls: Type[Self], vector_list: List[List[Self]], mod = 3, algorithm = "SINGLE_PATH") -> 'Bfloat16':
        #        vector input   scalar input   output
        # mod 0: fp32, fp32, fp32
        # mod 1: bf16, bf16, fp32
        # mod 2: bf16, fp32, fp32
        # mod 3: bf16, bf16, bf16
        '''
        Vector format:
        [[vector 0], [vector 1], ..., [vector n]]
        Result:
        reduce_sum(vector 0, vector 1, ..., vector n)
        '''
        for vl in vector_list:
            for v in vl:
                if not isinstance(v, cls):
                    raise FloatTypeError('INVALID_OPERAND', value = (vector_list))
        # initialize accumulator
        acc = cls(0, -126, 0)
        # Convert to FP32
        #fp32_vector_list = [[bf16_to_fp32(v) for v in vl] for vl in vector_list]
        result = Float32(0, 0, 0)
        
        #for v in fp32_vector_list:
        for v in vector_list:
            # convert acc to fp32
            acc = bf16_to_fp32(acc)
            # convert vector to fp32
            v = [bf16_to_fp32(e) for e in v]
            # decompose vector and acc
            acc_bit = acc.decompose()
            vector_bit = [e.decompose() for e in v]
            summation = hw_model.Summation(vector_bit, acc_bit, mod)
            acc_bit = summation.execute(algorithm)
            # compose acc
            acc = result.compose(acc_bit[0], acc_bit[1], acc_bit[2])
            # convert acc result to bf16
            acc = fp32_to_bf16(acc)
        return acc


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
    def __init__(self, sign: int, exponent: int, mantissa: int) -> None:
        super().__init__(sign, exponent, mantissa, _fp32_config)
    
    @classmethod
    def inttofp(cls, i: int) -> Self:
        # HW component
        # Exists in fpmiscop
        # Assuming 32-bit integer
        if not isinstance(i, int):
            raise FloatTypeError("InttoFP input should be integer number.")
        if (len(bin(i))-2) > 32:
            raise FloatTypeError('INVALID_INT', value = i, expected_type='32-bit integer')
        inttofp_obj = hw_model.InttoFP(i, 1)
        result_bit = inttofp_obj.execute()
        result = cls(0, 0, 0)
        result = result.compose(result_bit[0], result_bit[1], result_bit[2])
        return result

    def from_float(self, fp: float) -> Self:
        sign, exp_before_bias, mant = self._from_float_pre_cal(fp)
        exp = exp_before_bias - self.fp_int._config.bias
        return self.__class__(sign, exp, mant)
    
    def to_float(self) -> float:
        float_int = util.convert_float_int(self.fp_int.sign, self.fp_int.exponent, self.fp_int.mantissa, 
                                           (_fp64_config.sign_bitpos-1), _fp64_config.exponent_bits, _fp64_config.mantissa_bits, _fp32_config.mantissa_bits)
        float = util.hex64_to_double(util.int64_to_hex(float_int))
        return float

    def fp32_to_tffp32(self) -> tffloat32:
        return util.float_to_tffp32(float(self))

    @classmethod
    #def fma(cls, a: Union['Bfloat16', 'Float32'], b: Union['Bfloat16', 'Float32'], c: 'Float32', algorithm="SINGLE_PATH") -> Union['Bfloat16', 'Float32']:
    def fma(cls, a: Union['Bfloat16', 'Float32'], b: Union['Bfloat16', 'Float32'], c: 'Float32', algorithm="MULTI_PATH") -> Union['Bfloat16', 'Float32']:
        # mod 0: a, b = fp32, c = fp32
        # mod 1: a, b = bf16, c = bf16
        # mod 2: a, b = bf16, c = fp32
        mod_2 = isinstance(a, Bfloat16) & isinstance(b, Bfloat16) & isinstance(c, Float32)
        mod_0 = isinstance(a, Float32) & isinstance(b, Float32) & isinstance(c, Float32)
        if not mod_2 and not mod_0:
            raise FloatTypeError('INVALID_OPERAND', value = (a, b, c), expected_type='bf16/fp32, bf16/fp32, fp32')
        if mod_2:
            a_input = bf16_to_fp32(a)
            b_input = bf16_to_fp32(b)
            c_input = c
        else:
            a_input = a
            b_input = b
            c_input = c
        a_bit: Tuple[bit, bit, bit] = a_input.decompose()
        b_bit: Tuple[bit, bit, bit] = b_input.decompose()
        c_bit: Tuple[bit, bit, bit] = c_input.decompose()
        fma = hw_model.Fma(a_bit, b_bit, c_bit)
        result_bit = fma.execute(algorithm)
        result = Float32(0, 0, 0)
        result = result.compose(result_bit[0], result_bit[1], result_bit[2])
        return result

    @classmethod
    #def summation(cls: Type[Self], vector_list: List[List[Union['Bfloat16', Self]]], mod = 0) -> Union['Bfloat16', 'Float32']:
    def summation(cls: Type[Self], vector_list: List[List[Union['Bfloat16', Self]]], acc_input = 0.0, mod = 0, algorithm = "SINGLE_PATH") -> Union['Bfloat16', 'Float32']:
        #        vector input   scalar input   output
        # mod 0: fp32, fp32, fp32
        # mod 1: bf16, bf16, fp32
        # mod 2: bf16, fp32, fp32
        # mod 3: bf16, bf16, bf16
        '''
        Vector format:
        [[vector 0], [vector 1], ..., [vector n]]
        Result:
        reduce_sum(vector 0, vector 1, ..., vector n)
        '''
        for vl in vector_list:
            for v in vl:
                if mod == 0:
                    if not isinstance(v, cls):
                        raise FloatTypeError('INVALID_OPERAND', value = (vector_list), expected_type=cls)
                else:
                    if not isinstance(v, Bfloat16):
                        raise FloatTypeError('INVALID_OPERAND', value = (vector_list), expected_type=Bfloat16)
        result = Float32(0, 0, 0)
        if mod == 0:
            acc = cls(0, -126, 0)
            acc = acc.from_float(acc_input)
        elif mod == 1:
            acc = Bfloat16(0, -126, 0)
            acc = acc.from_float(acc_input)
        elif mod == 2:
            acc = cls(0, -126, 0)
            acc = acc.from_float(acc_input)
        else:
            raise FloatTypeError('INVALID_MOD', value = mod)
        for v in vector_list:
            if mod == 0:
                # no conversion
                acc = acc
                v = v
            elif mod == 1:
                # convert to fp32 for operation
                acc = bf16_to_fp32(acc)
                v = [bf16_to_fp32(e) for e in v]
            elif mod == 2:
                # convert vector to fp32 for operation
                acc = acc
                v = [bf16_to_fp32(e) for e in v]
            acc_bit = acc.decompose()
            vector_bit = [e.decompose() for e in v]
            #summation = hw_model.Summation(vector_bit, acc_bit)
            summation = hw_model.Summation(vector_bit, acc_bit, mod)
            acc_bit = summation.execute(algorithm)
            summation.set_acc(acc_bit)
            acc = result.compose(acc_bit[0], acc_bit[1], acc_bit[2])
            if mod == 1:
                acc = fp32_to_bf16(acc)
        return acc
    
    #def vth(self, )

# Converter function
def bf16_to_fp32(bf16: Bfloat16) -> Float32:
    """
    Bfloat16 -> Float32
    """
    a_bit: FPBitT = bf16.decompose()
    bf16tofp32 = hw_model.BF16toFP32(a_bit)
    result_bit = bf16tofp32.execute()
    result = Float32(0, 0, 0)
    result = result.compose(result_bit[0], result_bit[1], result_bit[2])
    return result

# converter function
def fp32_to_bf16(fp32: Float32) -> Bfloat16:
    """
    FIX: Float32 to Bfloat16
    """
    a_bit: FPBitT = fp32.decompose()
    fp32tobf16 = hw_model.FP32toBF16(a_bit)
    result_bit = fp32tobf16.execute()
    result = Bfloat16(0, 0, 0)
    result = result.compose(result_bit[0], result_bit[1], result_bit[2])
    return result


    # from_blahblah method
    # ex) from_fp32, from_fp64
    # ex) to_fp32, to_fp64
