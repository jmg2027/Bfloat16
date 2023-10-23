#from float_class import bf16, fp32, FloatBase, FloatTypeError
from typing import TypeVar, List, Dict, Tuple, Union, Type

from .floatclass import Bfloat16 as bf16
from .floatclass import Float32 as fp32
from .floatclass import FloatBase, FloatTypeError
from .floatclass import bf16_to_fp32, fp32_to_bf16
from .floatclass import bit
import float_class.hw_model as hw_model

# mod structure for operations
# (input, input)
# output imples accumulator, such as last operand

FloatBaseT = TypeVar('FloatBaseT', bf16, fp32)
FloatUnionT = Union[bf16, fp32]

ModStructureT = Dict[int, Tuple[Tuple[Type[FloatUnionT], ...], Type[FloatUnionT]]]
magic_method_mod: ModStructureT = {0: ((bf16, bf16), bf16), 1: ((fp32, fp32), fp32)}
fma_mod: ModStructureT = {0: ((bf16, bf16, bf16), bf16), 1: ((bf16, bf16, fp32), fp32), 2: ((fp32, fp32, fp32), fp32)}
sum_mod: ModStructureT = {0: ((List[List[bf16]],), bf16), 1: ((List[List[fp32]],), fp32)}   # List[List[FloatBaseT]] -> FloatBaseT

fp32_obj = fp32(0, 0, 0)

# Interface for operations
def add(a: FloatBaseT, b: FloatBaseT) -> FloatBaseT:
    # mod 0: a, b = bf16
    # mod 1: a, b = fp32
    mod = FloatBase._validate_operands_to_mod(a, b, mod_structure=magic_method_mod)
    return a + b

def sub(a: FloatBaseT, b: FloatBaseT) -> FloatBaseT:
    # mod 0: a, b = bf16
    # mod 1: a, b = fp32
    mod: int = FloatBase._validate_operands_to_mod(a, b, mod_structure=magic_method_mod)
    return a - b

def mul(a: FloatBaseT, b: FloatBaseT) -> FloatBaseT:
    # mod 0: a, b = bf16
    # mod 1: a, b = fp32
    mod: int = FloatBase._validate_operands_to_mod(a, b, mod_structure=magic_method_mod)
    print('Hi')
    return a * b

def fma(a: FloatUnionT, b: FloatUnionT, c: FloatUnionT) -> FloatUnionT:
    # mod 0: a, b = bf16, c = bf16
    # mod 1: a, b = bf16, c = fp32
    # mod 2: a, b = fp32, c = fp32
    mod = FloatBase._validate_operands_to_mod(a, b, c, mod_structure=fma_mod)
    return _FloatOperation.fma(a, b, c, mod, fma_mod)

def summation(vl: List[List[FloatBaseT]]) -> FloatBaseT:
    return vl


class _FloatOperation:
    # internal class for operations, except magic methods
    @staticmethod
    def fma(a: FloatUnionT, b: FloatUnionT, c: FloatUnionT, mod: int, fma_mod: Dict) -> FloatUnionT:
        # mod 0: a, b = bf16, c = bf16  
        # mod 1: a, b = bf16, c = fp32
        # mod 2: a, b = fp32, c = fp32
        if mod == 0:
            a = bf16_to_fp32(a)
            b = bf16_to_fp32(b)
            c = bf16_to_fp32(c)
        elif mod == 1:
            a = bf16_to_fp32(a)
            b = bf16_to_fp32(b)
        elif mod == 2:
            pass
        bit_repr = a.decompose(), b.decompose(), c.decompose()
        fma = hw_model.Fma(*bit_repr)
        result_bit = fma.excute()
        result = fp32_obj.compose(result_bit[0], result_bit[1], result_bit[2])
        if mod == 0:
            result = fp32_to_bf16(result)
        elif mod == 1:
            pass
        elif mod == 2:
            pass
        return result

    @staticmethod
    def summation(vl: List[List[FloatBaseT]], mod: int) -> FloatBaseT:
        if mod not in [0, 1, 2, 3]:
            raise ValueError("Invalid mod value")
            
        # Initialize the accumulator to start at zero. 
        # The type is determined by the mod.
        if mod in [0, 2]:
            accumulator = fp32(0, 0, 0)
        else:
            accumulator = bf16(0, 0, 0)

        for vec in vl:
            # Calculate the sum of a single vector.
            summation = hw_model.Summation(vec)
            result_bit = summation.excute()

            # Convert the result of hw_model.Summation to the actual type.
            if mod in [0, 2]:
                single_sum = fp32.compose(*result_bit)
            else:
                single_sum = bf16.compose(*result_bit)

            # Add the new sum to the previous accumulated value.
            accumulator += single_sum  # Using the + magic method

        # Convert the final result type based on the mod, if needed.
        if mod == 1:
            accumulator = bf16_to_fp32(accumulator)
        elif mod == 2:
            accumulator = fp32_to_bf16(accumulator)
        else:
            pass
        return accumulator