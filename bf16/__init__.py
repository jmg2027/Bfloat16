from .floatclass import Bfloat16 as bf16
from .floatclass import Float32 as fp32
from .floatclass import FloatBaseT

from .floatclass import tfbfloat16
from .floatclass import tffloat32
from .floatclass import TfFloatT

from .bitstring import BitString as bit
from .bitstring import SignedBitString as sbit
from .bitstring import UnsignedBitString as ubit

from typing import(
    Union,
    Type,
    TypeVar,
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional,
    Literal,
    Generic,
)

fp32_obj = fp32(0, 0, 0)
bf16_obj = bf16(0, 0, 0)
bit_zero = bit(1, '0')
bit_one  = bit(1, '1')

__all__ = ['bf16', 'fp32', 'bit', 'sbit', 'ubit', 'FloatBaseT', 'fp32_obj', 'bf16_obj', 'bit_zero', 'bit_one', 'tfbfloat16', 'tffloat32', 'TfFloatT']