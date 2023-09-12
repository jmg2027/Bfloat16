from .floatclass import Bfloat16 as bf16
from .floatclass import Float32 as fp32
from .floatclass import FloatBase

from .floatclass import tfbfloat16
from .floatclass import tffloat32

from .floatclass import FloatTypeError
from .floatclass import FloatValueError

from .floatclass import bf16_to_fp32
from .floatclass import fp32_to_bf16

from .floatint import FloatBaseInt as fpint
from .floatint import bf16_config
from .floatint import fp32_config
from .floatint import bf16_int
from .floatint import fp32_int


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

__all__ = ['bf16', 'fp32', 'bit', 'sbit', 'ubit', 'fp32_obj', 
           'FloatBase', 'fpint',
           'bf16_config', 'fp32_config', 'bf16_int', 'fp32_int',
           'bf16_obj', 'bit_zero', 'bit_one', 'tfbfloat16', 'tffloat32',
           'FloatTypeError', 'FloatValueError', 'bf16_to_fp32', 'fp32_to_bf16']
