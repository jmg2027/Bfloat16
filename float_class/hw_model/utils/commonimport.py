#from bf16 import bf16, fp32, FloatBase, bit, sbit, ubit, fp32_obj
#from float_class import * # why circular import is not a problem here?
#import importlib
from float_class.floatint import FloatBaseInt as fpint
from float_class.floatint import bf16_config, fp32_config
from float_class.bitstring import BitString as bit
from float_class.bitstring import SignedBitString as sbit
from float_class.bitstring import UnsignedBitString as ubit
from typing import Generic, TypeVar, Tuple
from ..utils import utils as hwutil
from ..utils.utils import isnan, isinf, iszero

#FloatBaseT = TypeVar('FloatBaseT', bound='FloatBase')
FloatBaseT = TypeVar('FloatBaseT')

#
FPBitT = Tuple[bit, bit, bit]