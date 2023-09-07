#from bf16 import bf16, fp32, FloatBase, bit, sbit, ubit, fp32_obj
#from float_class import * # why circular import is not a problem here?
import float_class as fc
from typing import Generic, TypeVar
from ..utils import utils as hwutil

bf16 = fc.bf16
fp32 = fc.fp32
bit = fc.bit
sbit = fc.sbit
ubit = fc.ubit
fp32_obj = fc.fp32_obj
bf16_obj = fc.bf16_obj

FloatBaseT = TypeVar('FloatBaseT', bound='float_class.FloatBase')