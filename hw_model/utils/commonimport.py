#from bf16 import bf16, fp32, FloatBase, bit, sbit, ubit, fp32_obj
from float_class import * # why circular import is not a problem here?
from typing import Generic, TypeVar
from ..utils import utils as hwutil

FloatBaseT = TypeVar('FloatBaseT', bound='FloatBase')