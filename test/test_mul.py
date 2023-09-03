from bf16 import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self


class TestMul(TestOperationBase):

    # Test set
    test_set :List[Tuple[TestInputT, TestInputT]]= [
            (1.0, 2.0),
            (2.0, 12.0),
            (-2.0, 12.0),
            (2.0, -12.0),
            (-2.0, -12.0),
            (123.124, 381.58),
            (123.124, -381.58),
            (-123.124, 381.58),
            (-123.124, -381.58),
            (0.00076, 0.3256),
            (-0.00076, 0.3256),
            (0.00076, -0.3256),
            (-0.00076, -0.3256),
            (111111111.111111111, 999999999999.999999999999),
            (0, 0),
            (0x80000000, 0x10000000),
            (bf16(1, 80, 30), bf16(0, -30, 55)),
            (fp32(1, 90, 800000), fp32(0, -1, 500000)),
            (bf16(1, 80, 30), fp32(0, -1, 500000))
    ]

    ftype: Type = fp32
    _INPUT_NUM = 2
    _TEST_SET_STRUCTURE = '[(num1, num2), (num3, num4), ...]'

    def __init__(self, ftype: Type[FloatBaseT] = fp32, test_set = test_set):
        super().__init__(ftype, test_set, 'mul')

    # this method should be defined in subclasses
    def _check_test_set(self, test_set: list):
        res = True
        # check structure
        if not isinstance(test_set, list):
            res = False
        for v in test_set:
            # check structure
            if not isinstance(v, tuple):
                res = False
            # number of input
            if not self._check_input_num:
                res = False
        # input type check is handled in cast_float function in util
        return res
