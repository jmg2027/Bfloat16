from .commonimport import *
from test.utils import *
from test.test_class import *

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

    test_operation = 'mul'
    _INPUT_NUM = 2
    _TEST_SET_STRUCTURE = '[(num1, num2), (num3, num4), ...]'
    mod_list = {0: (bf16, bf16), 1: (fp32, fp32)}

    def __init__(self, mod, test_set = test_set) -> None:
        super().__init__(mod, test_set, self.test_operation)

    # this method should be defined in subclasses
    def set_ftype(self, mod):
        ftype = self.mod_list[mod][0]
        return ftype

    # this method should be defined in subclasses
    def _check_test_set(self, test_set: list) -> bool:
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
