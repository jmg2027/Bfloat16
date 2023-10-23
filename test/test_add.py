from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

class TestAdd(TestOperationBase):
    test_set = [
            (2, 12),
            (-2, 12),
            (2, -12),
            (-2, -12),
            (25.0924, 24.8076),
            (-25.0924, 24.8076),
            (25.0924, -24.8076),
        (-25.0924, -24.8076),
            (4.5, 6),
            (-4.5, 6),
            (4.5, -6),
            (-4.5, -6),
            (123.124, 381.58),
            (123.124, -381.58),
            (-123.124, 381.58),
            (-123.124, -381.58),
            (0.00076, 0.3256),
            (-0.00076, 0.3256),
            (0.00076, -0.3256),
            (-0.00076, -0.3256),
            # exponent is far larger that does not affect to other addend
            (111111111.111111111, 999999999999.999999999999),
            (111111111.111111111, -999999999999.999999999999),
        # Corner cases
            (0, 0),
            (10.10293, -0.0000000000000000000000000000000000000001),
            (10.10293, 0.0000000000000000000000000000000000000001),
    # 100931731456
    # 1000727379968
            (101029300000, 999999999999),
            (101029300000, -999999999999),
        # zero case
            (2, -2),
            (-2, 2),
        # invert case
            (-4, 2),
            (2, -4),
        # When two add of mantissa is 1111_1111_1110
            (float(bf16(0, 0, bf16_config.mant_max)), float(bf16(0, 0, bf16_config.mant_max))),
        # When two add of mantissa is 0111_1111_1111
            (float(bf16(0, 8, bf16_config.mant_max)), float(bf16(0, 0, bf16_config.mant_max))),
    ]

    test_operation = 'add'
    _INPUT_NUM = 2
    _TEST_SET_STRUCTURE = '[(num1, num2), (num3, num4), ...]'
    mod_list = {0: (bf16, bf16), 1: (fp32, fp32)}

    def __init__(self, mod, test_set = test_set) -> None:
        super().__init__(mod, test_set, self.test_operation)
        input_ftype = self.set_input_ftype(mod)

    # this method should be defined in subclasses
    def set_ftype(self, mod):
        ftype = self.mod_list[mod][1]
        return ftype

    # this method should be defined in subclasses
    def set_input_ftype(self, mod):
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