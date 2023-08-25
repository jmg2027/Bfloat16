from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
from test.utils import *

from test.test_class import TestAbsClass

class TestMul(TestAbsClass):

    # Test set
    test_set = [
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

    input_num = 2
    test_set_structure = '[(num1, num2), (num3, num4), ...]'
    ftype = fp32
    operation = ftype.__mul__

    def __init__(self, test_set = test_set, ftype = ftype):
        self.set(test_set, ftype)

    def set(self, test_set, ftype):
        self.test_set, self.ftype = self.set_test_set(test_set), self.set_ftype(ftype)

    def set_test_set(self, test_set):
        if self._check_test_set(test_set):
            return test_set
        else:
            raise TypeError(f'{self.__name__} test_set structure should be:\n {self.test_set_structure}')

    def set_ftype(self, ftype):
        if (ftype == fp32) | (ftype == bf16):
            return ftype
        else:
            raise TypeError('Ftype should be bf16 or fp32 class')

    # this method should be defined in subclasses
    def _check_test_set(self, test_set: iter):
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
    
    def _check_input_num(self, v: list):
        res = True
        if len(v) != self.input_num:
            res = False
        return res

    def test_body(self, input: tuple):
        # input: hex(int), float, bf16, fp32
        operand = tuple(map(cast_float, input))
        print(*operand)
        
        print(f'{fp32.__mul__}({operand})')
        print(f'{self.ftype.__mul__}({operand})')
        print(f'{self.operation}({operand})')
        print(fp32.__mul__(fp32(0,0,0), fp32(0,1,0)))
        print(self.ftype.__mul__(fp32(0,0,0), fp32(0,1,0)))
        print(self.operation(fp32(0,1,0)))
        res = self.operation(fp32(0,0,0), fp32(0,1,0))
        res = self.operation(*operand)

        if self.ftype == bf16:
            tf_operand = tuple(map(convert_to_tfbf16, num))
            tfres = self.operation(tf_operand)
        elif self.ftype == fp32:
            tf_operand = tuple(map(convert_to_tffp32, num))
            tfres = self.operation(tf_operand)
        else: #is this necessary?
            raise TypeError('not supported test float type')
        
        check_float_equal(res, tfres)
        if check_float_equal:
            #test_res_str = f'PASSED {num1} * {num2}, res: {res}'
            test_res_str = f'PASSED {repr(self.operation)}{num}, res: {res}'
        else:
            #test_res_str = f'FAILED {num1} * {num2}, bf16: {res}, tfbf16: {tfres}'
            test_res_str = f'FAILED {repr(self.operation)}{num}, bf16: {res}, tfbf16: {tfres}'
        print(test_res_str)
        test_ret = list(i for i in num)
        test_ret.append(res)
        test_ret.append(test_res_str)
        return num, res, test_res_str
        #return test_ret

    def rand_test(self, times: int):
        test_list = []
        fail_list = []
        for i in range(times):
            a, b, fp32_res, test_res_str = self.test_body(float(random_bf16()), float(random_bf16()))
            test_list.append([a, b, fp32_res])
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return test_list

    def test(self):
        fail_list = []
        for v in self.test_set:
            test_res_str = self.test_body(v)
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return

        # test_body(a,b)
        # test_body(a,b,c)
