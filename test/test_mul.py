from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
from test.utils import *

from test.test_class import TestOperationBase

'''
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

    ftype = fp32
    _INPUT_NUM = 2
    _TEST_SET_STRUCTURE = '[(num1, num2), (num3, num4), ...]'

    def __init__(self, ftype = ftype, test_set = test_set):
        self.set(ftype, test_set)

    def set(self, ftype, test_set):
        self.test_set, self.ftype = self.set_test_set(test_set), self.set_ftype(ftype)
        self._f_ops = self._set_f_ops(ftype)
        self.operation = self._set_operation(self._f_ops)
        self.tf_operation = self._set_operation(self._TF_OPS)

    def set_test_set(self, test_set):
        if self._check_test_set(test_set):
            return test_set
        else:
            raise TypeError(f'{self.__name__} test_set structure should be:\n {self._TEST_SET_STRUCTURE}')

    def set_ftype(self, ftype):
        if (ftype == fp32) | (ftype == bf16):
            return ftype
        else:
            raise TypeError('Ftype should be bf16 or fp32 class')

    def _set_f_ops(self, ftype):
        return \
        {
            'mul': getattr(ftype, '__mul__', None),
            'add': getattr(ftype, '__add__', None),
            'fma': getattr(ftype, 'fma', None),
            'summation': getattr(ftype, 'summation', None),
        }

    def _set_operation(self, operation_dict: dict):
        operation = operation_dict.get(self._OP, None)
        if operation is None:
            raise ValueError(f"Unsupported operation {self._OP}")
        return operation

    def perform_operation(self, operand):
        return self.operation(*operand)

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
        if len(v) != self._INPUT_NUM:
            res = False
        return res

    def test_body(self, input: tuple):
        # input: hex(int), float, bf16, fp32
        operand = tuple(map(cast_float, input, [self.ftype]*self._INPUT_NUM))
        res = self.operation(*operand)

        tf_operand = tuple(map(conv_to_tf_dtype, input, [self.ftype]*self._INPUT_NUM))
        tfres = self.tf_operation(*tf_operand)
        
        check_float_equal(res, tfres)
        if check_float_equal:
            test_res_str = f'PASSED {self._OP}{input}, res: {res}'
        else:
            test_res_str = f'FAILED {self._OP}{input}, bf16: {res}, tfbf16: {tfres}'
        print(test_res_str)
        test_ret = list(i for i in input)
        test_ret.append(res)
        test_ret.append(test_res_str)
        return input, res, test_res_str
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

'''

class TestMul(TestOperationBase):

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

    ftype = fp32
    _INPUT_NUM = 2
    _TEST_SET_STRUCTURE = '[(num1, num2), (num3, num4), ...]'

    def __init__(self, ftype: Type[FloatType] = fp32, test_set = test_set):
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
