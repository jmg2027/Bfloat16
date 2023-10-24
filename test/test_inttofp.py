from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

test_set = [
    1,
    2,
    -1,
    -2,
    100,
    1000,
    -100,
    -1000,
    32709,
    -32709,
    0,
    1000000
]


def test_inttofp(num: int):
    bf16_res = bf16.inttofp(num)

    tfbf16_res = convert_int_to_tfbf16(num)
    
    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED {num}'
    else:
        test_res_str = f'FAILED {num}.inttofp(), lib: {bf16_res}, tf: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    fail_list = []
    max_int = 2 ** 31
    for i in range(times):
        test_res_str = test_inttofp(random.randint(-max_int, max_int - 1))
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for a in test_set:
        test_res_str = test_inttofp(a)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
