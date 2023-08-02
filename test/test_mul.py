from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

test_set = [
		(1.0, 2.0),
		(2, 12),
		(-2, 12),
		(2, -12),
		(-2, -12),
		(123.124, 381.58),
		(123.124, -381.58),
		(-123.124, 381.58),
		(-123.124, -381.58),
		(0.00076, 0.3256),
		(-0.00076, 0.3256),
		(0.00076, -0.3256),
		(-0.00076, -0.3256),
		(111111111.111111111, 999999999999.999999999999),
		(0, 0)
]


def test_mul_bf16(num1: float, num2: float):
    a = convert_to_bf16(num1)
    b = convert_to_bf16(num2)
    bf16_res = a * b

    tfa = convert_to_tfbf16(num1)
    tfb = convert_to_tfbf16(num2)
    tfbf16_res = tfa * tfb
    
    check_float_equal(bf16_res, tfbf16_res)
    if check_float_equal:
        test_res_str = f'PASSED {num1} * {num2}'
    else:
        test_res_str = f'FAILED {num1} * {num2}, bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def test_mul(num1: float, num2: float):
    a = convert_to_fp32(num1)
    b = convert_to_fp32(num2)
    fp32_res = a * b

    tfa = convert_to_tffp32(num1)
    tfb = convert_to_tffp32(num2)
    tffp32_res = tfa * tfb
    
    check_float_equal(fp32_res, tffp32_res)
    if check_float_equal:
#        test_res_str = f'PASSED {num1} * {num2}, res: {float(fp32_res)}'
        test_res_str = f'PASSED {num1} * {num2}, res: {fp32_res}'
    else:
        test_res_str = f'FAILED {num1} * {num2}, bf16: {fp32_res}, tfbf16: {tffp32_res}'
    print(test_res_str)
    return a, b, fp32_res, test_res_str

def rand_test(times: int):
    test_list = []
    fail_list = []
    for i in range(times):
        a, b, fp32_res, test_res_str = test_mul(float(random_bf16()), float(random_bf16()))
        test_list.append([a, b, fp32_res])
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return test_list

def test():
    fail_list = []
    for a, b in test_set:
        test_res_str = test_mul(a, b)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
