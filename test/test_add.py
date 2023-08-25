from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

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
        (float(bf16(0, 0, bf16.mant_max)), float(bf16(0, 0, bf16.mant_max))),
    # When two add of mantissa is 0111_1111_1111
        (float(bf16(0, 8, bf16.mant_max)), float(bf16(0, 0, bf16.mant_max)))
]

# FIX: Need to handle proper test method for bf16/fp32
def test_add_bf16(num1: float, num2: float):
    a = convert_to_bf16(num1)
    b = convert_to_bf16(num2)
    bf16_res = a + b

    tfa = convert_to_tfbf16(num1)
    tfb = convert_to_tfbf16(num2)
    tfbf16_res = tfa + tfb
    
    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED {num1} + {num2}'
    else:
        test_res_str = f'FAILED {num1} + {num2}, bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def test_add(num1: float, num2: float):
    a = convert_to_fp32(num1)
    b = convert_to_fp32(num2)
    fp32_res = a + b

    tfa = convert_to_tffp32(num1)
    tfb = convert_to_tffp32(num2)
    tffp32_res = tfa + tfb
    
    if check_float_equal(fp32_res, tffp32_res):
        test_res_str = f'PASSED {num1} + {num2}, res: {float(fp32_res)}'
    else:
        test_res_str = f'FAILED {num1} + {num2}, bf16: {fp32_res}, tfbf16: {tffp32_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    fail_list = []
    for i in range(times):
        test_res_str = test_add(float(random_bf16()), float(random_bf16()))
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for a, b in test_set:
        test_res_str = test_add(a, b)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
