from bf16.bf16 import Bfloat16 as bf16
from test.utils import *
#FAILED 1.3660947373317356e-17 + 7.081195439045951e-19, bf16: Bfloat16(1.431146867680866e-17, sign = 0, exponent=-56, mantissa=4), tfbf16: 1.44199e-17
#FAILED -2.337619662284851e-07 + -5.617039278149605e-09, bf16: Bfloat16(-2.384185791015625e-07, sign = 1, exponent=-22, mantissa=0), tfbf16: -2.40281e-07
#FAILED -4.5609180811904755e-36 + -1.8883141251609227e-34, bf16: Bfloat16(-1.925929944387236e-34, sign = 1, exponent=-112, mantissa=0), tfbf16: -1.94098e-34
#FAILED 8.04638123325432e-28 + 1.9524307404220042e-29, bf16: Bfloat16(8.204153414298523e-28, sign = 0, exponent=-90, mantissa=2), tfbf16: 8.26726e-28
#FAILED 17729624997888.0 + 264982302294016.0, bf16: Bfloat16(281474976710656.0, sign = 0, exponent=48, mantissa=0), tfbf16: 2.83674e+14
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


def test_add(num1: float, num2: float):
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
