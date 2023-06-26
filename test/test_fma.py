from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

# FAILED FMA(-30408704.0, 3.844499588012695e-06, -0.310546875), bf16: Bfloat16(-117.0, sign = 1, exponent=6, mantissa=106), tfbf16: -117.5
# FAILED FMA(474.0, -44544.0, -458752.0), bf16: Bfloat16(-21626880.0, sign = 1, exponent=24, mantissa=37), tfbf16: -2.14958e+07

test_set = [
		(1.0, 2.0, 4.0),
		(1.0, 2.0, 3.0),
		(100.0, 2.0, 3.0),
		(10000.0, 2.0, 3.0),
		(1.0, 2.0, 30000.0),
		(1000000000.0, 200.0, 3.0),
# debug: bf16 mantissa is 1 less than tf.bfloat16
# debug: 3000000000000000.0 mantissa in fp32 is 0101010_1000011110111111
# debug: in this case, isn't it not to round up case? It seems tf round this up
		(0.000000001, 0.11111112, 3000000000000000.0),
		(-1.0, 2.0, 4.0),
		(-1.0, 2.0, 3.0),
		(1.0, -2.0, 3.0),
		(1.0, 2.0, -3.0),
		(-1.0, -2.0, -3.0),
		(-100.0, 2.0, 3.0),
		(10000.0, -2.0, 3.0),
        # Special cases
        (0, 2.8, 4.5),
        (0, 0, 4.5),
        (4.5, 2.8, 0),
        (-5.4752209710517974e-18, 5.5340232221128655e+19, 0.00135040283203125)
]


def test_fma(num1: float, num2: float, num3: float):
    a = convert_to_bf16(num1)
    b = convert_to_bf16(num2)
    c = convert_to_bf16(num3)
    bf16_res = bf16.fma(a, b, c)

    tfa = convert_to_tfbf16(num1)
    tfb = convert_to_tfbf16(num2)
    tfc = convert_to_tfbf16(num3)
    tfbf16_res = tfa * tfb + tfc
    
    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED FMA({num1}, {num2}, {num3})'
    else:
        test_res_str = f'FAILED FMA({num1}, {num2}, {num3}), bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    fail_list = []
    for i in range(times):
        test_res_str = test_fma(float(random_bf16()), float(random_bf16()), float(random_bf16()))
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for a, b, c in test_set:
        test_res_str = test_fma(a, b, c)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
