from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

test_set = [
		(1.0, 2.0, 4.0),
		(1.0, 2.0, 3.0),
		(100.0, 2.0, 3.0),
		(10000.0, 2.0, 3.0),
		(1.0, 2.0, 30000.0),
		(1000000000.0, 200.0, 3.0),
		(0.000000001, 0.11111112, 3000000000000000.0),
		(-1.0, 2.0, 4.0),
		(-1.0, 2.0, 3.0),
		(1.0, -2.0, 3.0),
		(1.0, 2.0, -3.0),
		(-1.0, -2.0, -3.0),
		(-100.0, 2.0, 3.0),
		(10000.0, -2.0, 3.0)
]

def rand_fma_test(times: int):
    for i in range(times):
        test_fma(float(random_bf16()), float(random_bf16()), float(random_bf16()))
    return

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
        print(f'PASSED FMA({num1}, {num2}, {num3})')
    else:
        print(f'FAILED FMA({num1}, {num2}, {num3}), bf16: {bf16_res}, tfbf16: {tfbf16_res}')
    return

def test():
    for a, b, c in test_set:
        test_fma(a, b, c)
    return
