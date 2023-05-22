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

def test_mul(num1: float, num2: float):
    a = convert_to_bf16(num1)
    b = convert_to_bf16(num2)
    bf16_res = a * b

    tfa = convert_to_tfbf16(num1)
    tfb = convert_to_tfbf16(num2)
    tfbf16_res = tfa * tfb
    
    check_float_equal(bf16_res, tfbf16_res)
    if check_float_equal:
        print(f'PASSED {num1} * {num2}')
    else:
        print(f'FAILED {num1} * {num2}, bf16: {bf16_res}, tfbf16: {tfbf16_res}')
    return

def test():
    for a, b in test_set:
        test_mul(a, b)
    return
