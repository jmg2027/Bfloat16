from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

test_set = [
#		(2, 12),
#		(-2, 12),
#		(2, -12),
#		(-2, -12),
#		(25.0924, 24.8076),
#		(-25.0924, 24.8076),
#		(25.0924, -24.8076),
#		(-25.0924, -24.8076),
#		(123.124, 381.58),
#		(123.124, -381.58),
#		(-123.124, 381.58),
#		(-123.124, -381.58),
#		(0.00076, 0.3256),
#		(-0.00076, 0.3256),
#		(0.00076, -0.3256),
#		(-0.00076, -0.3256),
#		(111111111.111111111, 999999999999.999999999999),
#    # debug
#		(111111111.111111111, -999999999999.999999999999),
#    # Corner cases
#		(0, 0),
#		(10.10293, -0.0000000000000000000000000000000000000001),
#		(10.10293, 0.0000000000000000000000000000000000000001),
#   # 100931731456
#   # 1000727379968
#		(101029300000, 999999999999),
#		(101029300000, -999999999999),
    # zero case
        (2, -2),
#        (-2, 2),
#    # invert case
#        (-4, 2),
#        (2, -4)
]

def rand_add_test(times: int):
    for i in range(times):
        test_add(float(random_bf16()), float(random_bf16()))
    return

def test_add(num1: float, num2: float):
    a = convert_to_bf16(num1)
    b = convert_to_bf16(num2)
    bf16_res = a + b

    tfa = convert_to_tfbf16(num1)
    tfb = convert_to_tfbf16(num2)
    tfbf16_res = tfa + tfb
    
    if check_float_equal(bf16_res, tfbf16_res):
        print(f'PASSED {num1} + {num2}')
    else:
        print(f'FAILED {num1} + {num2}, bf16: {bf16_res}, tfbf16: {tfbf16_res}')
    return

def test():
    for a, b in test_set:
        test_add(a, b)
    return
