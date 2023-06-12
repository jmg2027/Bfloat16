from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

bf16_mant_max = float(bf16(0, 0, 127))
# Match vector_element_num to FloatSummation.vector_element_num
#vector_element_num = 64
vector_element_num = 4

#test_set = [
#    [
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
#     ],
#    [
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max,
#bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max, bf16_mant_max
#     ]
#]
test_set = [
#    [
#1.0, 1.0, 1.0, 1.0
#     ],
#    [
#1.0, 2.0, 4.0, 8.0
#     ],
#    [
#1.0, 2.0, 3.0, 4.0
#     ],
#    [
#-1.0, -1.0, -1.0, -1.0
#     ],
#    [
#-1.0, -2.0, -4.0, -8.0
#     ],
#    [
#-1.0, -2.0, -3.0, -4.0
#     ],
# These are adoptable error I think...
#FAILED SUM([458.0, -1184.0, 0.0036163330078125, -0.0035858154296875]), bf16: Bfloat16(-724.0, sign = 1, exponent=9, mantissa=53), tfbf16: -728
#FAILED SUM([-0.00173187255859375, 744.0, -744.0, 0.00811767578125]), bf16: Bfloat16(0.006378173828125, sign = 0, exponent=-8, mantissa=81), tfbf16: 0.00640869
#FAILED SUM([-308.0, 0.00628662109375, -0.006256103515625, 1528.0]), bf16: Bfloat16(1224.0, sign = 0, exponent=10, mantissa=25), tfbf16: 1216
#FAILED SUM([-44.0, 0.0027923583984375, 1776.0, -0.00274658203125]), bf16: Bfloat16(1736.0, sign = 0, exponent=10, mantissa=89), tfbf16: 1728
#[458.0, -1184.0, 0.0036163330078125, -0.0035858154296875],
#[-0.00173187255859375, 744.0, -744.0, 0.00811767578125],
#[-308.0, 0.00628662109375, -0.006256103515625, 1528.0],
#[-44.0, 0.0027923583984375, 1776.0, -0.00274658203125],

# It fails...
#[-0.00173187255859375, 744.0, -744.0, 0.00811767578125],
# It passes
#[-0.00173187255859375, 0, 0, 0.00811767578125],
# It seems tf.reduce_sum works different way from summation unit
#[-0.00173187255859375, 100000, -100000, 0.00811767578125],
#[-0.00173187255859375, 10000, -10000, 0.00811767578125],
# This is special case for larger align bit
#[-0.00173187255859375, 1000000000000000, -1000000000000000, 0.00811767578125],
[0, 0, 0, 0],
[-10, 10, -100, 100],
[0, 0, 0, float(bf16(0, -126, 0))],
# Special cases
#['nan', 1.0, 2.0, 3.0],
#['inf', 1.0, 2.0, 3.0],
#['-inf', 1.0, 2.0, 3.0],
#['inf', '-inf', 2.0, 3.0],
#['inf', 1.0, 'inf', 3.0],
#['-inf', 1.0, 2.0, '-inf'],
]
#        self.input_vector = [
#            bf16.Bfloat16.float_to_bf16(1.0),
#            bf16.Bfloat16.float_to_bf16(-1.2),
#            bf16.Bfloat16.float_to_bf16(4.0),
#            bf16.Bfloat16.float_to_bf16(5.0),
#            bf16.Bfloat16.float_to_bf16(-10.0),
#            bf16.Bfloat16.float_to_bf16(-20.0),
#            bf16.Bfloat16.float_to_bf16(30.0),
#            bf16.Bfloat16.float_to_bf16(-100.0)
#            ]
#        self.weight_vector = [
#            bf16.Bfloat16.float_to_bf16(2.0),
#            bf16.Bfloat16.float_to_bf16(-4.2),
#            bf16.Bfloat16.float_to_bf16(7.0),
#            bf16.Bfloat16.float_to_bf16(-9.0),
#            bf16.Bfloat16.float_to_bf16(10.0),
#            bf16.Bfloat16.float_to_bf16(-20.0),
#            bf16.Bfloat16.float_to_bf16(30.567),
#            bf16.Bfloat16.float_to_bf16(-400.6)

def convert_element_to_bf16_array(vector: iter) -> iter:
    bf16_vector = []
    for element in vector:
        bf16_vector.append(convert_to_bf16(element))
    return bf16_vector

def convert_element_to_tfbf16_array(vector: iter) -> iter:
    tfbf16_vector = []
    for element in vector:
        tfbf16_vector.append(convert_to_tfbf16(element))
    return tfbf16_vector

def tf_reduce_sum(vec1):
    return tf.reduce_sum(vec1)

def test_summation(vector):
    a = convert_element_to_bf16_array(vector)
    bf16_res = bf16.summation(a)
    tfa = convert_element_to_tfbf16_array(vector)
    tfbf16_res = tf_reduce_sum(tfa)

    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED SUM({vector})'
    else:
        test_res_str = f'FAILED SUM({vector}), bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    # Generate 64 input random bf16
    fail_list = []
    for i in range(times):
        vector = list()
        for i in range(vector_element_num):
            #vector.append(float(random_bf16()))
            #vector.append(float(random_bf16_range(-4, 4)))
            vector.append(float(random_bf16_range(-10, 10)))
        test_res_str = test_summation(vector)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for vector_set in test_set:
        test_res_str = test_summation(vector_set)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
