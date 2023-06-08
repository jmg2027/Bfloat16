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
#FAILED SUM([0.435546875, -0.1552734375, -1.9921875, -6.25]), bf16: Bfloat16(-7.9375, sign = 1, exponent=2, mantissa=126), tfbf16: -7.96875
#FAILED SUM([-22.125, -3.359375, 10.75, 1.4453125]), bf16: Bfloat16(-13.25, sign = 1, exponent=3, mantissa=84), tfbf16: -13.3125
#FAILED SUM([0.2177734375, -7.71875, -0.25390625, -11.8125]), bf16: Bfloat16(-19.5, sign = 1, exponent=4, mantissa=28), tfbf16: -19.625
#FAILED SUM([0.11669921875, 0.63671875, 0.0947265625, 0.6015625]), bf16: Bfloat16(1.4453125, sign = 0, exponent=0, mantissa=57), tfbf16: 1.45312
#FAILED SUM([-0.11767578125, -0.50390625, -7.5, -11.375]), bf16: Bfloat16(-19.375, sign = 1, exponent=4, mantissa=27), tfbf16: -19.5
#FAILED SUM([19.125, 0.72265625, 7.1875, -0.1845703125]), bf16: Bfloat16(26.75, sign = 0, exponent=4, mantissa=86), tfbf16: 26.875
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
            vector.append(float(random_bf16_range(-50, 50)))
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
