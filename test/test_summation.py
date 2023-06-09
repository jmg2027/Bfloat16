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
#FAILED SUM([-0.01708984375, -0.0498046875, -3.84375, -4.28125, 936.0, -0.00750732421875, 0.89453125, 96.5, -422.0, 0.00677490234375, -1832.0, -0.134765625, 0.0791015625, 0.828125, -1.0625, 222.0, 0.98046875, -47.0, -1032.0, -0.005767822265625, -0.15234375, -0.5390625, 0.039794921875, 510.0, -336.0, -0.08740234375, -61.75, 0.125, -146.0, -0.0196533203125, -5.5, -14.75, 3.953125, -143.0, 43.75, 0.045654296875, -23.25, -0.015380859375, 0.00567626953125, -7.0625, -0.0159912109375, 27.5, -6.5625, 11.9375, -1272.0, -4.9375, -0.0299072265625, -0.00174713134765625, 0.2392578125, -992.0, -0.00341796875, -284.0, -0.03076171875, 154.0, -1168.0, 227.0, 246.0, -152.0, 0.01513671875, -14.9375, -1656.0, 0.041015625, 334.0, -460.0]), bf16: Bfloat16(-3184.0, sign = 1, exponent=11, mantissa=71), tfbf16: -7264
#FAILED SUM([-4.96875, -0.263671875, -0.1064453125, 1.34375]), bf16: Bfloat16(-2.0, sign = 1, exponent=1, mantissa=0), tfbf16: -4
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
