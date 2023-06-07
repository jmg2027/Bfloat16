from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

bf16_mant_max = float(bf16(0, 0, 127))

test_set = [
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
[5.4375, -18.25, 1.078125, 0.166015625, -0.30859375, 8.8125, -1.2734375, -5.03125, -22.625, -24.625, 1.8203125, 0.220703125, 0.158203125, 2.5625, 10.5, -0.1015625, -0.1982421875, 0.2451171875, 6.4375, 13.875, -0.216796875, -0.1357421875, -0.06982421875, -0.859375, 0.37109375, 7.625, 1.9765625, 0.3125, -0.3046875, -0.11474609375, 6.21875, -7.6875, 1.328125, 0.12890625, -1.78125, 0.95703125, -7.6875, -2.578125, -4.84375, 11.375, 0.07958984375, 1.71875, 4.59375, -0.345703125, -1.5625, 0.07275390625, 10.4375, -2.65625, -2.890625, 4.375, 3.78125, -22.375, -0.7109375, 3.40625, -0.150390625, 0.12109375, -14.0, -0.48828125, -3.90625, -11.3125, 0.2890625, -0.318359375, -0.08642578125, -0.08984375]
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
        for i in range(64):
            #vector.append(float(random_bf16()))
            vector.append(float(random_bf16_range(-4, 4)))
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
