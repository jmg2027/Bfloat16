from bf16 import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

test_set = [
    {'input': [
        1.0,
        -1.2,
        4.0,
        5.0,
        -10.0,
        -20.0,
        30.0,
        -100.0,
            ],
    'weight': [
        2.0,
        -4.2,
        7.0,
        -9.0,
        10.0,
        -20.0,
        30.567,
        -400.6,
            ]},
    {'input': [
        1.0,
        -1.2,
        4.0,
        5.0,
        -10.0,
        -20.0,
        30.0,
        -100.0,
            ],
    'weight': [
        2.0,
        -4.2,
        7.0,
        -9.0,
        10.0,
        -20.0,
        30.567,
        -400.6,
            ]}
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
#
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

def tf_inner_product(vec1, vec2):
    return tf.tensordot(vec1, vec2, 1)

def test_summation(input_vector, weight_vector):
    a = convert_element_to_bf16_array(input_vector)
    b = convert_element_to_bf16_array(weight_vector)
    bf16_res = bf16.summation(a, b)
    tfa = convert_element_to_tfbf16_array(input_vector)
    tfb = convert_element_to_tfbf16_array(weight_vector)
    tfbf16_res = tf_inner_product(tfa, tfb)
    
    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED SUM({input_vector}, {weight_vector})'
    else:
        test_res_str = f'FAILED SUM({input_vector}, {weight_vector}), bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    # Generate 8 input random bf16
    input_vector = list()
    for i in range(8):
        input_vector.append(float(random_bf16()))
    # Generate 8 weight random bf16
    weight_vector = list()
    for i in range(8):
        weight_vector.append(float(random_bf16()))

    fail_list = []
    for i in range(times):
        test_res_str = test_summation(input_vector, weight_vector)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for vector_set in test_set:
        test_res_str = test_summation(vector_set['input'], vector_set['weight'])
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
