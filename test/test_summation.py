from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

bf16_mant_max = float(bf16(0, 0, 127))
# Match vector_element_num to FloatSummation.vector_element_num
#vector_element_num = 64
vector_element_num = 32

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
#    [[
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
#     ]],
# These are adoptable error I think...
#FAILED SUM([458.0, -1184.0, 0.0036163330078125, -0.0035858154296875]), bf16: Bfloat16(-724.0, sign = 1, exponent=9, mantissa=53), tfbf16: -728
#FAILED SUM([-0.00173187255859375, 744.0, -744.0, 0.00811767578125]), bf16: Bfloat16(0.006378173828125, sign = 0, exponent=-8, mantissa=81), tfbf16: 0.00640869
#FAILED SUM([-308.0, 0.00628662109375, -0.006256103515625, 1528.0]), bf16: Bfloat16(1224.0, sign = 0, exponent=10, mantissa=25), tfbf16: 1216
#FAILED SUM([-44.0, 0.0027923583984375, 1776.0, -0.00274658203125]), bf16: Bfloat16(1736.0, sign = 0, exponent=10, mantissa=89), tfbf16: 1728
#[[458.0, -1184.0, 0.0036163330078125, -0.0035858154296875],
#[-0.00173187255859375, 744.0, -744.0, 0.00811767578125],
#[-308.0, 0.00628662109375, -0.006256103515625, 1528.0],
#[-44.0, 0.0027923583984375, 1776.0, -0.00274658203125],]

# It fails...
#[-0.00173187255859375, 744.0, -744.0, 0.00811767578125],
# It passes
#[-0.00173187255859375, 0, 0, 0.00811767578125],
# It seems tf.reduce_sum works different way from summation unit
#[-0.00173187255859375, 100000, -100000, 0.00811767578125],
#[-0.00173187255859375, 10000, -10000, 0.00811767578125],
# This is special case for larger align bit
#[-0.00173187255859375, 1000000000000000, -1000000000000000, 0.00811767578125],
#[0, 0, 0, 0],
#[-10, 10, -100, 100],
#[0, 0, 0, float(bf16(0, -126, 0))],
## Special cases
#['nan', 1.0, 2.0, 3.0],
#['inf', 1.0, 2.0, 3.0],
#['-inf', 1.0, 2.0, 3.0],
#['inf', '-inf', 2.0, 3.0],
#['inf', 1.0, 'inf', 3.0],
#['-inf', 1.0, 2.0, '-inf'],
#FAILED SUM([1.2864312886424686e-20, 7.551635855174848e+19, 1.169720636290726e+38, 2.7249173913590775e+38]), bf16: Bfloat16(nan, sign = 0, exponent=128, mantissa=18), tfbf16: inf
#[1.2864312886424686e-20, 7.551635855174848e+19, 1.169720636290726e+38, 2.7249173913590775e+38],
#[float(bf16(0,bf16.exp_max, bf16.mant_max)), float(bf16(0,bf16.exp_max, bf16.mant_max)), float(bf16(0,bf16.exp_max, bf16.mant_max)), float(bf16(0,bf16.exp_max, bf16.mant_max))],
#[float(bf16(0,bf16.exp_max, 0)), float(bf16(0,bf16.exp_max, 0)), float(bf16(0,bf16.exp_max, 0)), float(bf16(0,bf16.exp_max, 0))],
#[float(bf16(0,bf16.exp_max-1, bf16.mant_max)), float(bf16(0,bf16.exp_max-1, bf16.mant_max)), float(bf16(0,bf16.exp_max-1, bf16.mant_max)), float(bf16(0,bf16.exp_max-1, bf16.mant_max))],
#[float(bf16(0,bf16.exp_max-1, bf16.mant_max))] * vector_element_num,
# Simple + cause wrong result:
# tf_vector_sum: 6.64614e+35
# bfloat16: 1.2773050271995676e+36
# python float: 1.2773861911840086e+36
# tf_reduce_sum: 1.2773050271995676e+36 (PASS)
#FAILED SUM([-2.4441935897665156e+20, 6.742775440216064e-07, -5.7931972727168054e-30, -2.1281525885478956e-32, 23776938950656.0, 3.3034284942917713e-20, 2.633107345841924e-35, -1.3658406274475593e-20, -2.6893296075324036e-20, 578813952.0, 14499809591296.0, -0.0458984375, -1.025638970498251e+22, -7.552869850948055e-26, 8.851494298807439e-20, 1.5795024613113804e+20, -16710107136.0, -7.34398467671166e+37, -0.00013446807861328125, 7.691059161069047e-19, -4.9611955367415196e-31, 5.662441253662109e-07, -6.810088283353266e-31, -2.9907629905160607e+37, 614180323328.0, 4.21614504913028e+36, -1.4653669987499396e-19, 54150947667968.0, 5.760204327437074e+33, -6.422418416702717e+24, -102804337197056.0, -4.579669976578771e-16, 106954752.0, 4.85722573273506e-17, -1032.0, 6.72084247699335e-25, -243712.0, 2.8719891998770765e+34, 6.273337765602162e-21, 2.55351295663786e-14, 1.1994118882018901e-22, -4.551514126383897e-35, 1.2048817095928447e-37, -3.893774191965349e-12, 0.0, 1.3624586956795388e+38, -7.229752424169529e-26, 6.938893903907228e-15, -4.188347653454561e+17, 3.4352836090169303e+28, -1.5784582449945876e+37, 2.244637310223896e-20, -4.4668785331246925e-37, 1.8821590109849165e-25, 76021760.0, -6.55527478390022e+34, -3.848728683538761e+23, 288072046477312.0, -2.0021496686510295e+37, 2576980377600.0, -0.0015106201171875, 374784.0, -3.8496412174785376e-28, 7559142440960.0]), bf16: Bfloat16(1.2773050271995676e+36, sign = 0, exponent=119, mantissa=118), tfbf16: 6.64614e+35
#[-2.4441935897665156e+20, 6.742775440216064e-07, -5.7931972727168054e-30, -2.1281525885478956e-32, 23776938950656.0, 3.3034284942917713e-20, 2.633107345841924e-35, -1.3658406274475593e-20, -2.6893296075324036e-20, 578813952.0, 14499809591296.0, -0.0458984375, -1.025638970498251e+22, -7.552869850948055e-26, 8.851494298807439e-20, 1.5795024613113804e+20, -16710107136.0, -7.34398467671166e+37, -0.00013446807861328125, 7.691059161069047e-19, -4.9611955367415196e-31, 5.662441253662109e-07, -6.810088283353266e-31, -2.9907629905160607e+37, 614180323328.0, 4.21614504913028e+36, -1.4653669987499396e-19, 54150947667968.0, 5.760204327437074e+33, -6.422418416702717e+24, -102804337197056.0, -4.579669976578771e-16, 106954752.0, 4.85722573273506e-17, -1032.0, 6.72084247699335e-25, -243712.0, 2.8719891998770765e+34, 6.273337765602162e-21, 2.55351295663786e-14, 1.1994118882018901e-22, -4.551514126383897e-35, 1.2048817095928447e-37, -3.893774191965349e-12, 0.0, 1.3624586956795388e+38, -7.229752424169529e-26, 6.938893903907228e-15, -4.188347653454561e+17, 3.4352836090169303e+28, -1.5784582449945876e+37, 2.244637310223896e-20, -4.4668785331246925e-37, 1.8821590109849165e-25, 76021760.0, -6.55527478390022e+34, -3.848728683538761e+23, 288072046477312.0, -2.0021496686510295e+37, 2576980377600.0, -0.0015106201171875, 374784.0, -3.8496412174785376e-28, 7559142440960.0],

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
#[['inf', 0, 0, 0], 
#['inf', 0, 0, 0], 
#['inf', 0, 0, 0], 
#['inf', 0, 0, 0]], 
#[['-inf', 0, 0, 0], 
#['-inf', 0, 0, 0], 
#['-inf', 0, 0, 0], 
#['-inf', 0, 0, 0]], 
#[['inf', 0, 0, 0], 
#['-inf', 0, 0, 0], 
#['inf', 0, 0, 0], 
#['-inf', 0, 0, 0]], 
[[1.0633823332454027e+37 for i in range(32)]],
]

def tf_reduce_sum(vec1):
    return tf.reduce_sum(vec1)

def tf_vector_sum(vec1):
    result = 0.0
    for i in vec1:
        result += i
    return result

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

def test_summation_bf16(vector_list):
    a = []
    tfa = []
    for vector in vector_list:
        a.append(convert_element_to_bf16_array(vector))
        tfa.append(convert_element_to_tfbf16_array(vector))
    bf16_res = bf16.summation(a)
    tfbf16_res = tf_reduce_sum(tfa)

    if check_float_equal(bf16_res, tfbf16_res):
        test_res_str = f'PASSED SUM({vector_list}), res: {bf16_res}'
    else:
        test_res_str = f'FAILED SUM({vector_list}), bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def convert_element_to_fp32_array(vector: iter) -> iter:
    fp32_vector = []
    for element in vector:
        fp32_vector.append(convert_to_fp32(element))
    return fp32_vector

def convert_element_to_tffp32_array(vector: iter) -> iter:
    tffp32_vector = []
    for element in vector:
        tffp32_vector.append(convert_to_tffp32(element))
    return tffp32_vector

def test_summation(vector_list):
    a = []
    tfa = []
    for vector in vector_list:
        a.append(convert_element_to_fp32_array(vector))
        tfa.append(convert_element_to_tffp32_array(vector))
    fp32_res = fp32.summation(a)
    tffp32_res = tf_reduce_sum(tfa)

    if check_float_equal(fp32_res, tffp32_res):
        test_res_str = f'PASSED SUM({vector_list}), res: {fp32_res}'
    else:
        test_res_str = f'FAILED SUM({vector_list}), fp32: {fp32_res}, tffp32: {tffp32_res}'
    print(test_res_str)
    return test_res_str

def rand_vector():
    vector = list()
    for i in range(vector_element_num):
        #vector.append(float(random_bf16()))
        vector.append(float(random_bf16_range(-10, 10)))
    return vector

def rand_test(times: int, vector_num: int = 16):
    # Generate 64 input random bf16
    fail_list = []
    for i in range(times):
        vector_list = list()
        for j in range(vector_num):
            vector_list.append(rand_vector())
        test_res_str = test_summation(vector_list)
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
