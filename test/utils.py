import tensorflow as tf
import random

from bf16.bf16 import Bfloat16 as bf16

def convert_to_bf16(num: float):
    return bf16.float_to_bf16(num)

def convert_to_tfbf16(num: float):
    return tf.cast(num, tf.bfloat16)

def check_float_equal(res1, res2):
    # nan cannot compare
    if (str(float(res1)) == 'nan') & (str(float(res2)) == 'nan'):
        return True
    if float(res1) == float(res2):
        return True
    else:
        return False

def random_bf16():
    min_exp = - bf16.exp_max + 1
    max_exp = bf16.exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = bf16.mant_max
    rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return bf16(rand_sign, rand_exp, rand_mant)

def check_fail_list(fail_list: list):
    print("TEST RESULT")
    if fail_list == []:
        fail_list.append("ALL TEST PASSED")
    else:
        print("Test failed")
    for i in fail_list:
        print(i)
    return

def check_fail_status(test_res_str: str) -> bool:
    if "FAILED" in test_res_str:
        fail_status = True
    else: 
        fail_status = False
    return fail_status