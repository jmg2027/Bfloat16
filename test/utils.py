import tensorflow as tf
import random

from bf16.bf16 import Bfloat16 as bf16

def convert_to_bf16(num: float):
    return bf16.float_to_bf16(num)

def convert_to_tfbf16(num: float):
    return tf.cast(num, tf.bfloat16)

def check_float_equal(res1, res2):
    if float(res1) == float(res2):
        return True
    else:
        return False

def random_bf16():
    min_exp = - bf16.exp_max - 1
    max_exp = bf16.exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = bf16.mant_max
    rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return bf16(rand_sign, rand_exp, rand_mant)
