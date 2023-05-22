from bf16.bf16 import Bfloat16 as bf16
import numpy as np
import tensorflow as tf


import test

def test_bf16_module():
    test.test_bf16module.test()

def test_pow():
    test.test_power.test()

def test_neg():
    test.test_neg.test()

def test_mul():
    test.test_mul.test()

def test_add():
    test.test_add.test()

def test_fma():
    test.test_fma.test()

def test_all():
    test_bf16_module()
    test_pow()
    test_neg()
    test_mul()
    test_add()
    test_fma()

def test_random_all():
    pass

def sum_test():
#    bf16.summation([])
    print(bf16.summation([]))
    input_vector = np.array([
        1.0,
        -1.2,
        4.0,
        5.0,
        -10.0,
        -20.0,
        30.0,
        -100.0,
            ])
    weight_vector = np.array([
        2.0,
        -4.2,
        7.0,
        -9.0,
        10.0,
        -20.0,
        30.567,
        -400.6,
            ])
    print(np.inner(input_vector, weight_vector))
    return


if __name__ == "__main__":
#    power_test()
#    neg_test()
#    mul_test()
#    add_test()
#    fma_test()
#    a = bf16.float_to_bf16(2.0)
#    b = bf16.float_to_bf16(-2.0)
#    a = bf16.float_to_bf16(-2.0)
#    b = bf16.float_to_bf16(4.0)
#    a = bf16.float_to_bf16(-2.0)
#    b = bf16.float_to_bf16(4.0)
#    c = bf16.float_to_bf16(3.0)
#    print(a+b)
#    sum_test()
#    rand_add_test(10)
#    test_mul()
    test_add()
#    test_fma()
    pass
