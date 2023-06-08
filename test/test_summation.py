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

def test_summation():
    test.test_summation.test()

def test_all():
    test_bf16_module()
    test_pow()
    test_neg()
    test_mul()
    test_add()
    test_fma()
    test_summation()

def test_rand_mul(times):
    test.test_mul.rand_test(times)
    return

def test_rand_add(times):
    test.test_add.rand_test(times)
    return

def test_rand_fma(times):
    test.test_fma.rand_test(times)
    return

def test_rand_summation(times):
    test.test_summation.rand_test(times)
    return

def test_random_all():
    pass


if __name__ == "__main__":
#    test_mul()
#    test_add()
#    test_fma()
#    test_summation()
#    test_rand_mul(1000)
#    test_rand_add(1000)
#    test_rand_fma(1000)
    test_rand_summation(10)
#    a = -8.715097876569077e+29
#    b = -9.769962616701378e-14
#    c = -9.907919180215091e+16
#    A = bf16.float_to_bf16(a)
#    B = bf16.float_to_bf16(b)
#    C = bf16.float_to_bf16(c)
#    res = bf16.fma(A, B, C)
#    print(res)
#    test.test_fma.test_fma(a, b, c)
#    A = bf16(0, 0, bf16.mant_max)
#    B = bf16(0, 0, bf16.mant_max)
#    print(A)
#    print(B)
#    a = float(A)
#    b = float(B)
#    test.test_add.test_add(a, b)

#    print(C)
#    tfc = test.test_fma.convert_to_tfbf16(c)
#    print(tfc)
    pass
