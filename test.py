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

def test_fptoint():
    test.test_fptoint.test()

def test_inttofp():
    test.test_inttofp.test()

def test_all():
    test_bf16_module()
    test_pow()
    test_neg()
    test_mul()
    test_add()
    test_fma()
    test_summation()
    test_fptoint()
    test_inttofp()

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

def test_rand_fptoint(times):
    test.test_fptoint.rand_test(times)

def test_rand_inttofp(times):
    test.test_inttofp.rand_test(times)

def test_random_all(times):
    test_rand_mul(times)
    test_rand_add(times)
    test_rand_fma(times)
    test_rand_summation(times)
    test_rand_fptoint(times)
    test_rand_inttofp(times)
    pass


if __name__ == "__main__":
#    test_mul()
#    test_add()
#    test_fma()
#    test_summation()
#    test_fptoint()
#    test_inttofp()
#    test_rand_mul(1000)
#    test_rand_add(10000)
#    test_rand_fma(1000)
    test_rand_summation(10)
#    test_rand_fptoint(1000)
#    test_rand_inttofp(1000)
