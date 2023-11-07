import numpy as np
import tensorflow as tf

from float_class import bf16, fp32

import test
import test.utils
import float_class.floatopwrap as opwrap

# delete opwrap
# make test auto convert to each types by its mod value
# summation has its own mod in floatclass

from typing import (
    Union,
    Type,
    TypeVar,
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional
    )


from test.utils import FloatBaseT


# make test as new class and define each operations with it
# such as: 
# def test(op, ftype):
# if op == 'mul':
# t = test.test_class.T(mul, ftype)
# t.test()

def test_mul(mod = 0):
    t = test.test_mul.TestMul(mod)
    t.test()

def test_add(mod = 0):
    t = test.test_add.TestAdd(mod)
    t.test()

def test_fma(mod = 0):
    t = test.test_fma.TestFMA(mod)
    t.test()

def test_summation(mod = 0):
    t = test.test_summation.TestSummation(mod)
    t.test()

def test_fptoint():
    test.test_fptoint.test()

def test_inttofp():
    test.test_inttofp.test()

def test_rand_mul(times = 1000, mod = 0):
    t = test.test_mul.TestMul(mod)
    t.rand_test(times)
    return

def test_rand_add(times = 1000, mod = 0):
    t = test.test_add.TestAdd(mod)
    t.rand_test(times)
    return

def test_rand_fma(times = 1000, mod = 0):
    t = test.test_fma.TestFMA(mod)
    t.rand_test(times)
    return

def test_rand_summation(times = 1000, mod = 0):
    t = test.test_summation.TestSummation(mod)
    t.rand_test(times)
    return

def test_rand_fptoint(times = 1000):
    test.test_fptoint.rand_test(times)

def test_rand_inttofp(times = 1000):
    test.test_inttofp.rand_test(times)

if __name__ == "__main__":
#    test_mul(mod = 0)
#    test_mul(mod = 1)
#    test_add(mod = 0)
#    test_add(mod = 1)
    test_fma(mod = 0)
#    test_fma(mod = 1)
#    test_fma(mod = 2)
#    test_summation(mod = 0)
#    test_summation(mod = 1)
#    test_summation(mod = 2)
#    test_summation(mod = 3)
#    test_fptoint()
#    test_inttofp()
#    test_rand_mul(mod = 0)
#    test_rand_mul(mod = 1)
#    test_rand_add(mod = 0)
#    test_rand_add(mod = 1)
#    test_rand_fma(mod = 0)
#    test_rand_fma(mod = 1)
#    test_rand_fma(mod = 2)
#    test_rand_summation(mod = 0)
#    test_rand_summation(mod = 1)
#    test_rand_summation(mod = 2)
#    test_rand_summation(mod = 3)
#    test_rand_fptoint()
#    test_rand_inttofp()
