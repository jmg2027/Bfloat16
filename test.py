from bf16.bf16 import Bfloat16 as bf16
import numpy as np
import tensorflow as tf

def default_test():
    a = bf16(1,12,25)
    #e = a.bf16_to_fp32()
    #print(e)
    #print(a.isnan())
    #print(a)
    #print(len(a.bin()))

    b = bf16(0,32,17)
    #f = b.bf16_to_fp32()
    #print(f)
    #print(b)

    c = bf16.float_to_bf16(1.6)
    d = bf16.float_to_bf16(-1.99999988079)

    #g = c.bf16_to_fp32()
    #print(g)
    #
    #print(c)
    #print(d)
    #
    #print(c.bf16_to_float())
    #print(d.bf16_to_float())
    #
    #print("Today")
    #print(c.bf16_to_tfbf16())
    #print(c.bf16_to_fp32())

    print(a.decompose_bf16())
    print(b.decompose_bf16())
    print(c.decompose_bf16())
    print(d.decompose_bf16())
    return

def power_test():
    a = bf16.float_to_bf16(1)
    print(a)
    print(a.pow(1))
    print(a.pow(3))
    print(a.pow(-125))

    print(a.pow(126))
    print(a.pow(127))
    print(a.pow(128))
    print(a.pow(129))
    print(a.pow(255))
    print(a.pow(256))
    print(a.pow(-126))
    print(a.pow(-127))
    print(a.pow(-128))
    print(a.pow(-255))
    print(a.pow(-256))
    return

def neg_test():
    a = bf16.float_to_bf16(3.56)
    print(a)
    print(-a)
    return

def bitstring_test():
    a = bf16.float_to_bf16(25.69704)
    print(a.decompose_bf16())
    bs, be, bm = a.decompose_bf16()
    c = bf16.compose_bf16(bs, be, bm)
    print(c)
    return

def mul_test_set(num1, num2):
    a = bf16.float_to_bf16(num1)
    b = bf16.float_to_bf16(num2)
    print(a)
    print(b)
    print(a*b)
    tfa = tf.cast(num1, tf.bfloat16)
    tfb = tf.cast(num2, tf.bfloat16)
    print(tfa * tfb)
    return

def mul_test():
    mul_test_set(2, 12)
    mul_test_set(-2, 12)
    mul_test_set(2, -12)
    mul_test_set(-2, -12)
    mul_test_set(123.124, 381.58)
    mul_test_set(123.124, -381.58)
    mul_test_set(-123.124, 381.58)
    mul_test_set(-123.124, -381.58)
    mul_test_set(0.00076, 0.3256)
    mul_test_set(-0.00076, 0.3256)
    mul_test_set(0.00076, -0.3256)
    mul_test_set(-0.00076, -0.3256)
    mul_test_set(111111111.111111111, 999999999999.999999999999)
    mul_test_set(0, 0)
    return

def add_test_set(num1, num2):
    a = bf16.float_to_bf16(num1)
    b = bf16.float_to_bf16(num2)
    tfa = tf.cast(num1, tf.bfloat16)
    tfb = tf.cast(num2, tf.bfloat16)
    bf16_res = a + b
    tf_res = tfa + tfb
    print(bf16_res)
    print(tf_res)
    print(bf16_res - tf_res)
    return

def add_test():
    add_test_set(2, 12)
    add_test_set(-2, 12)
    add_test_set(2, -12)
    add_test_set(-2, -12)
    add_test_set(25.0924, 24.8076)
    add_test_set(-25.0924, 24.8076)
    add_test_set(25.0924, -24.8076)
    add_test_set(-25.0924, -24.8076)
    add_test_set(123.124, 381.58)
    add_test_set(123.124, -381.58)
    add_test_set(-123.124, 381.58)
    add_test_set(-123.124, -381.58)
    add_test_set(0.00076, 0.3256)
    add_test_set(-0.00076, 0.3256)
    add_test_set(0.00076, -0.3256)
    add_test_set(-0.00076, -0.3256)
    add_test_set(111111111.111111111, 999999999999.999999999999)
    # Corner cases
    add_test_set(0, 0)
    add_test_set(10.10293, -0.0000000000000000000000000000000000000001)
    add_test_set(10.10293, 0.0000000000000000000000000000000000000001)
   # 100931731456
   # 1000727379968
    add_test_set(101029300000, 999999999999)
    add_test_set(101029300000, -999999999999)
    return

def fma_test_set(num1, num2, num3):
    a = bf16.float_to_bf16(num1)
    b = bf16.float_to_bf16(num2)
    c = bf16.float_to_bf16(num3)
#    print(a)
#    print(b)
    print(bf16.fma(a,b,c))
    tfa = tf.cast(num1, tf.bfloat16)
    tfb = tf.cast(num2, tf.bfloat16)
    tfc = tf.cast(num3, tf.bfloat16)
    print(tfa * tfb + tfc)
    return

def fma_test():
    fma_test_set(1.0, 2.0, 3.0)
    return

if __name__ == "__main__":
#    power_test()
#    neg_test()
#    mul_test()
    add_test()
#    fma_test()
    pass
