from bf16.bf16 import Bfloat16 as bf16

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
    a = bf16.float_to_bf16(3.56)
    print(a)
    print(a.pow(3))

    print(a.pow(126))
    print(a.pow(127))
    print(a.pow(128))
    print(a.pow(129))
    print(a.pow(256))
    print(a.pow(-126))
    print(a.pow(-127))
    print(a.pow(-128))
#    print(a.pow(-129))
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

def mul_test():
    a = bf16.float_to_bf16(12)
    b = bf16.float_to_bf16(2)
    print(a*b)
    print(12.0 * 2.0)
    return

if __name__ == "__main__":
    power_test()
#    neg_test()
#    mul_test()
    pass
