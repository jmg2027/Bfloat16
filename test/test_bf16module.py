from bf16.bf16 import Bfloat16 as bf16

def test():
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

    c = bf16.from_float(1.6)
    d = bf16.from_float(-1.99999988079)

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

    print(a.decompose())
    print(b.decompose())
    print(c.decompose())
    print(d.decompose())
    return

if __name__ == "__main__":
    test()
