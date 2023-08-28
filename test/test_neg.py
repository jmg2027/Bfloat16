from bf16.bf16 import Bfloat16 as bf16

def neg_test():
    a = bf16.from_float(3.56)
    print(a)
    print(-a)
    return
