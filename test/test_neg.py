from bf16.bf16 import Bfloat16 as bf16

def neg_test():
    a = bf16.float_to_bf16(3.56)
    print(a)
    print(-a)
    return
