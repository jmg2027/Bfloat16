from bf16.bf16 import Bfloat16 as bf16

def test():
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
