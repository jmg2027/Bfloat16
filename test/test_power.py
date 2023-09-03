from bf16 import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

def test():
    a = bf16.from_float(1)
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
