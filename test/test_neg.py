from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

def neg_test():
    a = bf16.from_float(3.56)
    print(a)
    print(-a)
    return
