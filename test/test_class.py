from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
from test.utils import *

from abc import *

class TestAbsClass(metaclass=ABCMeta):
    test_set: list
    input_num: int
    test_set_structure: str
    ftype: object
    operation: object

    @abstractmethod
    def __init__(self, ftype):
        pass

    @abstractmethod
    def set(self, test_set, ftype):
        pass

    @abstractmethod
    def set_test_set(self, test_set):
        pass

    @abstractmethod
    def set_ftype(self, ftype):
        pass

    @abstractmethod
    def _check_test_set(self, test_set):
        pass

    @abstractmethod
    def test_body(self, *args):
        pass

    @abstractmethod
    def rand_test(self, times: int):
        pass

    @abstractmethod
    def test(self):
        pass

