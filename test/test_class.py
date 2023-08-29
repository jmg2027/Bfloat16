from test.utils import *
from bf16 import *

from abc import ABCMeta, abstractmethod

from typing import Generic


FloatTffloatT = TypeVar('FloatTffloatT', bf16, fp32, tfbfloat16, tffloat32)
OpFuncT = Callable[[Tuple[FloatTffloatT, ...]], FloatTffloatT]


class TestAbsClass(Generic[FloatBaseT], metaclass=ABCMeta):
    test_set: List
    ftype: Type[FloatBaseT]
    _INPUT_NUM: int
    
    @abstractmethod
    def __init__(self, ftype: Type[FloatBaseT], test_set: list, operation: str) -> None:
        pass

    @abstractmethod
    def set(self, ftype: Type[FloatBaseT], test_set: list, operation: str) -> None:
        pass
 
    @abstractmethod
    def set_test_set(self, test_set: list) -> None:
        pass

    @abstractmethod
    def set_ftype(self, ftype: Type[FloatBaseT]) -> None:
        pass

    @abstractmethod
    def _set_f_ops(self, ftype: Type[FloatBaseT]) -> None:
        pass

    @abstractmethod
    def _set_operation(self, operation_dict: Dict, op) -> None:
        pass
 
    # this method should be defined in subclasses
    @abstractmethod
    def _check_test_set(self, test_set: List) -> None:
        pass
    
    @abstractmethod
    def _check_input_num(self, v: List) -> None:
        pass

    @abstractmethod
    def test_body(self, input: Tuple) -> None:
        pass

    @abstractmethod
    def rand_test(self, times: int) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass


class TestOperationBase(TestAbsClass[FloatBaseT]):
    """
    Parent class for testing operations
    """
    test_set: List
    ftype: Type[FloatBaseT]
    _INPUT_NUM: int
    _TEST_SET_STRUCTURE: str

    _TF_OPS: Dict[str, OpFuncT] = {
        'mul': lambda a, b: a * b,
        'add': lambda a, b: a + b,
        'fma': lambda a, b, c: a * b + c,
        'summation': tf.reduce_sum,
    } # type: ignore
    
    def __init__(self, ftype: Type[FloatBaseT], test_set: list, operation: str) -> None:
        self.set(ftype, test_set, operation)

    def set(self, ftype: Type[FloatBaseT], test_set: list, operation: str) -> None:
        self.test_set, self.ftype = self.set_test_set(test_set), self.set_ftype(ftype)
        self._f_ops: Dict[str, Union[OpFuncT, None]] = self._set_f_ops(ftype)
        self.operation = self._set_operation(self._f_ops, operation)
        self.tf_operation = self._set_operation(self._TF_OPS, operation) # type: ignore
        self.op: str = operation
 
    def set_test_set(self, test_set: list) -> None:
        if self._check_test_set(test_set):
            return test_set
        else:
            raise TypeError(f'{self.__name__} test_set structure should be:\n {self._TEST_SET_STRUCTURE}')

    def set_ftype(self, ftype: Type[FloatBaseT]) -> Type[FloatBaseT]:
        if (ftype == fp32) | (ftype == bf16):
            return ftype
        else:
            raise TypeError('Ftype should be bf16 or fp32 class')

    def _set_f_ops(self, ftype: Type[FloatBaseT]) -> Dict[str, Union[OpFuncT, None]]:
        return \
        {
            'mul': getattr(ftype, '__mul__', None),
            'add': getattr(ftype, '__add__', None),
            'fma': getattr(ftype, 'fma', None),
            'summation': getattr(ftype, 'summation', None),
        }

    def _set_operation(self, operation_dict: \
                       Dict[str, Union[OpFuncT, None]], op: str) \
                        -> Union[OpFuncT, None]:
        operation: Union[OpFuncT, None] = operation_dict.get(op, None)
        if operation is None:
            raise ValueError(f"Unsupported operation {self.op}")
        return operation
 
    # this method should be defined in subclasses
    def _check_test_set(self, test_set: list) -> bool:
        return False
    
    def _check_input_num(self, v: list) -> bool:
        res = True
        if len(v) != self._INPUT_NUM:
            res = False
        return res

    def test_body(self, input: Tuple[Union[int, float, bf16, fp32], ...]):
        # input: hex(int), float, bf16, fp32
        operand = tuple(map(cast_float, input, [self.ftype]*self._INPUT_NUM))
        res = self.operation(*operand)

        tf_operand = tuple(map(conv_to_tf_dtype, input, [self.ftype]*self._INPUT_NUM))
        tfres = self.tf_operation(*tf_operand)
        
        check_float_equal(res, tfres)
        if check_float_equal:
            test_res_str = f'PASSED {self.op}{input}, res: {res}'
        else:
            test_res_str = f'FAILED {self.op}{input}, bf16: {res}, tfbf16: {tfres}'
        print(test_res_str)
        test_ret = list(i for i in input)
        test_ret.append(res)
        test_ret.append(test_res_str)
        return input, res, test_res_str
        #return test_ret

    def rand_test(self, times: int):
        test_list = []
        fail_list = []
        for i in range(times):
            a, b, fp32_res, test_res_str = self.test_body(float(random_bf16()), float(random_bf16()))
            test_list.append([a, b, fp32_res])
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return test_list

    def test(self):
        fail_list = []
        for v in self.test_set:
            test_res_str = self.test_body(v)
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return