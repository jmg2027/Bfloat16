from test.utils import *
from .commonimport import *

from abc import ABCMeta, abstractmethod


FloatTffloatT = TypeVar('FloatTffloatT', bf16, fp32, tfbfloat16, tffloat32)
OpFuncT = Callable[[Tuple[FloatTffloatT, ...]], FloatTffloatT]


class TestAbsClass(metaclass=ABCMeta):
    # Test will have number of mods...
    # In case of fma, it will have 3 mods
    # mod 0: a, b = bf16, c = bf16
    # mod 1: a, b = bf16, c = fp32
    # mod 2: a, b = fp32, c = fp32
    # then it will have 3 test sets
    # test_set_0: List[Tuple[bf16, bf16, bf16]]
    # test_set_1: List[Tuple[bf16, bf16, fp32]]
    # test_set_2: List[Tuple[fp32, fp32, fp32]]
    # by test set types, it will be able to know which mod it should use
    # and it will be able to know which operation it should use
    # You need to convert test_set to format of test_set
    # It will be given 
    test_set: List
    mod_structure: Dict[int, Tuple[Type, ...]]
    _INPUT_NUM: int
    
    @abstractmethod
    def __init__(self, mod: int, test_set: list, operation: str) -> None:
        pass

    @abstractmethod
    def set(self, mod: int, test_set: list, operation: str) -> None:
        pass
 
    @abstractmethod
    def set_test_set(self, test_set: list) -> None:
        pass

    @abstractmethod
    def set_mod(self, mod: int) -> None:
        pass

    @abstractmethod
    def _set_f_ops(self, mod: int) -> None:
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


class TestOperationBase(TestAbsClass):
    """
    Parent class for testing operations
    """
    test_set: List
    mod_structure: Dict[int, Tuple[Type, ...]]
    #ftype: Type[Self]
    _INPUT_NUM: int
    _TEST_SET_STRUCTURE: str
    mod_list: Dict

    _TF_OPS: Dict[str, OpFuncT] = {
        'mul': lambda a, b: a * b,
        'add': lambda a, b: a + b,
        'fma': lambda a, b, c: a * b + c,
        'summation': tf.reduce_sum,
    } # type: ignore
    
    def __init__(self, mod, test_set: List, operation: str) -> None:
        self.set(mod, test_set, operation)

    def set(self, mod, test_set: List, operation: str) -> None:
        self.test_set, self.mod = self.set_test_set(test_set), self.set_mod(mod)
        self.ftype = self.set_ftype(mod)
        self._f_ops: Dict[str, Union[OpFuncT, None]] = self._set_f_ops(self.ftype)
        self.operation = self._set_operation(self._f_ops, operation)
        self.tf_operation = self._set_operation(self._TF_OPS, operation) # type: ignore
        self.op: str = operation

    def set_mod(self, mod: int) -> int:
        mod_value = self.mod_list.get(mod, None)
        if mod_value is None:
            raise ValueError(f"Unsupported mod {self.mod}")
        return mod
 
    def set_test_set(self, test_set: List) -> List:
        if self._check_test_set(test_set):
            return test_set
        else:
            raise TypeError(f'{self.__class__.__name__} test_set structure should be:\n {self._TEST_SET_STRUCTURE}')

    @abstractmethod
    def set_ftype(self, mod) -> Type[Self]:
        pass

    def _set_f_ops(self, ftype: Type[Self]) -> Dict[str, Union[OpFuncT, None]]:
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
            raise ValueError(f"Unsupported operation {op}")
        return operation
 
    # this method should be defined in subclasses
    def _check_test_set(self, test_set: list) -> bool:
        return False
    
    def _check_input_num(self, v: list) -> bool:
        res = True
        if len(v) != self._INPUT_NUM:
            res = False
        return res

    def test_body(self, input: Tuple[Union[int, float, bf16, fp32], ...]) -> Tuple:
        # input: hex(int), float, bf16, fp32
        operand = tuple(map(cast_float, input, [self.ftype]*self._INPUT_NUM))
        #print(operand)
        res = self.operation(*operand)
        #print(res)

        #tf_operand = tuple(map(conv_to_tf_dtype, input, [self.ftype]*self._INPUT_NUM))
        tf_operand = tuple(map(conv_to_tf_dtype, operand, [self.ftype]*self._INPUT_NUM))
        #print(tf_operand)
        tfres = self.tf_operation(*tf_operand)
        
        if check_float_equal(res, tfres):
            test_res_str = f'PASSED {self.op}{input}, res: {res}'
        else:
            test_res_str = f'FAILED {self.op}{input}, lib: {res}, tf: {tfres}'
        print(test_res_str)
        test_ret = list(i for i in input)
        test_ret.append(res)
        test_ret.append(test_res_str)
        return input, res, test_res_str

    def rand_test(self, times: int):
        test_list = []
        fail_list = []
        for i in range(times):
            #(a, b), fp32_res, test_res_str = self.test_body((float(random_bf16()), float(random_bf16())))
            (a, b), fp32_res, test_res_str = self.test_body((random_fp(self.ftype), random_fp(self.ftype)))
            test_list.append([a, b, fp32_res])
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return test_list

    def test(self):
        fail_list = []
        for v in self.test_set:
            (a, b), fp32_res, test_res_str = self.test_body(v)
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return
