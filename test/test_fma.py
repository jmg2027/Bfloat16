from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

# FAILED FMA(-30408704.0, 3.844499588012695e-06, -0.310546875), bf16: Bfloat16(-117.0, sign = 1, exponent=6, mantissa=106), tfbf16: -117.5
# FAILED FMA(474.0, -44544.0, -458752.0), bf16: Bfloat16(-21626880.0, sign = 1, exponent=24, mantissa=37), tfbf16: -2.14958e+07

class TestFMA(TestOperationBase):
    test_set = [
    #        (1.0, 2.0, 4.0),
    #        (1.0, 2.0, 2.0),
    #        (1.0, 2.0, 3.0),
    #        (100.0, 2.0, 3.0),
    #        (10000.0, 2.0, 3.0),
    #        (1.0, 2.0, 30000.0),
    #        (1000000000.0, 200.0, 3.0),
    ## debug: bf16 mantissa is 1 less than tf.bfloat16
    ## debug: 3000000000000000.0 mantissa in fp32 is 0101010_1000011110111111
    ## debug: in this case, isn't it not to round up case? It seems tf round this up
    #        (0.000000001, 0.11111112, 3000000000000000.0),
    #        (-1.0, 2.0, 4.0),
    #        (-1.0, 2.0, 3.0),
    #        (1.0, -2.0, 3.0),
    #        (1.0, 2.0, -3.0),
    #        (-1.0, -2.0, -3.0),
    #        (-100.0, 2.0, 3.0),
    #        (100.0, 2.0, -3.0),
    #        (10000.0, -2.0, 3.0),
    #        # Special cases
    #        (0, 2.8, 4.5),
    #        (0, 0, 4.5),
    #        (4.5, 2.8, 0),
    #        (-5.4752209710517974e-18, 5.5340232221128655e+19, 0.00135040283203125),
    #        (-3124765696.0, 4.3706804842003066e-10, 8.69215202331543),
    #        (-1.0, 1.0, 8.69215202331543),
    #        (1.0, 1.0, 8.69215202331543),
    ## case: exp_diff = -2 and subtract -> needs lzc
    #(-8224078003437568.0, 1.888424983115578e-38, 2.5540332323568454e-22),
    #(8224078003437568.0, 1.888424983115578e-38, 2.5540332323568454e-22),

    #(bf16_obj.from_hex(0x08f1), bf16_obj.from_hex(0xffad), bf16_obj.from_hex(0x774b))
    ]
 
    test_operation = 'fma'
    _INPUT_NUM = 3
    _TEST_SET_STRUCTURE = '[(num1, num2, num3), (num4, num5, num6), ...]'
    mod_list = {0: (fp32, fp32, fp32), 1: (bf16, bf16, bf16), 2: (bf16, bf16, fp32)}

    def __init__(self, mod, test_set = test_set) -> None:
        super().__init__(mod, test_set, self.test_operation)
        self.input_ftype = self.set_input_ftype(mod)

    # this method should be defined in subclasses
    def set_ftype(self, mod):
        ftype = self.mod_list[mod][2]
        return ftype

    # this method should be defined in subclasses
    def set_input_ftype(self, mod):
        ftype = self.mod_list[mod][0]
        return ftype
 
    # override
    def test_body(self, input: Tuple[Union[int, float, bf16, fp32], ...]) -> Tuple:
        # input: hex(int), float, bf16, fp32
        operand_input = tuple(map(cast_float, (input[0], input[1]), [self.input_ftype]*(self._INPUT_NUM - 1)))
        operand_output = cast_float(input[2], self.ftype)
        operand = (*operand_input ,operand_output)
        res = self.operation(*operand)
        #print(res)

        ## when input operands are bf16 and output is fp32, tf dtype conversion should be like this:
        ## float -> bf16(for input type) -> fp32(for operation) 
        #tf_operand_input = tuple(map(conv_to_tf_dtype, (input[0], input[1]), [self.input_ftype]*(self._INPUT_NUM - 1)))
        #tf_operand_input = tuple(map(conv_to_tf_dtype, (tf_operand_input[0], tf_operand_input[1]), [self.ftype]*(self._INPUT_NUM - 1)))
        #tf_operand_output = conv_to_tf_dtype(input[2], self.ftype)
        # for precision, cast operands to tf.float64. and cast result to ftype
        tf_operand = tuple(map(convert_to_tffp64, input))
        tfres = self.tf_operation(*tf_operand)
        tfres = conv_from_tf_to_tf_dtype(tfres, self.ftype)
        
        #test_res_str = f'{[i.hex() for i in (*operand_input, operand_output)]}\t\t{res.hex()}'
        if check_float_equal(res, tfres):
            test_res_str = f'PASSED {self.op}{input}, res: {res}'
            #test_res_str = f'PASSED {self.op}{[i.hex() for i in (*operand_input, operand_output)]}, res: {res.hex()}'
        else:
            test_res_str = f'FAILED {self.op}{input}, lib: {res}, tf: {tfres}, ulp_error: {calc_ulp_error(res, tfres)}'
            #test_res_str = f'FAILED {self.op}{[i.hex() for i in (*operand_input, operand_output)]}, res: {res.hex()}'
        print(test_res_str)
        test_ret = list(i for i in input)
        test_ret.append(res)
        test_ret.append(test_res_str)
        return input, res, test_res_str

    # override
    def rand_test(self, times: int):
        test_list = []
        fail_list = []
        for i in range(times):
            (a, b, c), fp32_res, test_res_str = self.test_body((random_fp(self.input_ftype), random_fp(self.input_ftype), random_fp(self.ftype)))
            test_list.append([a, b, c, fp32_res])
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return test_list

    # override
    def test(self):
        fail_list = []
        for v in self.test_set:
            (a, b, c), fp32_res, test_res_str = self.test_body(v)
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return

    # this method should be defined in subclasses
    def _check_test_set(self, test_set: list) -> bool:
        res = True
        # check structure
        if not isinstance(test_set, list):
            res = False
        for v in test_set:
            # check structure
            if not isinstance(v, tuple):
                res = False
            # number of input
            if not self._check_input_num:
                res = False
        # input type check is handled in cast_float function in util
        return res   
