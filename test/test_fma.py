from .commonimport import *
from test.utils import *
from test.test_class import TestOperationBase

from typing_extensions import Self

# FAILED FMA(-30408704.0, 3.844499588012695e-06, -0.310546875), bf16: Bfloat16(-117.0, sign = 1, exponent=6, mantissa=106), tfbf16: -117.5
# FAILED FMA(474.0, -44544.0, -458752.0), bf16: Bfloat16(-21626880.0, sign = 1, exponent=24, mantissa=37), tfbf16: -2.14958e+07

class TestFMA(TestOperationBase):
    test_set = [
    #        (1.0, 2.0, 4.0),
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
            # Special cases
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
    # FAILED fma(Float32(1.658261946118133e-32, sign = 0, exponent=-106, mantissa=2896938), Float32(0.0002102892758557573, sign = 0, exponent=-13, mantissa=6062361), Float32(-3.50296132728099e-36, sign = 1, exponent=-118, mantissa=1376223)), lib: Float32(-1.5814429489254797e-38, sign = 1, exponent=-126, mantissa=2896946), tf: -1.5814289359408364e-38, ulp_error: 100
    #(1.658261946118133e-32, 0.0002102892758557573, -3.50296132728099e-36)
    # FAILED fma(Float32(-3.4955471162181546e-23, sign = 1, exponent=-75, mantissa=2689223), Float32(3.922678904061092e+20, sign = 0, exponent=68, mantissa=2760316), Float32(0.01334461197257042, sign = 0, exponent=-7, mantissa=5940060)), lib: Float32(-0.0003672972379717976, sign = 1, exponent=-12, mantissa=4231629), tf: -0.000367296946933493, ulp_error: 10
    #(float('nan'), 1.0, 2.0),

    #(bf16_obj.from_hex(0x08f1), bf16_obj.from_hex(0xffad), bf16_obj.from_hex(0x774b)),
#    (fp32(sign = 0, exponent=-36, mantissa=6579274), fp32(sign = 0, exponent=102, mantissa=6455005), fp32(sign = 0, exponent=-37, mantissa=7291628)),
#    (fp32(sign = 1, exponent=110, mantissa=4879951), fp32(sign = 0, exponent=-22, mantissa=6206616), fp32(sign = 1, exponent=79, mantissa=7117663)),
    # almost zero case
#    (fp32(sign = 0, exponent=-16, mantissa=3610392), fp32(sign = 0, exponent=-111, mantissa=7339142), fp32(sign = 0, exponent=-126, mantissa=146679)),
#    (fp32(sign = 0, exponent=-6, mantissa=1669669), fp32(sign = 0, exponent=-120, mantissa=489366), fp32(sign = 1, exponent=-125, mantissa=870642)),
#    (fp32(sign = 0, exponent=-1, mantissa=5494362), fp32(sign = 1, exponent=-126, mantissa=3357491), fp32(sign = 0, exponent=-126, mantissa=5231156)),
#    (fp32(sign = 1, exponent=122, mantissa=2374758), fp32(sign = 0, exponent=4, mantissa=6939185), fp32(sign = 1, exponent=127, mantissa=5624403)),
#    (fp32(sign = 0, exponent=-82, mantissa=116089), fp32(sign = 0, exponent=-44, mantissa=5087916), fp32(sign = 1, exponent=-126, mantissa=5820685)),
    #(fp32(sign = 1, exponent=62, mantissa=7427298), fp32(sign = 1, exponent=65, mantissa=5220737), fp32(sign = 1, exponent=127, mantissa=5526448)),
    #(fp32(sign = 1, exponent=-121, mantissa=5557884), fp32(sign = 0, exponent=42, mantissa=407632), fp32(sign = 0, exponent=-79, mantissa=6235593)),
    #(fp32(sign = 0, exponent=-61, mantissa=2616410), fp32(sign = 1, exponent=109, mantissa=1622550), fp32(sign = 1, exponent=24, mantissa=4674590)), 
#    (fp32(sign = 0, exponent=84, mantissa=7180711), fp32(sign = 1, exponent=-62, mantissa=5976674), fp32(sign = 1, exponent=-2, mantissa=780509)),
#    (fp32(sign = 1, exponent=48, mantissa=3825956), fp32(sign = 1, exponent=-13, mantissa=3249702), fp32(sign = 0, exponent=10, mantissa=533179)),
    (fp32(sign = 1, exponent=-83, mantissa=32340), fp32(sign = 1, exponent=3, mantissa=93621), fp32(sign = 1, exponent=-104, mantissa=7984477)),
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
        res = self.operation(*operand, algorithm = "MULTI_PATH")
        #print(res)

        ## when input operands are bf16 and output is fp32, tf dtype conversion should be like this:
        ## float -> bf16(for input type) -> fp32(for operation) 
        tf_operand_input = tuple(map(conv_to_tf_dtype, (input[0], input[1]), [self.input_ftype]*(self._INPUT_NUM - 1)))
        tf_operand_input = tuple(map(conv_to_tf_dtype, (tf_operand_input[0], tf_operand_input[1]), [self.ftype]*(self._INPUT_NUM - 1)))
        tf_operand_output = conv_to_tf_dtype(input[2], self.ftype)
        if self.input_ftype == self.ftype:
            # for mod 0, 1
            tf_operand = tuple(map(convert_to_tffp64, input))
        else:
            # for mod 2
            tf_operand = (*tf_operand_input, tf_operand_output)
        # for precision, cast operands to tf.float64. and cast result to ftype
        tfres = self.tf_operation(*tf_operand)
        tfres = conv_from_tf_to_tf_dtype(tfres, self.ftype)
        tfres = check_subnorm(tfres)
        
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
            # JMG: fix here
            #exp_a, exp_b, exp_c = self.set_exponent(24)
            #exp_a, exp_b, exp_c = self.set_exponent(24)
            #exp_a, exp_b, exp_c = self.set_exponent(10)
            #exp_a, exp_b, exp_c = self.set_exponent(-10)
            #(a, b, c), fp32_res, test_res_str = self.test_body((random_fp(self.input_ftype, exp_a, exp_a), random_fp(self.input_ftype, exp_b, exp_b), random_fp(self.ftype, exp_c, exp_c)))
            (a, b, c), fp32_res, test_res_str = self.test_body((random_fp(self.input_ftype), random_fp(self.input_ftype), random_fp(self.ftype)))
            test_list.append([a, b, c, fp32_res])
            if check_fail_status(test_res_str):
                fail_list.append(test_res_str)
        check_fail_list(fail_list)
        return test_list

    def set_exponent(self, i: int):
        #diff = random.randint(i, bf16_config.exp_max)
        diff = i
        exp_a = random.randint(-bf16_config.exp_max + 1, bf16_config.exp_max)
        exp_b = random.randint(-bf16_config.exp_max + 1, bf16_config.exp_max)
        exp_c_pre = exp_a + exp_b - diff
        exp_c = exp_c_pre if (-bf16_config.exp_max + 1 < exp_c_pre) and (exp_c_pre < bf16_config.exp_max) else 0
        return exp_a, exp_b, exp_c

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
