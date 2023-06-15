from bf16.bf16 import Bfloat16 as bf16
from test.utils import *

test_set = [
    0.0,
    0.00000019,
    1.0,
    2.0,
    3.0,
    -1.0,
    -2.0,
    -3.0,
    -1.24,
    1.24,
    10.34,
    100.0000001,
    100.9999999,
    32767,
    32768,
    -32768,
    -32769,
]


def test_fptoint(fp: float):
    a = convert_to_bf16(fp)
    bf16_res = bf16.fptoint(a)

    tfa = convert_to_tfbf16(fp)
    tfbf16_res = convert_tfbf16_to_int(tfa)

    # debug
    print('bf16', bf16_res)
    print('tfbf16', tfbf16_res)
    
    if bf16_res == tfbf16_res:
        test_res_str = f'PASSED {a}'
    else:
        test_res_str = f'FAILED {a}.fptoint(), bf16: {bf16_res}, tfbf16: {tfbf16_res}'
    print(test_res_str)
    return test_res_str

def rand_test(times: int):
    fail_list = []
    for i in range(times):
        test_res_str = test_fptoint(float(random_bf16_range(0, 14)))
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return

def test():
    fail_list = []
    for a in test_set:
        test_res_str = test_fptoint(a)
        if check_fail_status(test_res_str):
            fail_list.append(test_res_str)
    check_fail_list(fail_list)
    return
