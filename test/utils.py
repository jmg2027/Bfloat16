import tensorflow as tf
import random

from typing import (
    Union,
    Type,
    TypeVar,
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional,
    )

from bf16 import bf16, fp32, FloatBaseT
from bf16.utils import bf16_ulp_dist

def conv_from_float(f: float, ftype: Type[FloatBaseT]) -> FloatBaseT:
    return ftype.from_float(f)

def convert_to_bf16(num: float) -> bf16:
    return bf16.from_float(num)

def convert_to_fp32(num: float) -> fp32:
    return fp32.from_float(num)

def conv_to_tf_dtype(num: float, ftype: Type[FloatBaseT] = fp32): # type: ignore
    '''
    num: float, ftype: Type[FloatBaseT] -> tf.bfloat16, tf.float32
    '''
    if ftype == bf16:
        return convert_to_tfbf16(num)
    elif ftype == fp32:
        return convert_to_tffp32(num)
    else:
        raise TypeError("Use this function only for Bfloat16 and Float32 type")

def convert_to_tfbf16(num: float) -> tf.bfloat16: # type: ignore
    return tf.cast(float(num), tf.bfloat16)

def convert_to_tffp32(num: float) -> tf.float32: # type: ignore
    return tf.cast(float(num), tf.float32)

def convert_tfbf16_to_int(num: tf.bfloat16) -> int: # type: ignore
    return int(num)

def convert_int_to_tfbf16(num: int) -> tf.bfloat16: # type: ignore
    return tf.cast(num, tf.bfloat16)

def cast_float(frepr: Union[int, float, bf16, fp32], \
                ftype: Type[FloatBaseT] = fp32) \
                -> FloatBaseT:
    # use float representation for integer inputs
    # integer inputs are used to hex input
    if isinstance(frepr, int) &  (ftype == fp32):
        # integer/hex repr
        return ftype.from_hex(frepr & 0xFFFF_FFFF)
    elif isinstance(frepr, int) &  (ftype == bf16):
        # integer/hex repr
        return ftype.from_hex(frepr & 0xFFFF)
    elif isinstance(frepr, float):
        # python float repr
        return ftype.from_float(frepr)
    elif isinstance(frepr, ftype):
        # need not to cast
        return frepr
    elif isinstance(frepr, bf16) & (ftype == fp32):
        # bf16 -> fp32
        return bf16.bf16_to_fp32(frepr)
    elif isinstance(frepr, fp32) & (ftype == bf16):
        # bf16 -> fp32
        return fp32.fp32_to_bf16(frepr)
    else:
        raise TypeError(f"Supported float representations are: hex(int), float, bf16, fp32. Current input: {frepr}")
        

def check_float_equal(res1: Union[bf16, fp32], res2) -> bool:
    # nan cannot be compared
    if (str(float(res1)) == 'nan') & (str(float(res2)) == 'nan'):
        return True
    bf16_ulp_error = bf16_ulp_dist(res1, res2)
    # ulp error under 2
    #if float(res1) == float(res2):
    if bf16_ulp_error <= 2:
        return True
    else:
        return False

def random_bf16() -> bf16:
    min_exp = - bf16.exp_max + 1
    max_exp = bf16.exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = bf16.mant_max
    rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return bf16(rand_sign, rand_exp, rand_mant)

def random_fp32() -> fp32:
    min_exp = - fp32.exp_max + 1
    max_exp = fp32.exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = fp32.mant_max
    rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return fp32(rand_sign, rand_exp, rand_mant)

def random_bf16_range(exp_min: int = -10, exp_max: int = 10) -> bf16:
    min_exp = exp_min 
    max_exp = exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = bf16.mant_max
    rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return bf16(rand_sign, rand_exp, rand_mant)

def check_fail_list(fail_list: list) -> None:
    print("TEST RESULT")
    if fail_list == []:
        fail_list.append("ALL TEST PASSED")
    else:
        print("Test failed")
    for i in fail_list:
        print(i)
    return

def check_fail_status(test_res_str: str) -> bool:
    if "FAILED" in test_res_str:
        fail_status = True
    else: 
        fail_status = False
    return fail_status
