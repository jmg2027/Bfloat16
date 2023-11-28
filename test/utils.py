import tensorflow as tf
import random
from jaxtyping import BFloat16 as tfbfloat16
from jaxtyping import Float32 as tffloat32
from jaxtyping import Float64 as tffloat64

from .commonimport import bf16, fp32, bf16_obj, fp32_obj, FloatBase, bf16_config, fp32_config

from float_class.utils import bf16_ulp_dist, fp32_ulp_dist
from typing import Generic, TypeVar, Union, Type

from float_class.floatclass import bf16_to_fp32, fp32_to_bf16


FloatBaseT = Union[fp32, bf16]
TestInputT = Union[int, float, bf16, fp32]

def conv_from_float(f: float, ftype: Type[FloatBaseT]) -> FloatBaseT:
    ftype_obj: FloatBaseT = ftype(0, 0, 0)
    return ftype_obj.from_float(f)

def convert_to_bf16(num: float) -> bf16:
    return bf16_obj.from_float(num)

def convert_to_fp32(num: float) -> fp32:
    return fp32_obj.from_float(num)

def conv_to_tf_dtype(num: float, ftype: Type[FloatBaseT] = fp32) -> Union[tfbfloat16, tffloat32]:
    '''
    num: float, ftype: Type[FloatBaseT] -> tf.bfloat16, tf.float32
    '''
    if ftype == bf16:
        return convert_to_tfbf16(num)
    elif ftype == fp32:
        return convert_to_tffp32(num)
    else:
        raise TypeError("Use this function only for Bfloat16 and Float32 type")

def conv_from_tf_to_tf_dtype(num: tffloat64, ftype: Type[FloatBaseT] = fp32) -> Union[tfbfloat16, tffloat32]:
    '''
    num: float, ftype: Type[FloatBaseT] -> tf.bfloat16, tf.float32
    '''
    if ftype == bf16:
        return convert_from_tf_to_tfbf16(num)
    elif ftype == fp32:
        return convert_from_tf_to_tffp32(num)
    else:
        raise TypeError("Use this function only for Bfloat16 and Float32 type")

def convert_from_tf_to_tfbf16(num: tffloat64) -> tfbfloat16:
    return tf.cast(num, tf.bfloat16)

def convert_from_tf_to_tffp32(num: tffloat64) -> tffloat32:
    return tf.cast(num, tf.float32)

def convert_to_tfbf16(num: float) -> tfbfloat16:
    return tf.cast(float(num), tf.bfloat16)

def convert_to_tffp32(num: float) -> tffloat32:
    return tf.cast(float(num), tf.float32)

def convert_to_tffp64(num: float) -> tffloat64:
    return tf.cast(float(num), tf.float64)

def convert_tfbf16_to_int(num: tfbfloat16) -> int:
    return int(num)

def convert_int_to_tfbf16(num: int) -> tfbfloat16:
    return tf.cast(num, tf.bfloat16)

def cast_float(frepr: TestInputT, \
                ftype: Type[FloatBaseT] = fp32) \
                -> FloatBaseT:
    # use float representation for integer inputs
    # integer inputs are used to hex input
    ftype_obj: FloatBaseT = ftype(0, 0, 0)
    if isinstance(frepr, int) &  (ftype == fp32):
        # integer/hex repr
        return ftype_obj.from_hex(frepr & 0xFFFF_FFFF)
    elif isinstance(frepr, int) &  (ftype == bf16):
        # integer/hex repr
        return ftype_obj.from_hex(frepr & 0xFFFF_0000 >> 16)
    elif isinstance(frepr, float):
        # python float repr
        return ftype_obj.from_float(frepr)
    elif isinstance(frepr, ftype):
        # need not to cast
        return frepr
    elif isinstance(frepr, bf16) & (ftype == fp32):
        # bf16 -> fp32
        return bf16_to_fp32(frepr)
    elif isinstance(frepr, fp32) & (ftype == bf16):
        # bf16 -> fp32
        return fp32_to_bf16(frepr)
    else:
        raise TypeError(f"Supported float representations are: hex(int), float, bf16, fp32. Current input: {frepr}")
        

def check_float_equal(res1: Union[bf16, fp32], res2) -> bool:
    # nan cannot be compared
    ulp_error = calc_ulp_error(res1, res2)
    # ulp error under 2
    if ulp_error <= 1:
        return True
    else:
        return False

def calc_ulp_error(res1: Union[bf16, fp32], res2):
    # nan cannot be compared
    if (str(float(res1)) == 'nan') & (str(float(res2)) == 'nan'):
        ulp_error = 0
    elif isinstance(res1, bf16):
        ulp_error = bf16_ulp_dist((res1), float(res2))
    elif isinstance(res1, fp32):
        ulp_error = fp32_ulp_dist((res1), float(res2))
    # I won't mind -0.0 and 0.0
    elif (float(res1) == 0.0) & (float(res2) == -0.0):
        ulp_error = 0
    elif (float(res1) == -0.0) & (float(res2) == 0.0):
        ulp_error = 0
    else:
        raise TypeError(f"Input type should be bf16 or fp32. Current input: {type(res1)}")
    return ulp_error

def random_fp(ftype, exp_min: int = -bf16_config.exp_max + 1, exp_max: int = bf16_config.exp_max):
    if ftype == bf16:
        return random_bf16(exp_min, exp_max)
    elif ftype == fp32:
        return random_fp32(exp_min, exp_max)
    else:
        raise TypeError("Use this function only for Bfloat16 and Float32 type")

def random_bf16(exp_min: int = -10, exp_max: int = 10) -> bf16:
    min_exp = exp_min 
    max_exp = exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = bf16_config.mant_max
    if rand_exp == -127:
        rand_mant = min_mant
    else:
        rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return bf16(rand_sign, rand_exp, rand_mant)

def random_fp32(exp_min: int = -10, exp_max: int = 10) -> fp32:
    min_exp = exp_min
    max_exp = exp_max
    rand_exp = random.randint(min_exp, max_exp)
    min_mant = 0
    max_mant = fp32_config.mant_max
    if rand_exp == -127:
        rand_mant = min_mant
    else:
        rand_mant = random.randint(min_mant, max_mant)
    rand_sign = random.randint(0,1)
    return fp32(rand_sign, rand_exp, rand_mant)

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
