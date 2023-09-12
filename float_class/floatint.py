from .bitstring import BitString as bit
from .bitstring import SignedBitString as sbit
from .bitstring import UnsignedBitString as ubit

from typing import Tuple, Dict
from typing_extensions import Self


class _FPConfig:
    __slots__ = ('sign_bitpos', 'exponent_bits', 'mantissa_bits',
                 'bias', 'exp_max', 'mant_max')

    def __init__(self, bw: int, e: int, m: int):
        self.sign_bitpos = bw
        self.exponent_bits = e
        self.mantissa_bits = m
        self.bias = (1 << (e - 1)) - 1
        self.exp_max = 1 << (e - 1)
        self.mant_max = (1 << m) - 1
        pass

_fp64_config = _FPConfig(64, 11, 52)
_fp32_config = _FPConfig(32, 8, 23)
_bf16_config = _FPConfig(16, 8, 7)

# For external usage
fp64_config = _fp64_config
fp32_config = _fp32_config
bf16_config = _bf16_config


class FloatBaseInt:
    '''
    Internal variable structure of FloatBase
    Used in hw_model
    Contains _FPConfig, sign, exponent, mantissa, iszero, isinf, isnan, isden, isoverflow, isunderflow
    Compose, decompose
    '''
    def __init__(self, sign: int, exponent: int, mantissa: int,     
                config: _FPConfig) -> None:
        # call setter in __init__
        self._config = config
        self.set(sign, exponent, mantissa)

    def set(self, sign: int, exponent: int, mantissa: int) -> None:
        self.sign: int = self.set_sign(sign)
        self.exponent: int = self.set_exponent(exponent)
        self.mantissa: int = self.set_mantissa(mantissa)

    def set_sign(self, sign: int) -> int:
        if not (sign == 0 or sign == 1):
            raise FloatValueError(f"Sign value out of range", value = sign, constraint = "0 or 1")
        return sign

    def set_exponent(self, exponent: int) -> int:
        if not (0 - self._config.bias) <= exponent <= self._config.exp_max:
            raise FloatValueError(f"Exponent value out of range", value = exponent, constraint = f"{-self._config.bias} ~ {self._config.exp_max}")
        return exponent

    def set_mantissa(self, mantissa: int) -> int:
        if not 0 <= mantissa <= self._config.mant_max:
            raise FloatValueError(f"Mantissa value out of range", value=mantissa, constraint=f"0 ~ {self._config.mant_max}")
        return mantissa
    
    def check_config(self) -> str:
        if self._config == _bf16_config:
            return 'bf16'
        elif self._config == _fp32_config:
            return 'fp32'
        else:
            raise FloatTypeError("FloatBaseInt should have _FPConfig")

    def check_type(self, fp_type: str) -> bool:
        if fp_type == 'bf16':
            return self._config == _bf16_config
        elif fp_type == 'fp32':
            return self._config == _fp32_config
        else:
            raise FloatTypeError("Float config supports bf16 or fp32 only", value = fp_type, expected_type = "bf16 or fp32")

    
class FloatTypeError(Exception):
    ERRORS: Dict[str, str] = {
        'INVALID_OPERAND': 'Operands should be Bfloat16/Float32 objects.',
        'TYPE_MISMATCH': 'Operands should be of the same type.',
        'INVALID_INT': 'The integer should be 32-bit.',
        'INVALID_FLOAT': 'Invalid float type for the operation.',
    }
    def __init__(self, error_type: str, value = None, expected_type = None) -> None:
        if error_type in self.ERRORS:
            self.message = f'{self.ERRORS[error_type]}. Got: {value}, Expected: {expected_type}'
        else:
            self.message = f'{error_type}. Got: {value}, Expected: {expected_type}'
        super().__init__(self.message)


class FloatValueError(Exception):
    ERRORS: Dict[str, str] = {
        'OVERFLOW': 'Overflow occurred.',
        'UNDERFLOW': 'Underflow occurred.',
        'INVALID_EXPONENT': 'Invalid exponent value.',
        'INVALID_MANTISSA': 'Invalid mantissa value.',
    }

    def __init__(self, error_type: str, value = None, constraint = None) -> None:
        if error_type in self.ERRORS:
            self.message = f'{self.ERRORS[error_type]}. Got: {value}, Constraint: {constraint}'
        else:
            self.message = f'{error_type}. Got: {value}, Constraint: {constraint}'
        super().__init__(self.message)


fp32_int = FloatBaseInt(0, 0, 0, _fp32_config)
bf16_int = FloatBaseInt(0, 0, 0, _bf16_config)