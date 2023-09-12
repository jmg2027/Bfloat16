import logging

from abc import ABCMeta, abstractmethod
from typing import(
    Union,
    Type,
    TypeVar,
    List,
    Tuple,
    Dict,
    Callable,
    Any,
    Optional,
    Literal,
    Generic,
)
from typing_extensions import Self, TypeAlias

BitStrT = TypeVar('BitStrT', bound='BitString')

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class BitString:
    """
    BitString(bitwidth: int, value: str)\n
    BitString class contains binary arithmetics\n
    Supports verilog type indexing and slicing\n
    You should specify bitwidth. If so, value is '0 or '1 for its sign\n
    Input: string of '01' or binary string (You may use bin() or f'{int:b}' or just string)\n
    For negative number, only stores 2's complement and not sign attribute\n
    Operation results are given in its full bitwidth. Handle overflow with set_bitwidth
    BitString is same as UnsignedBitString
    """
    
    def __init__(self, bitwidth: int, value: str) -> None:
        self.set(bitwidth, value)

    def set_bitwidth(self, bitwidth: int) -> None:
        if not (isinstance(bitwidth, int) & (bitwidth > 0)):
            raise BitStrTypeError("INVALID_BITWIDTH", (self, bitwidth))
        self.set(bitwidth, self.value) # type: ignore
    
    def set_bin(self, value: str) -> None:
        if not isinstance(value, str):
            raise BitStrTypeError("INVALID_VALUE", (self, value))
        self.set(self.bitwidth, value)

    def set(self, bitwidth: int, value: str) -> None:
        # Negative input check
        # Process 0b, -0b pattern. returns sequence of 0 and 1s 
        is_neg = False
        if value.startswith('-'):
            value = self._binary_string_process(value[1:])
            # For sign bit
            value = '0' + value
            value = self.twos_complement(self._binary_string_process(value))
            is_neg = True
        else:
            value = self._binary_string_process(value)

        if len(value) > bitwidth:
            logger.warning(f'Value {value} Overflow. Truncating.')
            value = value[-bitwidth:]
        elif len(value) < bitwidth:
            logger.warning(f'Value {value} is shorter than the specified bitwidth. Padding rest bits.')
            if is_neg:
                value = self._padding(bitwidth, value, '1')
            else:
                value = self._padding(bitwidth, value, '0')
            
        if not all(p in '01' for p in value):
            raise BitStrValueError('INVALID_BIT', value)
        
        self.bin: str = value
        self.bitwidth: int = bitwidth
        pass

    @classmethod
    def _padding(cls, bitwidth: int, value: str, pad: str) -> str:
        value = pad * (bitwidth - len(value)) + value
#        print(f'{value} Padding to {pad}')
        return value

    def _binary_string_process(self, value:str) -> str:
        if value.startswith('0b'):
            value = value[2:]
        return value

    def __len__(self) -> int:
        return len(self.bin)
    
    def __getitem__(self, index: Union[slice, int]) -> Self:
        """
        Supports Verilog type slicing and indexing
        [msb:lsb]
        """
        if isinstance(index, slice):
            py_start, py_stop, py_step = index.indices(len(self))
            start, stop, step = len(self.bin) - py_start - 1, len(self.bin) - py_stop, py_step
            return_bitstring = ''.join(self.bin[i] for i in range(start, stop, step)) 
            return self.__class__(len(return_bitstring), return_bitstring)
        elif isinstance(index, int):
            verilog_index = len(self.bin) - index - 1
            return_bitstring = self.bin[verilog_index: verilog_index + 1]
            return self.__class__(1, return_bitstring)
        else:
            raise BitStrTypeError('INVALID_INDEX', (self, index))

    def __setitem__(self, index: int, value: str) -> None:
        if isinstance(index, int):
            if value not in '01':
                raise BitStrValueError('INVALID_BIT', value)
            self.bin = self.bin[:index] + value + self.bin[index + 1]
        else:
            raise BitStrTypeError('INVALID_INDEX', (self, index))

    # Operations
    def _sign_extend(self, new_width: int, value: str) -> str:
        sign_extension: str = self.bin[0] * (new_width - len(value))
        return sign_extension + value

    def _apply_extension(self, other: Self, operation: Callable[[Self, Self], Self], result_bitwidth: int) -> Self:
        a = self.__class__(result_bitwidth, self._sign_extend(result_bitwidth, self.bin))
        b = self.__class__(result_bitwidth, self._sign_extend(result_bitwidth, other.bin))
        result: Self = operation(a, b)
        if len(result.bin) > result_bitwidth:
            logger.warning(f'{a} and {b} Overflow occurred in BitString operation.')
        sign_extended_result: str = self._sign_extend(result_bitwidth, result.bin[-result_bitwidth:])
        c = self.__class__(result_bitwidth, sign_extended_result)
        return c

    # Methods
    # Full adder
    @classmethod
    def add_binary(cls, bin1: str, bin2: str) -> str:
        max_len: int = max(len(bin1), len(bin2))
        carry = 0
        result = ''
        a, b = cls._padding(max_len, bin1, bin1[0]), cls._padding(max_len, bin2, bin2[0])

        for i in range(max_len - 1, -1, -1):
            current_bit_a = int(a[i])
            current_bit_b = int(b[i])

            sum_bit: int = current_bit_a ^ current_bit_b ^ carry
            carry = (current_bit_a & current_bit_b) | (current_bit_b & carry) | (carry & current_bit_a)

            result = str(sum_bit) + result
        
        if carry:
            result = '1' + result
        else:
            result = '0' + result
        return result

    @classmethod
    def carry_sum_add_binary(cls, a: str, b: str, c: str) -> Tuple[str, str]:
        sum = ''
        carry = '0'
        max_len: int = max(len(a), len(b), len(c))

        for i in range(max_len - 1, -1, -1):
            current_bit_a = int(a[i])
            current_bit_b = int(b[i])
            current_bit_c = int(c[i])

            sum_bit = current_bit_a ^ current_bit_b ^ current_bit_c
            carry_bit = (current_bit_a & current_bit_b) | (current_bit_b & current_bit_c) | (current_bit_c & current_bit_a)

            sum = str(sum_bit) + sum
            carry = str(carry_bit) + carry

        return sum, carry

    @classmethod
    def carry_sum_add(cls: Type[Self], a: Self, b: Self, c: Self) -> Tuple[Self, Self]:
        sum_bin, carry_bin = cls.carry_sum_add_binary(a.bin, b.bin, c.bin)
        # extend sum bit
        sum: Self = cls(len(carry_bin), sum_bin)
        carry: Self = cls(len(carry_bin), carry_bin)
        return sum, carry

    @classmethod
    def add_bitstring(cls: Type[Self], bit1: Self, bit2: Self) -> Self:
        result_bin = cls.add_binary(bit1.bin, bit2.bin)
        bitstring: Self = cls(len(result_bin), result_bin)
        return bitstring
    
    @classmethod
    def mul_bitstring(cls: Type[Self], bit1: Self, bit2: Self) -> Self:
        bin1: str = bit1.bin
        bin2: str = bit2.bin
        result = '0'

        for i, bit in enumerate(reversed(bin2)):
            if bit == '1':
                partial_product: str = f'{bin1}{"0" * i}'
                result = cls.add_binary(result, partial_product)

        return cls(len(bin1) + len(bin2), result)

    # Concatenation
    def concat(self, other: Self) -> Self:
        if not isinstance(other, BitString):
            raise BitStrTypeError('TWO_BITSTRING', (self, other))
        if not (type(self) == type(other)):
            raise BitStrTypeError('SAME_CLASS', (self, other))
        return self.__class__(len(f'{self.bin}{other.bin}'), f'{self.bin}{other.bin}')

    # Unary operators
    def __neg__(self) -> Self:
        """
        Two's complement
        """
        return ~self + self.__class__(self.bitwidth, '1')

    def __invert__(self) -> Self:
        """
        NOT operation
        """
        return self.__class__(self.bitwidth, ''.join('1' if b == '0' else '0' for b in self.bin))
    
    # Reduction operators
    def reduceand(self) -> Self:
        result: Self = self[0]
        for i in range(self.bitwidth):
            result = result & self[i]
        return result

    def reduceor(self) -> Self:
        result: Self = self[0]
        for i in range(self.bitwidth):
            result = result | self[i]
        return result

    def reducexor(self) -> Self:
        result: Self = self[0]
        for i in range(self.bitwidth):
            result = result ^ self[i]
        return result

    def _check_cond_operation(self, other: Self) -> None:
        self._check_two_bitstring(other)
        self._check_same_class(other)
        pass

    def _check_two_bitstring(self, other: Self) -> None:
        if not isinstance(other, BitString):
            raise BitStrTypeError('TWO_BITSTRING', (self, other))
        pass
    
    def _check_same_class(self, other: Self) -> None:
        if not (type(self) == type(other)):
            raise BitStrTypeError('SAME_CLASS', (self, other))
        pass

    # Bitwise
    def __and__(self, other: Self) -> Self:
        self._check_cond_operation(other)
        result_bitwidth: int = max(self.bitwidth, other.bitwidth)
        op: Callable[[Self, Self], Self] = lambda a, b: self.__class__(result_bitwidth, bin(int(a) & int(b)))
        return self._apply_extension(other, op, result_bitwidth)

    # JMG: FIX
    def __or__(self, other: Self) -> Self:
        self._check_cond_operation(other)
        result_bitwidth: int = max(self.bitwidth, other.bitwidth)
        op: Callable[[Self, Self], Self] = lambda a, b: self.__class__(result_bitwidth, bin(int(a) | int(b)))
        return self._apply_extension(other, op, result_bitwidth)

    def __xor__(self, other: Self) -> Self:
        self._check_cond_operation(other)
        result_bitwidth: int = max(self.bitwidth, other.bitwidth)
        op: Callable[[Self, Self], Self]= lambda a, b: self.__class__(result_bitwidth, bin(int(a) ^ int(b)))
        return self._apply_extension(other, op, result_bitwidth)
    
    # Comparison
    def __lt__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) < int(other)

    def __le__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) <= int(other)

    def __eq__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) == int(other)

    def __ne__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) != int(other)
        #return self.__class__(1, bin(int(self) != int(other)))

    def __ge__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) >= int(other)

    def __gt__(self, other: Self) -> bool:
        self._check_cond_operation(other)
        return int(self) > int(other)

    # Shift
    def __lshift__(self, n: int) -> Self:
        if n > self.bitwidth:
            return self.__class__(self.bitwidth, '0')
        else:
            return self.__class__(self.bitwidth, self.bin[n:] + '0' * n)

    def __rshift__(self, n: int) -> Self:
        if n > self.bitwidth:
            return self.__class__(self.bitwidth, '0')
        # string [:-0] returns '', so - slicing should not be used
        return self.__class__(self.bitwidth, '0' * n + self.bin[0:self.bitwidth-n])

    def arith_rshift(self, n: int) -> Self:
        if n > self.bitwidth:
            return self.__class__(self.bitwidth, self.bin[0] * self.bitwidth)
        # string [:-0] returns '', so - slicing should not be used
        return self.__class__(self.bitwidth, self.bin[0] * n + self.bin[0:self.bitwidth-n])

    def __ilshift__(self, n: int) -> None:
        if n > self.bitwidth:
            return self.set_bin('0' * self.bitwidth)
        else:
            return self.set_bin((self.bin[n:] + '0' * n))

    def __irshift__(self, n: int) -> None:
        if n > self.bitwidth:
            return self.set_bin('0' * self.bitwidth)
        return self.set_bin('0' * n + self.bin[0:self.bitwidth-n])

    def __add__(self, other: Self) -> Self:
        self._check_two_bitstring(other)
        result_bitwidth = max(self.bitwidth, other.bitwidth)
        op: Callable[[Self, Self], Self]= lambda a, b: self.add_bitstring(a, b)
        return self._apply_extension(other, op, result_bitwidth)

    def __sub__(self, other: Self) -> Self:
        self._check_two_bitstring(other)
        result_bitwidth: int = max(self.bitwidth, other.bitwidth)
        op: Callable[[Self, Self], Self] = lambda a, b: self.add_bitstring(a, -b)
        return self._apply_extension(other, op, result_bitwidth)
    
    def __mul__(self, other: Self) -> Self:
        self._check_two_bitstring(other)
        result_bitwidth: int = max(self.bitwidth, other.bitwidth) * 2
        op: Callable[[Self, Self], Self] = lambda a, b: self.mul_bitstring(a, b)
        return self._apply_extension(other, op, result_bitwidth)
    
    # Formats
    def __int__(self) -> int:
        return int(self.bin, 2)

    def __hex__(self) -> str:
        return hex(int(self))

    @classmethod
    def from_int(cls: Type[Self], width: int, value: int, signed: bool = False) -> Union[Self, 'SignedBitString']:
        binary_str: str = format(value, f'0{width}b')
        return cls(width, binary_str) if not signed else SignedBitString(width, binary_str)

    @classmethod
    def from_hex(cls: Type[Self], width: int, value: str, signed: bool = False) -> Union[Self, 'SignedBitString']:
        binary_str: str = format(int(value, 16), f'0{width}b')
        return cls(width, binary_str) if not signed else SignedBitString(width, binary_str)
    
    '''
    @classmethod
    def int_to_bit(cls, value: int) -> 'BitString':
        """
        For use of unsigned integer
        """
        if value < 0:
            raise BitStrValueError('int_to_bit method should not be used with negative integer')
        return BitString(bin(value)[2:])
    '''

    @staticmethod
    def twos_complement(value: str) -> str:
        inverted: str = ''.join('1' if bit == '0' else '0' for bit in value)
        inverted_int: int = int(inverted, 2) + 1
        return bin(inverted_int)[2:]

    def __repr__(self) -> str:
        return f'[{self.bitwidth-1}:0] {self.bin}'
    
    def __str__(self) -> str:
        return self.bin


class SignedBitString(BitString):
    """
    To contain signed or unsigned attribute, use SignedBitString
    bitwidth must be given explicit
    """
    """
    TBD:
    bin(1) is now considered as -1. Need to fix this
    """
    def __init__(self, bitwidth: int, value: str) -> None:
        super().__init__(bitwidth, value)
        self.sign: str = self.bin[0]
    
    def _sign_extend(self, new_width, value) -> str:
        sign_extension: str = self.sign * (new_width - len(value))
        return sign_extension + value

    # Absolute value
    def __abs__(self) -> None:
        return

    # Formats
    def __int__(self) -> int:
        if self.sign == '1':
            return int(f'-{self.twos_complement(self.bin)}', 2)
        else:
            return int(self.bin, 2)

    def __hex__(self) -> str:
        return hex(int(self))

UnsignedBitString: TypeAlias = BitString


class BitStrTypeError(Exception):
    ERRORS: Dict[str, str] = {
        'TWO_BITSTRING': 'Operation needs two BitString instances.',
        'SAME_CLASS': 'Operation needs two instances with the same class.',
        'INVALID_TYPE': 'Invalid type for the operation.',
        'INVALID_BITWIDTH': 'Bitstring bitwidth should be a positive integer.',
        'INVALID_VALUE': 'Bitstring value should be a string.',
        'INVALID_INDEX': 'Invalid index type for BitString.',
    }
    def __init__(self, error_type: str, value = None) -> None:
        if error_type in self.ERRORS:
            self.message = f'{self.ERRORS[error_type]}. Got: {value}'
        else:
            self.message = f'{error_type}. Got: {value}'
        super().__init__(self.message)


class BitStrValueError(Exception):
    ERRORS: Dict[str, str] = {
        'INVALID_CHARS': 'Bitstring should contain 0 and 1 or in binary string style only.',
        'INVALID_BIT': 'Bitstring should contain 0 and 1 only.',
    }

    def __init__(self, error_type: str, value = None) -> None:
        if error_type in self.ERRORS:
            self.message = f'{self.ERRORS[error_type]}. Got: {value}'
        else:
            self.message = f'{error_type}. Got: {value}'
        super().__init__(self.message)