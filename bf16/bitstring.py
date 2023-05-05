import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BitString:
    """
    BitString class contains binary arithmetics
    Supports verilog type indexing and slicing
    You should specify bitwidth. If so, value is '0 or '1 for its sign
    Input: string of '01' or binary string (You may use bin() or f'{int:b}' or just string)
    For negative number, only stores 2's complement and not sign attribute
    Operation results are given in its full bitwidth. Handle overflow with set_bitwidth
    """
    """
    TBD:
    """
    def __init__(self, bitwidth: int, value: str = None) -> None:
        if value == None:
            value = '0'
        # Negative input check
        # Process 0b, -0b pattern. returns sequence of 0 and 1s 
        if value.startswith('-'):
            value = value[1:]
            value = self.twos_complement(self._binary_string_process(value))
        else:
            value = self._binary_string_process(value)
        self.set(bitwidth, value)

    def set_bitwidth(self, bitwidth: int):
        self.set(bitwidth, self.value)
    
    def set_bin(self, value: str):
        self.set(self.bitwidth, value)

    def set(self, bitwidth: int, value: str) -> None:
        value = self._binary_string_process(value)

        if len(value) > bitwidth:
            logger.warning(f'Value {value} Overflow. Truncating.')
            value = value[bitwidth:]
        elif len(value) < bitwidth:
            logger.warning(f'Value {value} is shorter than the specified bitwidth. Padding with zeros.')
            value = self._padding(bitwidth, value)
            
        if not all(p in '01' for p in value):
            raise ValueError('Bitstring should contain 0 and 1 or in binary string style only')
        
        self.bin = value
        self.bitwidth = bitwidth
        pass

    @classmethod
    def _padding(cls, bitwidth, value: 'str'):
        value = value.zfill(bitwidth)
        return value

    def _binary_string_process(self, value:str) -> str:
        if value.startswith('0b'):
            value = value[2:]
        return value

    def __len__(self):
        return len(self.bin)
    
    def __getitem__(self, index):
        """
        Supports Verilog type slicing and indexing
        [msb:lsb]
        """
        if isinstance(index, slice):
            py_start, py_stop, py_step = index.indices(len(self))
            start, stop, step = len(self.bin) - py_start - 1, len(self.bin) - py_stop, py_step
            return_bitstring = ''.join(self.bin[i] for i in range(start, stop, step)) 
            return BitString(len(return_bitstring), return_bitstring)
        elif isinstance(index, int):
            verilog_index = len(self.bin) - index - 1
            return self.bin[1, verilog_index]
        else:
            raise TypeError('Invalid index type')

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if value not in '01':
                raise ValueError('Bitstring should contain 0 and 1 only')
            self.bin = self.bin[:index] + value + self.bin[index + 1]
        else:
            raise TypeError('Invalid index type')

    # Operations
    def _sign_extend(self, value, new_width):
        sign_extension = self.sign * (new_width - len(value))
        return sign_extension + value

    def _apply_extension(self, other: 'BitString', operation) -> 'BitString':
        max_width = max(self.bitwidth, other.bitwidth)
        a = BitString(max_width, self._sign_extend(max_width, self.bin))
        b = BitString(max_width, self._sign_extend(max_width, other.bin))
        result = operation(a, b)
        return result

    # Methods
    @classmethod
    def add_binary(cls, bin1: str, bin2: str):
        max_len = max(len(bin1), len(bin2))
        bin1, bin2 = cls._padding(max_len, bin1), cls._padding(max_len, bin2)

        result = ""
        carry = 0

        for i in range(max_len - 1, -1, -1):
            bit_sum = int(bin1[i]) + int(bin2[i]) + carry

            if bit_sum == 3:
                result = "1" + result
                carry = 1
            elif bit_sum == 2:
                result = "0" + result
                carry = 1
            else:
                result = str(bit_sum) + result
                carry = 0

        if carry == 1:
            result = "1" + result

        return result
    
    @staticmethod
    def mul_binary(bin1: str, bin2: str):
        result = '0'
        for i, bit in enumerate(reversed(bin2)):
            if bit == '1':
                partial_product = f'{bin1}{"0"*i}'
                result = BitString.add_binary(result, partial_product)
        return result

    # Concatenation
    def concat(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Concatenation operation needs two Bitstring instances')
        return BitString(len(f'{self.bin}{other.bin}'), f'{self.bin}{other.bin}')

    # Unary operators
    def __neg__(self) -> 'BitString':
        """
        Two's complement
        """
        return ~self + BitString(self.bitwidth, '1')

    def __invert__(self) -> 'BitString':
        """
        NOT operation
        """
        return BitString(self.bitwidth, ''.join('1' if b == '0' else '0' for b in self.bin))
    
    # Bitwise
    def __and__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise AND operation needs two Bitstring instances')
        op = lambda a, b: BitString(max(self.bitwidth, other.bitwidth), bin(int(a) & int(b)))
        return self._apply_extension(other, op)

    def __or__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise OR operation needs two Bitstring instances')
        op = lambda a, b: BitString(max(self.bitwidth, other.bitwidth), bin(int(a) | int(b)))
        return self._apply_extension(other, op)

    def __xor__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise XOR operation needs two Bitstring instances')
        op = lambda a, b: BitString(max(self.bitwidth, other.bitwidth), bin(int(a) ^ int(b)))
        return self._apply_extension(other, op)
    
    # Comparison
    def __lt__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) < int(other)

    def __le__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) <= int(other)

    def __eq__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) == int(other)

    def __ne__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) != int(other)

    def __ge__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) >= int(other)

    def __gt__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Compare operation needs two Bitstring instances')
        return int(self) > int(other)

    # Shift
    def __lshift__(self, n: int):
        return BitString(self.bitwidth, self.bin[n:] + '0' * n)

    def __rshift__(self, n: int):
        return BitString(self.bitwidth, '0' * n + self.bin[:-n])

    def __ilshift__(self, n: int):
        return self.set_bin(self.bin[n:] + '0' * n)

    def __irshift__(self, n: int):
        return self.set_bin('0' * n + self.bin[:-n])

    def _operation_preprocess(self, other, operation):
        return

    def __add__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Addition operation needs two Bitstring instances')
        c = self.add_binary(self.bin, other.bin)
        return BitString(len(c), c)

    def __sub__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Subtraction operation needs two Bitstring instances')
        c = self.add_binary(self.bin, (-other).bin)
        return BitString(len(c), c)
    
    def __mul__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Subtraction operation needs two Bitstring instances')
        c = self.mul_binary(self.bin, other.bin)
        return BitString(len(c), c)
    
    # Formats
    def __int__(self):
        return int(self.bin, 2)

    def __hex__(self):
        return hex(int(self))

    @classmethod
    def from_int(cls, value, width, signed=False):
        binary_str = format(value, f'0{width}b')
#        return UnsignedBitString(binary_str, width) if not signed else SignedBitString(binary_str, width)
        return cls(binary_str, width) if not signed else SignedBitString(binary_str, width)

    @classmethod
    def from_hex(cls, value, width, signed=False):
        binary_str = format(int(value, 16), f'0{width}b')
#        return UnsignedBitString(binary_str, width) if not signed else SignedBitString(binary_str, width)
        return cls(binary_str, width) if not signed else SignedBitString(binary_str, width)
    
    @classmethod
    def int_to_bit(cls, value: int) -> 'BitString':
        """
        For use of unsigned integer
        """
        if value < 0:
            raise ValueError('int_to_bit method should not be used with negative integer')
        return BitString(bin(value)[2:])

    @staticmethod
    def twos_complement(value: str) -> str:
        inverted = ''.join('1' if bit == '0' else '0' for bit in value)
        inverted_int = int(inverted, 2) + 1
        return bin(inverted_int)[2:]

    def __repr__(self):
        return f'bit: {self.bin}'
    
    def __str__(self):
        return self.bin


    
class SignedBitString(BitString):
    """
    To contain signed or unsigned attribute, use SignedBitString
    bitwidth must be given explicit
    """
    def __init__(self, bitwidth: int, value: str) -> None:
        super().__init__(bitwidth, value)
        self.sign = self.bin[0]

    @classmethod
    def _padding(cls, bitwidth, value):
        if value[0] == '1':
            value = '1' * (bitwidth - len(value)) + value
        elif value[0] == '0':
            value = value.zfill(bitwidth)
        return value
    
    def _sign_extend(self, value, new_width):
        sign_extension = self.sign * (new_width - len(value))
        return sign_extension + value

    def _apply_extension(self, other, operation):
        max_width = max(self.bitwidth, other.bitwidth)
        a = SignedBitString(max_width, self._sign_extend(max_width, self.bin))
        b = SignedBitString(max_width, self._sign_extend(max_width, other.bin))
        result = operation(a, b)
        if len(result.bin) > max_width:
            logger.warning(f'{a} and {b} Overflow occurred in SignedBitString operation.')
            result.bin = result.bin[-max_width:]
        return result

    # Formats
    def __int__(self):
        if self.sign:
            return int(f'-{self.bin}', 2)
        else:
            return int(self.bin, 2)

    def __hex__(self):
        return hex(int(self))



class UnsignedBitString(BitString):
    def __init__(self, bitwidth, value):
        if value == None:
            value = '0'
        # Negative input check
        if value.startswith('-'):
            raise ValueError('UnsignedBitString cannot handle negative values.')
        super().__init__(bitwidth, value)

    def _apply_extension(self, other, operation):
        max_width = max(self.bitwidth, other.bitwidth)
        a = UnsignedBitString(max_width, self.bin.zfill(max_width))
        b = UnsignedBitString(max_width, other.bin.zfill(max_width))
        result = operation(a, b)
        if len(result.bin) > max_width:
            logger.warning(f'{a} and {b} Overflow occurred in UnsignedBitString operation.')
            result.bin = result.bin[-max_width:]
        return result

    # Override binary arithmetic operations if needed

if __name__ == '__main__':
    import random

    def assert_equal(expected, actual, test_name):
        if expected != actual:
            print(f"{test_name}: FAIL (Expected {expected}, got {actual})")
        else:
            print(f"{test_name}: PASS")


    def test_bitstring_classes():
        # Test cases
        test_cases = [
            {'width': 4, 'value1': '1101', 'value2': '0101'},
            {'width': 8, 'value1': '10011011', 'value2': '01100101'},
            {'width': 16, 'value1': '1101101011011010', 'value2': '0101010101010101'}
        ]

        for test_case in test_cases:
            width = test_case['width']
            value1 = test_case['value1']
            value2 = test_case['value2']

            # Test BitString class
            bitstring1 = BitString(bitwidth=width, value=value1)
            bitstring2 = BitString(bitwidth=width, value=value2)

            assert_equal(value1, bitstring1.bin, "BitString init")
            assert_equal(value2, bitstring2.bin, "BitString init")

            # Test SignedBitString class
            signed_bitstring1 = SignedBitString(bitwidth=width, value=value1)
            signed_bitstring2 = SignedBitString(bitwidth=width, value=value2)

            assert_equal(value1, signed_bitstring1.bin, "SignedBitString init")
            assert_equal(value2, signed_bitstring2.bin, "SignedBitString init")

            # Test UnsignedBitString class
            unsigned_bitstring1 = UnsignedBitString(bitwidth=width, value=value2)
            unsigned_bitstring2 = UnsignedBitString(bitwidth=width, value=value2)

            assert_equal(value2, unsigned_bitstring1.bin, "UnsignedBitString init")
            assert_equal(value2, unsigned_bitstring2.bin, "UnsignedBitString init")

            # Test operations for each class
            for class_name, class_instance1, class_instance2 in [("BitString", bitstring1, bitstring2),
                                                                ("SignedBitString", signed_bitstring1, signed_bitstring2),
                                                                ("UnsignedBitString", unsigned_bitstring1, unsigned_bitstring2)]:

                add_result = class_instance1 + class_instance2
                sub_result = class_instance1 - class_instance2
                mul_result = class_instance1 * class_instance2

                add_expected_result = bin(int(value1, 2) + int(value2, 2))[2:].zfill(width)
                sub_expected_result = bin(int(value1, 2) - int(value2, 2))[2:].zfill(width)
                mul_expected_result = bin(int(value1, 2) * int(value2, 2))[2:].zfill(width)

                assert_equal(add_expected_result, add_result.bin, f"{class_name} Add")
                assert_equal(sub_expected_result, sub_result.bin, f"{class_name} Sub")

                assert_equal(mul_expected_result, mul_result.bin, f"{class_name} Mul")

    test_bitstring_classes()