class BitString:
    """
    BitString class contains binary arithmetics
    Supports verilog type indexing and slicing
    Input: string of '01' or binary string
    For negative number, - should be headed to value
    """
    """
    TBD:
    Support for hex input
    Should this class support negative numbers?
    """
    def __init__(self, value: str, bitwidth: int = None) -> None:
        # Preprocess for binary string
        if self._check_bin_style(value):
            value = value[2:]
        if not all(p in '01' for p in value):
            raise ValueError('Bitstring should contain 0 and 1 or in binary string style only')
        if bitwidth is None:
            self._bin = value
        elif bitwidth < len(value):
            raise ValueError('Bitwidth should be greater than or equal to the length of the value')
        else:
            self._bin = "0" * (bitwidth - len(value)) + value
    
    def _check_bin_style(self, value):
        return value[0:2] == '0b'

    def __len__(self):
        return len(self._bin)
    
    def __getitem__(self, index):
        """
        Supports Verilog type slicing and indexing
        [msb:lsb]
        """
        if isinstance(index, slice):
            py_start, py_stop, py_step = index.indices(len(self))
            start, stop, step = len(self._bin) - py_start - 1, len(self._bin) - py_stop, py_step
            return BitString(''.join(self._bin[i] for i in range(start, stop, step)))
        elif isinstance(index, int):
            verilog_index = len(self._bin) - index - 1
            return self._bin[verilog_index]
        else:
            raise TypeError('Invalid index type')

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if value not in '01':
                raise ValueError('Bitstring should contain 0 and 1 only')
            self._bin = self._bin[:index] + value + self._bin[index + 1]
        else:
            raise TypeError('Invalid index type')

    def __neg__(self) -> 'BitString':
        """
        Two's complement
        """
        return ~self + BitString('1')

    def __invert__(self) -> 'BitString':
        """
        NOT operation
        """
        return BitString(''.join('1' if b == '0' else '0' for b in self._bin))

    # Operations
    # Bitwise
    def __and__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise AND operation needs two Bitstring instances')
        max_len = max(len(self), len(other))
        return BitString(bin(int(self) & int(other)), max_len)

    def __or__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise OR operation needs two Bitstring instances')
        max_len = max(len(self), len(other))
        return BitString(bin(int(self) | int(other)), max_len)

    def __xor__(self, other: 'BitString') -> 'BitString':
        if not isinstance(other, BitString):
            raise TypeError('Bitwise XOR operation needs two Bitstring instances')
        max_len = max(len(self), len(other))
        return BitString(bin(int(self) ^ int(other)), max_len)
    
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
    def __lshift__(self, other):
        return

    def __rshift__(self, other):
        return

    def __ilshift__(self, other):
        return

    def __irshift__(self, other):
        return
    



    def __add__(self, other: 'BitString') -> 'BitString':
#        result = self.add_binary(self._bin, other._bin)
        return BitString(bin(int(self) + int(other)))

    def __sub__(self, other: 'BitString') -> 'BitString':
        """
        This does not work properly
        TBD
        """
#        other_neg = -other
#        result = self.add_binary(self._bin, other_neg._bin)
        return BitString(bin(int(self) - int(other)))
    
    def __mul__(self, other: 'BitString') -> 'BitString':
        result = self.mul_binary(self._bin, other._bin)
        return BitString(result)
    
    # Methods
    @staticmethod
    def add_binary(bin1, bin2):
        max_len = max(len(bin1), len(bin2))
        bin1, bin2 = bin1.zfill(max_len), bin2.zfill(max_len)

        result = list()
        carry = 0

        for i in range(max_len - 1, -1, -1):
            sum = int(bin1[i]) + int(bin2[i]) + carry
            carry, bit = divmod(sum, 2)
            result.insert(0, str(bit))
        if carry:
            result.insert(0, str(carry))
        return ''.join(result)
    
    @staticmethod
    def mul_binary(bin1, bin2):
        result = '0'
        for i, bit in enumerate(reversed(bin2)):
            if bit == '1':
                partial_product = f'{bin1}{"0"*i}'
                result = BitString.add_binary(result, partial_product)
        return result

    # Formats
    def __int__(self):
        return int(self._bin, 2)
    
    @classmethod
    def int_to_bit(cls, value: int) -> 'BitString':
        """
        For use of unsigned integer
        """
        if value < 0:
            raise ValueError('int_to_bit method should not be used with negative integer')
        return BitString(bin(value)[2:])

    def __repr__(self):
        return f'bit: {self._bin}'
    
    def __str__(self):
        return self._bin

#a = BitString('1010')
#b = BitString('1010')
#print(int(a+b))
#print(int(a*b))
#print(a)
#print(BitString.int_to_bit(3))
#print(~a)
#print(-a)
#print(a-b)
#c = BitString(bin(30))
#d = BitString('100010')
#print(c&d)
#print(a<b)
#print(c<d)
#print(c>d)
#print(c==d)
#print(a==b)
a = BitString('1010')
print(-a)
print(a+(-a))
