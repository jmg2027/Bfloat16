import bf16.bf16 as bf16

class FloatAddition:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        result_sign = self.a.sign ^ self.b.sign
        result_exponent = self.a.exponent - self.a.bias + self.b.exponent - self.b.bias
        print(result_exponent)
        result_mantissa = (self.a.mantissa + (1 << self.a.mantissa_bits)) * (self.b.mantissa + (1 << self.b.mantissa_bits))
        print(result_mantissa)

        # Handle rounding and normalization
        while result_mantissa >= (1 << (self.a.mantissa_bits + 1)):
            result_mantissa >>= 1
            result_exponent += 1

        result_mantissa &= (1 << self.a.mantissa_bits) - 1

        print(bin(result_mantissa))
        result = self.a.compose_float(result_sign, result_exponent, result_mantissa)
        return Bfloat16(result, self.a.mantissa_bits, self.a.exponent_bits)
