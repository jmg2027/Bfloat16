Bfloat16

Bfloat16 python library

Development version: 3.8.10

Packages needed: tensorflow, numpy

Introduction:
Bfloat16 is the library modeling BF16/FP32 hardware floating point operations. It contains two types of floating point class: Bfloat16, Float32. Both classes include integer values of sign, exponent, mantissa of their own objects. Each class includes operations such as add, multiply, fused-multiply-add, sum-and-accumulate, etc. To describe hardware behavior and make it easy to interpret into Verilog, Bitstring class is used.

How to use:
from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
Refer to test.py

Details:
 Bfloat16(sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max)
BF16(Brain Floating Point) type class which supports for 8-bit exponent/7-bit mantissa.
No subnormal numbers are supported
