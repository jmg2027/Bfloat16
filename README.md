# Bfloat16

## Bfloat16 python library

### Development version
 3.8.10

### Dependencies
 tensorflow, numpy

### Introduction
 Bfloat16 is the library modeling BF16/FP32 hardware floating point operations. It contains two types of floating point class: **Bfloat16**, **Float32**. Both classes include integer values of sign, exponent, mantissa of their own objects. Each class includes operations such as add, multiply, fused-multiply-add, sum-and-accumulate, etc. To describe hardware behavior and make it easy to interpret into Verilog, **Bitstring** class is used.
 **Caution:** There's some 1~2 ulp error

### How to use
```python
>>> from bf16.bf16 import Bfloat16 as bf16
>>> from bf16.bf16 import Float32 as fp32
>>> A = bf16(1, -10, 100)
>>> B = bf16(0, 5, 35)
>>> print(A + B)
Bfloat16(40.75, sign = 0, exponent=5, mantissa=35)
```
```python
>>> A = bf16.float_to_bf16(-1.32)
>>> B = bf16.float_to_bf16(3.156)
>>> print(A + B)
Bfloat16(1.8359375, sign = 0, exponent=0, mantissa=107)
```
```python
>>> D = fp32.float_to_fp32(6.871)
>>> E = fp32.float_to_fp32(-10.235)
>>> F = fp32.float_to_fp32(0.00640869)
>>> print(fp32.fma(D, E, F))
Float32(-70.31827545166016, sign = 1, exponent=6, mantissa=828149)
```

### Details
 #### Bfloat16(sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max)
 - BF16(Brain Floating Point) type class which supports for 8-bit exponent/7-bit mantissa.

 - Subnormal numbers flush to zero
 #### Float32(sign: int = 0, exponent: int = exp_max, mantissa: int = mant_max)
 - Single precision FP32 type class which supports for 8-bit exponent/23-bit mantissa.

 - Subnormal numbers flush to zero

