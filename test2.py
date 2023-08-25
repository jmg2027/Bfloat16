from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
import numpy as np
import tensorflow as tf

import test

#A = bf16.float_to_bf16(-30408704.0)
#B = bf16.float_to_bf16(3.844499588012695e-06)
#C = bf16.float_to_bf16(-0.310546875)
#
#p = A*B
#
#print(A)
#print(B)
#print(C)
#
#print(A.decompose_bf16())
#print(B.decompose_bf16())
#print(C.decompose_bf16())
#
#print(p)
#print(p+C)
#
#print(p.decompose_bf16())
#print((p+C).decompose_bf16())

A = bf16.float_to_bf16(-1.32)
B = bf16.float_to_bf16(-3.156)
C = bf16.float_to_bf16(0.00640869)
D = fp32.float_to_fp32(6.871)
E = fp32.float_to_fp32(-10.235)
F = fp32.float_to_fp32(0.00640869)

#print(A)
#print(B)
#print(C)
#print(D)
#
print(A+B)
print(D+E)
#print(A+B+C+D)
#print(E)

#print(A.bf16_to_fp32())
#print(B)
#print(B.fp32_to_bf16())

print(bf16.fma(A, B, C))
print(fp32.fma(D, E, F))

print(B.hex())
print(B)
print(bf16.from_hex(0xc04a))

print(E.hex())
print(E)
print(fp32.from_hex(0xc123c28f))

a = [0x4309, 0x4019, 0x3ab3, 0xbcd9, 0xbec3, 0x4450, 0x3e02, 0x3e45, 0x3f77, 0xbceb, 0xc14b, 0x3b7e, 0xbe45, 0xc1a8, 0x3f25, 0xc174, 0x41c0, 0xbfed, 0x448a, 0xbc24, 0xbc96, 0xc1d5, 0x3e58, 0xbbf6, 0xbbb5, 0xc0b8, 0xbf67, 0x413f, 0xbc90, 0xba83, 0x3e6e, 0xbb4d]
b = [0x40f4, 0x40af, 0xbf03, 0xc38b, 0xbff9, 0xc479, 0xbb98, 0x418d, 0xbdec, 0x3e5c, 0xbebd, 0x3eab, 0x3acf, 0xbe0d, 0x4409, 0xc116, 0x3eac, 0x4335, 0x420d, 0xc27a, 0xc054, 0xc112, 0xc1ca, 0x4119, 0xc104, 0x3afd, 0x405e, 0xbc6c, 0x3ff1, 0xbbd6, 0x4062, 0x4157]

ab = [list(map(bf16.from_hex, a)), list(map(bf16.from_hex, b))]
print(bf16.summation(ab))

print(bf16.float_to_bf16('-inf').hex())

print(test.test_add.cast_float(0.1))
