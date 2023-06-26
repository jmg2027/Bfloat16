from bf16.bf16 import Bfloat16 as bf16
from bf16.bf16 import Float32 as fp32
import numpy as np
import tensorflow as tf

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
B = bf16.float_to_bf16(3.156)
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