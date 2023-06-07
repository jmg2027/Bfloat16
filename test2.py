from bf16.bf16 import Bfloat16 as bf16
import numpy as np
import tensorflow as tf

A = bf16.float_to_bf16(-30408704.0)
B = bf16.float_to_bf16(3.844499588012695e-06)
C = bf16.float_to_bf16(-0.310546875)

p = A*B

print(A)
print(B)
print(C)

print(A.decompose_bf16())
print(B.decompose_bf16())
print(C.decompose_bf16())

print(p)
print(p+C)

print(p.decompose_bf16())
print((p+C).decompose_bf16())
