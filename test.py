from float_class import bf16, fp32, bf16_to_fp32, fp32_to_bf16, bf16_obj, fp32_obj


A = bf16(1, -10, 100)
B = bf16(0, 5, 35)
print(A + B)

A = bf16_obj.from_float(-1.32)
B = bf16_obj.from_float(3.156)
print(A + B)

D = fp32_obj.from_float(6.871)
E = fp32_obj.from_float(-10.235)
F = fp32_obj.from_float(0.00640869)
#print(D * E)
#D = fp32_obj.from_float(1.0)
#E = fp32_obj.from_float(2.0)
#F = fp32_obj.from_float(-4.0)
#D = bf16_obj.from_float(6.871)
#E = bf16_obj.from_float(-10.235)
#F = bf16_obj.from_float(0.00640869)
#print(D)
#print(E)
#print(F)
#print(fp32.fma(D, E, F))

bf16(1, 0, 127) + bf16_obj.from_hex(0x4060)
fp32_obj.from_float(-1.5) + fp32(0, 1, 0)
bf16_obj.from_float(2.0).__add__(bf16_obj.from_float(12.0))

bf16.fma(bf16(1, 0, 127), bf16_obj.from_hex(0x4060), bf16_obj.from_float(-12))
fp32.fma(bf16(1, 0, 127), bf16_obj.from_hex(0x4060), fp32_obj.from_float(-12))
fp32.fma(fp32_obj.from_float(1.0), fp32_obj.from_hex(0x40000000), fp32(1, 3, 7906263))

bf16(0, 1, 100).pow(3)
fp32_obj.from_float(0.001).pow(10)

-bf16(0, -10, 15)
fp32_obj.from_hex(0x41f00000).__neg__()

bf16(0, 2, 84).fptoint()
fp32_obj.from_float(0.05896311).fptoint()

bf16.inttofp(300)
fp32.inttofp(-24690)

bf16_to_fp32(bf16(0, 21, 46))

fp32_to_bf16(fp32(0, -10, 461500))

bf16(1, 0, 127) * bf16_obj.from_hex(0x4060)
fp32_obj.from_float(-1.5) * fp32(0, 1, 0)
bf16_obj.from_float(2.0).__mul__(bf16_obj.from_float(12.0))
