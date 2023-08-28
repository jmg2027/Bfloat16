from bf16 import utils
from bf16.bf16 import Float32 as fp32
from bf16.bf16 import Bfloat16 as bf16
import test

test_set = [
		(2, 12),
		(-2, 12),
		(2, -12),
		(-2, -12),
		(25.0924, 24.8076),
		(-25.0924, 24.8076),
		(25.0924, -24.8076),
		(-25.0924, -24.8076),
        (4.5, 6),
        (-4.5, 6),
        (4.5, -6),
        (-4.5, -6),
		(123.124, 381.58),
		(123.124, -381.58),
		(-123.124, 381.58),
		(-123.124, -381.58),
		(0.00076, 0.3256),
		(-0.00076, 0.3256),
		(0.00076, -0.3256),
		(-0.00076, -0.3256),
        # exponent is far larger that does not affect to other addend
		(111111111.111111111, 999999999999.999999999999),
		(111111111.111111111, -999999999999.999999999999),
    # Corner cases
		(0, 0),
		(10.10293, -0.0000000000000000000000000000000000000001),
		(10.10293, 0.0000000000000000000000000000000000000001),
   # 100931731456
   # 1000727379968
		(101029300000, 999999999999),
		(101029300000, -999999999999),
    # zero case
        (2, -2),
        (-2, 2),
    # invert case
        (-4, 2),
        (2, -4),
    # When two add of mantissa is 1111_1111_1110
        (float(bf16(0, 0, bf16.mant_max)), float(bf16(0, 0, bf16.mant_max))),
    # When two add of mantissa is 0111_1111_1111
        (float(bf16(0, 8, bf16.mant_max)), float(bf16(0, 0, bf16.mant_max)))
]

# Use this for conversion of test set into fdata.h
def test_set_convert_to_chess_test(test_list):
	hex_set = list()
	for a, b in test_list:
		h1, h2 = utils.float_to_hex(a), utils.float_to_hex(b)
		#hex_set.append((h1, h2))
		hex_set.append(h1)
		hex_set.append(h2)
	print(hex_set)
	return hex_set

# Use this for conversion of test set into fdata.h
# 3 inputs
def test_set_convert_to_chess_test_fma(test_list):
	hex_set = list()
	for a, b, c in test_list:
		h1, h2, h3 = utils.float_to_hex(a), utils.float_to_hex(b), utils.float_to_hex(c)
		#hex_set.append((h1, h2))
		hex_set.append(h1)
		hex_set.append(h2)
		hex_set.append(h3)
		hex_set.append('\n')
	print(hex_set)
	return hex_set

# Use this for conversion of test set into fdata.h
# 3 inputs
def test_set_convert_to_chess_test_summation(test_list):
	hex_set = list()
	for a, b, c in test_list:
		h1, h2, h3 = utils.float_to_hex(a), utils.float_to_hex(b), utils.float_to_hex(c)
		#hex_set.append((h1, h2))
		hex_set.append(h1)
		hex_set.append(h2)
		hex_set.append(h3)
		hex_set.append('\n')
	print(hex_set)
	return hex_set

def tohex(b):
	return hex(int(b, 2))

# float -> fp32 -> sign, exponent, mantissa
def decompose_float(float_num: float):
	fp32_obj = fp32.from_float(float_num)
	s, e, m = fp32_obj.decompose()
	return hex(int(s)), hex(int(e)), hex(int(m))

# hex -> fp32 ex) hex_to_fp32('0x40c00000')
def hex_to_fp32(h: str) -> 'fp32':
	return fp32.from_float(utils.hex_to_float(h[2:]))

# hex -> sign, exponent, mantissa ex) decompose_hex(0x40c00000)
def decompose_hex(h: int):
	f = hex_to_fp32(hex(h))
	return decompose_float(f)

#a = 49.900001525878906
#b = 0x418f3333
#print(decompose_float(a))
#print(decompose_hex(b))

#a = fp32(0, -126, 16)
#print(a)
#print(a.hex())

def rand_vec():
	return test.test_summation.rand_vector()

def rand_vec_to_hex(rand_vec):
	rand_bf16 = test.test_summation.convert_element_to_bf16_array(rand_vec)
	rand_hex = list()
	for i in rand_bf16:
		rand_hex.append(i.hex())
	return rand_hex

def rand_single_test(vector_num: int = 2):
	single_test_set = list()
	for i in range(vector_num):
		single_test_set.append(rand_vec())
	return single_test_set

def rand_test_set(set_num: int = 1):
	test_set = list()
	for i in range(set_num):
		test_set.append(rand_single_test())
	return test_set

def list_to_string(l):
	if isinstance(l, list):
		return f'[{", ".join(list_to_string(e) for e in l)}]\n'
	else:
		return str(l)

def test_rand_summation():
	test_set = rand_test_set()
	test_set_hex = list()
	for vector_set in test_set:
		test_vector_hex = list()
		for v in vector_set:
			test_vector_hex.append(rand_vec_to_hex(v))
		test_set_hex.append(test_vector_hex)
	bf16_set_list = list()
	for vector_set in test_set:
		bf16_vector_list = list()
		for v in vector_set:
			bf16_vector_list.append(test.test_summation.convert_element_to_bf16_array(v))
		bf16_set_list.append(bf16_vector_list)
	res_list = list()
	for vector_set in bf16_set_list:
		res = bf16.summation(vector_set)
		res_list.append(res)
	print(test_set_hex)
	with open('fdata.h', 'w') as f:
		fdata_str = list_to_string(test_set_hex)
		f.write(fdata_str)
	with open('test_set.txt', 'w') as f:
		test_set_str = list_to_string(test_set)
		f.write(test_set_str)
	with open('result.txt', 'w') as f:
		res_list_str = list_to_string(res_list)
		f.write(res_list_str)
	return

def test_summation(vector_set):
	res = bf16.summation(vector_set)
	print(res)
	return

#test_rand_summation()
vec_set = [[137.0, 2.390625, 0.00136566162109375, -0.0264892578125, -0.380859375, 832.0, 0.126953125, 0.1923828125, 0.96484375, -0.0286865234375, -12.6875, 0.003875732421875, -0.1923828125, -21.0, 0.64453125, -15.25, 24.0, -1.8515625, 1104.0, -0.010009765625, -0.018310546875, -26.625, 0.2109375, -0.00750732421875, -0.005523681640625, -5.75, -0.90234375, 11.9375, -0.017578125, -0.00099945068359375, 0.232421875, -0.0031280517578125]
, [7.625, 5.46875, -0.51171875, -278.0, -1.9453125, -996.0, -0.004638671875, 17.625, -0.115234375, 0.21484375, -0.369140625, 0.333984375, 0.00157928466796875, -0.1376953125, 548.0, -9.375, 0.3359375, 181.0, 35.25, -62.5, -3.3125, -9.125, -25.25, 9.5625, -8.25, 0.00193023681640625, 3.46875, -0.014404296875, 1.8828125, -0.00653076171875, 3.53125, 13.4375]
]

bf16_set = list()
print(vec_set)
for i in vec_set:
	print(i)
	bf16_set.append(test.test_summation.convert_element_to_bf16_array(i))

test_summation(bf16_set)
