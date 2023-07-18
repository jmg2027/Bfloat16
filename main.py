from bf16 import utils
from bf16.bf16 import Float32 as fp32
from bf16.bf16 import Bfloat16 as bf16

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

def tohex(b):
	return hex(int(b, 2))

# float -> fp32 -> sign, exponent, mantissa
def decompose_float(float_num: float):
	fp32_obj = fp32.float_to_fp32(float_num)
	s, e, m = fp32_obj.decompose()
	return hex(int(s)), hex(int(e)), hex(int(m))

# hex -> fp32 ex) hex_to_fp32('0x40c00000')
def hex_to_fp32(h: str) -> 'fp32':
	return fp32.float_to_fp32(utils.hex_to_float(h[2:]))

# hex -> sign, exponent, mantissa ex) decompose_hex(0x40c00000)
def decompose_hex(h: int):
	f = hex_to_fp32(hex(h))
	return decompose_float(f)

a = 49.900001525878906
b = 0x418f3333
print(decompose_float(a))
print(decompose_hex(b))
