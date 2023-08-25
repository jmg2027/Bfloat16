import test

# Use this for conversion of test set into fdata.h
def convert_fdata(test_list):
	hex_set = list()
	for a, b in test_list:
		h1, h2 = utils.float_to_hex(a), utils.float_to_hex(b)
		#hex_set.append((h1, h2))
		hex_set.append(h1)
		hex_set.append(h2)
	print(hex_set)
	return hex_set

# run python random test
def mul_hex_list():
    mul_test_list = test.test_mul.rand_test(10)
    a_input, b_input, res = list(), list(), list()
    for e in mul_test_list:
        a_input.append(e[0].hex())
        b_input.append(e[1].hex())
        res.append(e[2].hex())
    return a_input, b_input, res

# run python random test
def fma_hex_list():
    fma_test_list = test.test_fma.rand_test(10)
    a_input, b_input, c_input, res = list(), list(), list(), list()
    for e in fma_test_list:
        a_input.append(e[0].hex())
        b_input.append(e[1].hex())
        c_input.append(e[1].hex())
        res.append(e[2].hex())
    return a_input, b_input, c_input, res

a, b, res = mul_hex_list()
aa, bb, cc, res = fma_hex_list()

def conv_fdata_zip(z: zip):
    s = list()
    for i in z:
        s.append(f'{", ".join(i)}')
    return ',\n'.join(s)
#print(conv_fdata_zip(zip(a, b, res)))
print(conv_fdata_zip(zip(a, b)))
print('\n'.join(res))
print(conv_fdata_zip(zip(aa, bb, cc)))
print('\n'.join(res))
#    return
# make random inputs fdata.h
# save result values as result.txt
# run chessmk
# compare test.mem & result.txt
