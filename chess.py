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
    a_input = list()
    b_input = list()
    res = list()
    for e in mul_test_list:
        a_input.append(e[0].hex())
        b_input.append(e[1].hex())
        res.append(e[2].hex())
    return

#def convert_fdata(a, b, res):
map()
#    return
# make random inputs fdata.h
# save result values as result.txt
# run chessmk
# compare test.mem & result.txt
