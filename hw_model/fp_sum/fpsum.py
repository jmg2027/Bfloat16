import bf16.bf16 as bf16

# Mul & Reduce Unit
# AIP architecture (hyunpil.kim@samsung.com)
# FMUL -> reg -> ALIGN SHIFT -> ADDER TREE -> POST ADDER & FACC
# 32BIT FLOAT ACC = ACCUM(8 BF16 INPUT, 8 BF16 WEIGHT)
# ACC + reduce_sum(BF16 A, BF16 B)
'''
FMUL
Floating point multiplyer
sign: 8 sign bits are xored
exp: 9 exps, including previous ACC result goes in max tree
8 sum of exps (mul_exp)
mant:
8 mans multiplies, h m m m m m m m 0 0 0 (11bits)
mant is signed
mul res: 22bits
With sign bit, previous ACC is being negated or not
'''
'''
Align shifter
Tree architecture
SQNR difference
"it is more accurate than result of just two floating point addition due to non-rounding tree addition"
"over 16bit align shifter is enough to cover range of FP16 and BF16. But, we used 32bit align shifter because of single precision accumulation and sharing with integer adder tree"
Align bit width should be variable
"floating point adder tree with 32bit align shifter always shows more accurate than results of single precision floating point (FP32) FMAC"
diff(max_exp, mul_exp) -> shift amount for each mul_exp
shift mul_mants with 8 signs (arithmetic shift right)
XXX.X_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX
shift previous ACC result with max_exp - acc_exp
res: 9 aligned mants
'''
'''
Adder tree
Aligned inputs
"32bit aligned values from align shift block are separated into msb and lsb part"
"The results of msb and the most significant 4 bits of lsb part will be sumed in the floating point post adder block"
20bits of msb_out, lsb_out
'''
'''
Post adder float
post add and fp accumulation
"first adds two results from adder tree"
tree_hap = 37bits? 40bits?
	assign	msb_hap = $signed(msb_in + lsb_in[19:16]);
msb of tree_hap is sign -> negate rest
close path: "If there are similar size of inputs with same sign, the position of leading one is found within [35:31]. "
far path: "if there are similar size of input with different sign, the position of leading one is found within [31:0]"
unrounded result = 0000_1.XXX_XXXX_XXXX_XXXX_XXXX_XXXX_RSSS_SSSS
'''

class FloatSummation:
    '''
    8 BF16 vectors (assumes list input)
    '''
    
    align_bitwidth = 32
    def __init__(self, iterable):
        self.input_vector = iterable
        self.weight_vector = iterable
        self.acc = bf16.Bfloat16(0, 0, 0)
        # This is for test
        self.input_vector = [
            bf16.Bfloat16.float_to_bf16(1.0),
            bf16.Bfloat16.float_to_bf16(-1.2),
            bf16.Bfloat16.float_to_bf16(4.0),
            bf16.Bfloat16.float_to_bf16(5.0),
            bf16.Bfloat16.float_to_bf16(-10.0),
            bf16.Bfloat16.float_to_bf16(-20.0),
            bf16.Bfloat16.float_to_bf16(30.0),
            bf16.Bfloat16.float_to_bf16(-100.0)
            ]
        self.weight_vector = [
            bf16.Bfloat16.float_to_bf16(1.0),
            bf16.Bfloat16.float_to_bf16(-1.2),
            bf16.Bfloat16.float_to_bf16(4.0),
            bf16.Bfloat16.float_to_bf16(5.0),
            bf16.Bfloat16.float_to_bf16(-10.0),
            bf16.Bfloat16.float_to_bf16(-20.0),
            bf16.Bfloat16.float_to_bf16(30.0),
            bf16.Bfloat16.float_to_bf16(-100.0)
            ]

    def set_align_bitwidth(self, n: int):
        self.align_bitwidth = n

    def summation(self):
        # Extract vectors
        # Decompose elements
#        decompsed_vector = (map(bf16.Bfloat16.decompose_bf16, self.vector_elements))
#        sign_v, exp_v, mant_v = zip(*decompsed_vector)
        sign_i, exp_i, mant_i = list(zip(*(map(bf16.Bfloat16.decompose_bf16, self.input_vector))))
        sign_w, exp_w, mant_w = list(zip(*(map(bf16.Bfloat16.decompose_bf16, self.weight_vector))))

        # 

        # FMUL

        # Align shifter

        # Adder tree

        #Post adder & accumulation

        summation = bf16.Bfloat16.compose_bf16(0, 0, 0)
        return summation
