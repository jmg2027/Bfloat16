import bf16.bf16 as bf16

# Mul & Reduce Unit
# AIP architecture (hyunpil.kim@samsung.com)
# FMUL -> reg -> ALIGN SHIFT -> ADDER TREE -> POST ADDER & FACC
# 32BIT FLOAT ACC = ACCUM(8 BF16 INPUT, 8 BF16 WEIGHT)
# ACC + reduce_sum(BF16 A, BF16 B)

# FIX: Change name to Inner product
# FIX: Make summation logic
# FIX: Summation: Sum(bf16 vectors , ACC(FP32/BF16))
# FIX: output: Singel FP32/BF16

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

class FloatMRU:
    '''
    8 BF16 vectors (assumes list input)
    FIX: 64(32) BF16 vectors
    16 bits accumulator
    '''
    
    align_bitwidth = 32
    def __init__(self, input_vector, weight_vector):
        self.input_vector = input_vector
        self.weight_vector = weight_vector
#        self.acc = bf16.Bfloat16(0, -127, 0)
        # This is for test

    def set_align_bitwidth(self, n: int):
        self.align_bitwidth = n

    def summation(self):
        self.acc = bf16.Bfloat16(0, -127, 0)
        # Extract vectors
        # Decompose elements
#        decompsed_vector = (map(bf16.Bfloat16.decompose_bf16, self.vector_elements))
#        sign_v, exp_v, mant_v = zip(*decompsed_vector)
        sign_i, exp_i, mant_i = list(zip(*(map(bf16.Bfloat16.decompose, self.input_vector))))
        sign_w, exp_w, mant_w = list(zip(*(map(bf16.Bfloat16.decompose, self.weight_vector))))
        sign_acc, exp_acc, mant_nohidden_acc = self.acc.decompose()
        exp_acc_signed = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, f'0{exp_acc.bin}')
        # Treat acc as fp32 in module -> mantissa to 24bits
        mant_acc_us = bf16.ubit(24, f'1{mant_nohidden_acc.bin}{16*"0"}')

        # Process elements
        # Exponent to signed bitstring
        exp_i_signed = []
        for i in exp_i:
            exp_i_signed.append(bf16.sbit(bf16.Bfloat16.exponent_bits + 2, f'0{i}'))
        exp_w_signed = []
        for i in exp_w:
            exp_w_signed.append(bf16.sbit(bf16.Bfloat16.exponent_bits + 2, f'0{i}'))
        bias_signed = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, bin(bf16.Bfloat16.bias))
        
#        print('exp_i_signed', exp_i_signed)
#        print('exp_w_signed', exp_w_signed)
        # Adjust hidden bit to mantissa
        # Adjust 3 zero bits
        # FIX: GRS
        # FIX: Alignment size = inner sum mantissa bitwidth, 24 + align_bitwidth
        # FIX: for bf16 input, concat 16*'0' + align_bitwidth
        mant_i_us = []
        for i in mant_i:
            mant_i_us.append(bf16.ubit(bf16.Bfloat16.mantissa_bits + 1, f'1{i}000'))
        mant_w_us = []
        for i in mant_w:
            mant_w_us.append(bf16.ubit(bf16.Bfloat16.mantissa_bits + 1, f'1{i}000'))
        
#        print('mant_i_us', mant_i_us)
#        print('mant_w_us', mant_w_us)

        # Normal case

        # FMUL
        # Calculate sign bit
        o_sign = []
        for i in range(len(sign_i)):
            o_sign.append(sign_i[i] ^ sign_w[i])
        # Calculate exponent
        w_exp_mul = []
        for i in range(len(exp_i_signed)):
            w_exp_mul.append(exp_i_signed[i] + exp_w_signed[i] - bias_signed)
        # append for acc
        w_exp_mul.append(exp_acc_signed)
        o_exp_mul = w_exp_mul

        #Max tree
        l0_exp_max = [0,0,0,0]
        l0_exp_max[0] = w_exp_mul[0] if w_exp_mul[0] > w_exp_mul[1] else w_exp_mul[1]
        l0_exp_max[1] = w_exp_mul[2] if w_exp_mul[2] > w_exp_mul[3] else w_exp_mul[3]
        l0_exp_max[2] = w_exp_mul[4] if w_exp_mul[4] > w_exp_mul[5] else w_exp_mul[5]
        l0_exp_max[3] = w_exp_mul[6] if w_exp_mul[6] > w_exp_mul[7] else w_exp_mul[7]

        l1_exp_max = [0,0]
        l1_exp_max[0] = l0_exp_max[0] if l0_exp_max[0] > l0_exp_max[1] else l0_exp_max[1]
        l1_exp_max[1] = l0_exp_max[2] if l0_exp_max[2] > l0_exp_max[3] else l0_exp_max[3]

        l2_exp_max = l1_exp_max[0] if l1_exp_max[0] > l1_exp_max[1] else l1_exp_max[1]

        l3_exp_max = l2_exp_max if l2_exp_max > o_exp_mul[8] else o_exp_mul
        o_max_exp = l3_exp_max

        #mantissa multiplication
        o_mul_mant = []
        for i in range(len(mant_i_us)):
            o_mul_mant.append(mant_i_us[i] * mant_w_us[i])
#        print(o_mul_mant)
        #negate acc mantissa if signed
        add_mant_unsigned = bf16.sbit(25, f'0{mant_acc_us}')
        o_add_mant = -add_mant_unsigned if sign_acc == bf16.bit(1, '1') else add_mant_unsigned

        # Align shifter
        shamt = []
        for i in range(len(o_exp_mul)):
            shamt.append(int(o_max_exp - o_exp_mul[i]))
        # t_mant_mul = [31:0]
        w_mant_mul = []
        for i in range(len(o_sign)):
            t_mant_mul = bf16.sbit(32, f'0{o_mul_mant[i]}{9*"0"}')
            if o_sign[i] == bf16.bit(1, '1'):
                w_mant_mul.append(-t_mant_mul)
            else:
                w_mant_mul.append(t_mant_mul)
        # accumulator
        w_mant_mul.append(bf16.sbit(32, f'{o_add_mant[24]}{o_add_mant}{6*"0"}'))

        # mantissa shift
        o_aligned = []
        for i in range(len(w_mant_mul)):
            t_aligned = w_mant_mul[i].arith_rshift(shamt[i])
            o_aligned.append(t_aligned)

        # Adder tree
        # divide into lsb, msb
        w_aligned_lsb = []
        for i in range(len(o_aligned)):
            w_aligned_lsb.append(bf16.bit(20, f'{4*"0"}{o_aligned[i][15:0]}'))
        w_aligned_msb = []
        for i in range(len(o_aligned)):
            w_aligned_msb.append(bf16.bit(20, f'{4*"0"}{o_aligned[i][31:16]}'))
        
        o_lsb_out = w_aligned_lsb[0] + \
                    w_aligned_lsb[1] + \
                    w_aligned_lsb[2] + \
                    w_aligned_lsb[3] + \
                    w_aligned_lsb[4] + \
                    w_aligned_lsb[5] + \
                    w_aligned_lsb[6] + \
                    w_aligned_lsb[7] + \
                    w_aligned_lsb[8]

        o_msb_out = w_aligned_msb[0] + \
                    w_aligned_msb[1] + \
                    w_aligned_msb[2] + \
                    w_aligned_msb[3] + \
                    w_aligned_msb[4] + \
                    w_aligned_msb[5] + \
                    w_aligned_msb[6] + \
                    w_aligned_msb[7] + \
                    w_aligned_msb[8]

        #print(repr(o_lsb_out))
        #print(repr(o_msb_out))

        #Post adder & accumulation
        msb_in = bf16.sbit(20, o_msb_out.bin)
        lsb_in = bf16.sbit(20, o_lsb_out.bin)
        msb_hap = bf16.sbit(21, f'{(msb_in + lsb_in[19:16]).bin}')
        tree_hap = bf16.sbit(37, f'{msb_hap.bin}{lsb_in[15:0].bin}')
        zero_hap = tree_hap == bf16.sbit(37, '0')
        p_tree_hap = bf16.ubit(36, f'{(-tree_hap).bin if tree_hap[36].bin == 1 else tree_hap.bin}')

        # Leading zero count for close path
        if p_tree_hap[35].bin == 1:
            rshamt = 4
        elif p_tree_hap[34].bin == 1:
            rshamt = 3
        elif p_tree_hap[33].bin == 1:
            rshamt = 2
        elif p_tree_hap[32].bin == 1:
            rshamt = 1
        elif p_tree_hap[31].bin == 1:
            rshamt = 0
        else:
            rshamt = 0

        close_path = p_tree_hap >> rshamt
    
        # Leading zero count for far path
        clz_in = p_tree_hap[31:0]
        if bf16.hwutil.leading_zero_count(clz_in) < 32:
            lshamt = bf16.hwutil.leading_zero_count(clz_in)
        else:
            lshamt = 0
        
        far_path = clz_in << lshamt
        rnd_in = far_path if rshamt == 0 else close_path[31:0]

        rnd = rnd_in[7]
        sticky = rnd_in[6:0].reduceor()
        round = (rnd & sticky) | (rnd_in[8] & rnd & ~sticky)

        normed = bf16.ubit(33, (rnd_in + (round << 7)).bin)

        if zero_hap:
            t_exp = bf16.ubit(8, '0')
        elif rshamt > 0:
            t_exp = bf16.ubit(8, bin(int(o_max_exp) + rshamt + 2))
        else:
            t_exp = bf16.ubit(8, bin(int(o_max_exp) + 2 - lshamt))
        
        # Post normalization
        out_sign = tree_hap[36]
        out_exp = t_exp + 1 if normed[32].bin == 1 else t_exp
        # 7bits of fraction
        out_frac = normed[31:24] if normed[32].bin == 1 else normed[30:23]

        print(out_sign, out_exp, out_frac)

        summation = bf16.Bfloat16.compose(out_sign, out_exp, out_frac)
        return summation
#        return summation
