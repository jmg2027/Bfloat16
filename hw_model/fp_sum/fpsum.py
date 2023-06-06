import bf16.bf16 as bf16

# Summation Unit

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
    64(32) BF16 vectors
    16 bits accumulator
    '''
    
    align_bitwidth = 32
    def __init__(self, vector):
        self.vector = vector
#        self.acc = bf16.Bfloat16(0, -127, 0)
        # This is for test

    def set_align_bitwidth(self, n: int):
        # Should set align bitwidth more than 8
        self.align_bitwidth = n

    def summation(self):
        self.acc = bf16.Bfloat16(0, -127, 0)
        # Extract vectors
        # Decompose elements
#        decompsed_vector = (map(bf16.Bfloat16.decompose_bf16, self.vector_elements))
#        sign_v, exp_v, mant_v = zip(*decompsed_vector)
        sign_v, exp_v, mant_v = list(zip(*(map(bf16.Bfloat16.decompose_bf16, self.vector))))
        sign_acc, exp_acc, mant_nohidden_acc = self.acc.decompose_bf16()
        exp_acc_signed = bf16.sbit(bf16.Bfloat16.exponent_bits + 2, f'0{exp_acc.bin}')
        # Treat acc as fp32 in module -> mantissa to 24bits
        mant_acc_us = bf16.ubit(24, f'1{mant_nohidden_acc.bin}{16*"0"}')

        # Process elements
        # Exponent to signed bitstring
        exp_v_signed = []
        for i in exp_v:
            exp_v_signed.append(bf16.sbit(bf16.Bfloat16.exponent_bits + 2, f'0{i}'))
        
        #print('exp_v_signed', exp_v_signed)
        # Adjust hidden bit to mantissa
        # FIX: Alignment size = inner sum mantissa bitwidth, 24 + align_bitwidth
        # FIX: for bf16 input, concat 16*'0' + align_bitwidth
        mant_v_us = []
        for i in mant_v:
            mant_v_us.append(bf16.ubit(bf16.Bfloat16.mantissa_bits + 1, f'1{i}'))
        
        #print('mant_v_us', mant_v_us)

        # Normal case
        #Max tree
        l0_exp_max = [0] * 32
        for i in range(len(l0_exp_max)):
            l0_exp_max[i] = exp_v_signed[2*i] if exp_v_signed[2*i] > exp_v_signed[2*i+1] else exp_v_signed[2*i+1]

        l1_exp_max = [0] * 16
        for i in range(len(l1_exp_max)):
            l1_exp_max[i] = l0_exp_max[2*i] if l0_exp_max[2*i] > l0_exp_max[2*i+1] else l0_exp_max[2*i+1]

        l2_exp_max = [0] * 8
        for i in range(len(l2_exp_max)):
            l2_exp_max[i] = l1_exp_max[2*i] if l1_exp_max[2*i] > l1_exp_max[2*i+1] else l1_exp_max[2*i+1]

        l3_exp_max = [0] * 4
        for i in range(len(l3_exp_max)):
            l3_exp_max[i] = l2_exp_max[2*i] if l2_exp_max[2*i] > l2_exp_max[2*i+1] else l2_exp_max[2*i+1]

        l4_exp_max = [0] * 2
        for i in range(len(l4_exp_max)):
            l4_exp_max[i] = l3_exp_max[2*i] if l3_exp_max[2*i] > l3_exp_max[2*i+1] else l3_exp_max[2*i+1]

        l5_exp_max = [0]
        for i in range(len(l5_exp_max)):
            l5_exp_max[i] = l4_exp_max[2*i] if l4_exp_max[2*i] > l4_exp_max[2*i+1] else l4_exp_max[2*i+1]
        
        # Acc exp
        o_max_exp = l5_exp_max[0] if l5_exp_max[0] > exp_acc_signed else exp_acc_signed

        # Align shifter
        shamt = []
        for i in range(len(exp_v_signed)):
            shamt.append(int(o_max_exp - exp_v_signed[i]))
        # Acc shift
        shamt.append(int(o_max_exp - exp_acc_signed))

        # mantissa: x.xxx_xxxx
        # shifted mantissa: x.xxx_xxxx_xxxx_...._xxxx
        #                   < align shifter length  >
        # align_bitwidth - mantissa_bits - 1 - 1
        # 64 entries -> 6 bits
        # Extend to align shifter length+C+S+6
        # Extended mantissa: 0000_0000_h.mmm_mmmm0_...._0000
        #                    <   align shifter length+8  >
        # Make mantissa to signed bitstring
        # signed mantissa: SSSS_SSSS_h.mmm_mmmm_...._xxxx
        #                  < align shifter length+8 >
        # added mantissa: CSxx_xxxx_x.xxx_xxxx_...._xxxx
        #                 <   align shifter length+8   >
        # Carry bit is not used, so remove it
        align_bit = self.align_bitwidth
        sum_bit = align_bit + 8
        mant_v_sign = []
        for i in range(len(sign_v)):
            mant_sign = bf16.sbit(sum_bit, f'{8 * "0"}{mant_v_us[i]}{(align_bit - 9) * "0"}')
            if sign_v[i] == bf16.bit(1, '1'):
                mant_v_sign.append(-mant_sign)
            else:
                mant_v_sign.append(mant_sign)
        # accumulator
        mant_acc_sign = bf16.sbit(sum_bit, f'{8 * "0"}{mant_acc_us}{(align_bit - 9) * "0"}')
        mant_v_sign.append(mant_acc_sign)

        # mantissa shift
        # shifted & signed mantissa: Sx.xxx_xxxx_xxxx_...._xxxx
        #                            < align shifter length+1 >
        mant_v_aligned = []
        for i in range(len(mant_v_sign)):
            mant_aligned = mant_v_sign[i].arith_rshift(shamt[i])
            mant_v_aligned.append(mant_aligned)

        # Adder tree
        # 64 entries -> 6 bits
        # added mantissa: CSxx_xxxx.xxxx_xxxx_...._xxxx
        #                 <   align shifter length+8   >
        # Carry bit is not used, so remove it
        mant_add = bf16.sbit(sum_bit, '0')
        for i in range(len(mant_v_aligned)):
            mant_add = mant_add + mant_v_aligned[i]


        print('mant_add: ', repr(mant_add))

        # Post adder & accumulation
        # Sign bitpos: align shifter length + 7
        mant_add_sign = mant_add[sum_bit-2]
        zero_mant_result = mant_add[sum_bit-3:0] == bf16.sbit(sum_bit-2, '0')
        mant_add_nocarry = mant_add[sum_bit-2:0]
        # mantissa result before sign removal: Sxx_xxxx.xxxx_xxxx_...._xxxx
        # mantissa result: xx_xxxx.xxxx_xxxx_...._xxxx
        #                  <  align shifter length+6 >
        mant_add_result_before_sign_remove = bf16.ubit(sum_bit-1, f'{(-mant_add_nocarry).bin if mant_add_sign else mant_add_nocarry.bin}')
        mant_add_result = mant_add_result_before_sign_remove[sum_bit-3:0]

        # Leading zero count for close path
        # to sum_bit - 8
        if mant_add_result[sum_bit-3].bin == '1':
            rshamt = 5
            print('Im in 5')
        elif mant_add_result[sum_bit-4].bin == '1':
            rshamt = 4
            print('Im in 4')
        elif mant_add_result[sum_bit-5].bin == '1':
            rshamt = 3
            print('Im in 3')
        elif mant_add_result[sum_bit-6].bin == '1':
            rshamt = 2
            print('Im in 2')
        elif mant_add_result[sum_bit-7].bin == '1':
            rshamt = 1
            print('Im in 1')
        elif mant_add_result[sum_bit-8].bin == '1':
            rshamt = 0
            print('Im in 0')
        else:
            rshamt = 0
            print('Im in else')

        close_path = mant_add_result >> rshamt
    
        # Leading zero count for far path
        clz_in = mant_add_result[sum_bit-9:0]
        if bf16.hwutil.leading_zero_count(clz_in) < (sum_bit-8):
            lshamt = bf16.hwutil.leading_zero_count(clz_in)
        else:
            lshamt = 0
        print('rshamt', rshamt)
        print('lshamt', lshamt)
        
        far_path = clz_in << lshamt
        rnd_in = far_path if rshamt == 0 else close_path

        # round in: x.xxxx_xxxx_xxxx_...._xxxx
        #                  < align shifter length+1 >
        print('rnd_in: ', rnd_in)
        rnd = rnd_in[7]
        sticky = rnd_in[6:0].reduceor()
        round = (rnd & sticky) | (rnd_in[8] & rnd & ~sticky)

        normed = bf16.ubit(align_bit+1, (rnd_in + (round << 7)).bin)

        if zero_mant_result:
            t_exp = bf16.ubit(8, '0')
        elif rshamt > 0:
            t_exp = bf16.ubit(8, bin(int(o_max_exp) + rshamt + 2))
        else:
            t_exp = bf16.ubit(8, bin(int(o_max_exp) + 2 - lshamt))
        
        # Post normalization
        out_sign = mant_add_sign
        out_exp = t_exp + 1 if normed[align_bit].bin == 1 else t_exp
        # 7bits of fraction
        out_frac = normed[align_bit-1:align_bit-7] if normed[align_bit].bin == 1 else normed[align_bit-2:align_bit-8]

        print(out_sign, out_exp, out_frac)
        print(repr(out_sign))

        summation = bf16.Bfloat16.compose_bf16(out_sign, out_exp, out_frac)
        return summation
#        return summation
