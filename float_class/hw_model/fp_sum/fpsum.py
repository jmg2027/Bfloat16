from ..utils.commonimport import *
from typing import List, Union

#TestT = Union[int, float, bf16, fp32]


# Summation Unit

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
    Internal operations are treated in FP32
    '''
    '''
    FIX: Number of vector input to power of 2(n)
    FIX: Make 4 input summation unit for debugging!
    FIX: ACC can be FP32/BF16, output also
    '''
    
    align_bitwidth = 32
    #align_bitwidth = 64
    vector_element_num = 32
    #vector_element_num = 4
    def __init__(self, vector: FPBitVectorT, acc: FPBitT, mod: int = 0) -> None:
        # vector_list = [Element, Element, ...]
        self.vector = vector
        self.acc = acc
        self.mod = mod
        pass

    def set_align_bitwidth(self, n: int) -> None:
        # Should set align bitwidth more than 10: mantissa bits + hidden bit + 
        self.align_bitwidth: int = n

    def set_vector_element_num(self, n: int) -> None:
        # Make it power of 2
        self.vector_element_num: int = n

    def set_vector(self, vector: FPBitVectorT) -> None:
        self.vector = vector
    
    def set_acc(self, acc: FPBitT) -> None:
        self.acc = acc
    
    def set_mod(self, mod: int) -> None:
        self.mod = mod

    def execute(self, algorithm = "SINGLE_PATH") -> FPBitT:
        if algorithm == "SINGLE_PATH":
            ret_sign, ret_exp, ret_mant = self.execute_single_path()
        elif algorithm == "MULTI_PATH":
            ret_sign, ret_exp, ret_mant = self.execute_multi_path()
        else:
            raise TypeError(f"FMA supports SINGLE_PATH, MULTI_PATH algorithm, not {algorithm}")
        return ret_sign, ret_exp, ret_mant

    def execute_single_path(self) -> FPBitT:
        # Special cases
        # Need handle accumulator later...
        # input
        isnormal = False
        # Conditions
        # In vector if there's any nan -> nan
        # nan inputs clearance
        # inf + inf = inf
        # -inf + -inf = -inf
        # In vector if there's any inf with same sign
        any_nan = False
        any_inf = False
        inf_num = 0
        neg_inf_num = 0
        inf_sign = 0
        out_sign = bit(1, '0')
        out_exp = bit(1, '0')
        out_frac = bit(1, '0')
        for i in self.vector:
            any_nan |= isnan(i)
            if isinf(i):
                inf_num = inf_num + 1
                if i[0] == bit(1, '1'):    # negative sign
                    neg_inf_num = neg_inf_num + 1

        any_nan |= isnan(self.acc)
        if isinf(self.acc):
            inf_num = inf_num + 1
            if self.acc[0] == bit(1, '1') >> 31:   # negative sign
                neg_inf_num = neg_inf_num + 1

        all_inf_same_sign = False
        if inf_num != 0:
            any_inf = True
            if neg_inf_num == 0:
                all_inf_same_sign = True
                inf_sign = 0
            elif neg_inf_num == inf_num:
                all_inf_same_sign = False
                inf_sign = 1

        # In vector if there's any nan -> nan
        # nan inputs clearance
        if any_nan:
            out_sign = bit(1, '0')
            out_exp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            out_frac = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # inf + inf = inf
        # -inf + -inf = -inf
        # In vector if there's any inf with same sign
        elif any_inf and all_inf_same_sign:
            out_sign = bit(1, str(inf_sign))
            out_exp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            out_frac = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # -inf + +inf = nan
        # In vector if there's any inf with different sign
        elif any_inf and not all_inf_same_sign:
            out_sign = bit(1, '0')
            out_exp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            out_frac = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        else:
            isnormal = True

        if isnormal:
            # Extract vectors
            # Decompose elements
    #        decompsed_vector = (map(bf16.Bfloat16.decompose_bf16, self.vector_elements))
    #        sign_v, exp_v, mant_v = zip(*decompsed_vector)
            # make vector of sign, exp, mant
            sign_v, exp_v, mant_v = [list(v) for v in zip(*self.vector)]
            # JMG
            #print(sign_v)
            sign_acc, exp_acc, mant_nohidden_acc = self.acc
            exp_acc_signed = bit(fp32_config.exponent_bits, f'{exp_acc.bin}')
            # Treat acc as fp32 in module -> mantissa to 24bits
            mant_acc_us = ubit(fp32_config.mantissa_bits + 1, f'{exp_acc_signed.reduceor()}{mant_nohidden_acc.bin}')
            #print('mant_nohidden_acc', repr(mant_nohidden_acc))
            #print('mant_acc_us', repr(mant_acc_us))

            # Process elements
            # Exponent to signed bitstring
            # FIX: For 0 input, use 0 as hidden bit
            exp_v_signed: List[bit] = []
            hidden_bit: List[bit] = []
            for i in exp_v:
                exp_v_signed.append(bit(fp32_config.exponent_bits, f'{i}'))
                hidden_bit.append(i.reduceor())
            
            # Adjust hidden bit to mantissa
            # FIX: Alignment size = inner sum mantissa bitwidth, 24 + align_bitwidth
            # FIX: for bf16 input, concat 16*'0' + align_bitwidth
            # FIX: For 0 input, use 0 as hidden bit
            mant_v_us: List[ubit] = []
            for i in range(len(mant_v)):
                mant_v_us.append(ubit(fp32_config.mantissa_bits + 1, f'{hidden_bit[i]}{mant_v[i]}'))
            

            # Normal case
            #Max tree

            # Find tree level
            element_num = self.vector_element_num
            element_num_shift = element_num
            tree_level = 0
            while element_num_shift != 1:
                element_num_shift = element_num_shift >> 1
                tree_level += 1

            '''
            # Exponent max tree
            # Make zero initialized dual list of tree
            # [[0] * 2**(tree_level), [0] * 2**(tree_level-1), ..., [0]]
            # Initialize first stage with exp_v_signed
            # Compare previous stage (i) and record it to current stage(i+1)
            exp_max_tree: List[List[sbit]] = [[sbit(fp32_config.exponent_bits + 2, '0'*(fp32_config.exponent_bits + 2))] * (2**(tree_level-i)) for i in range(tree_level+1)]
            # Initialize
            for i in range(len(exp_v_signed)):
                exp_max_tree[0][i] = exp_v_signed[i]
            # Compare and record
            for i in range(tree_level):
                for j in range(len(exp_max_tree[i+1])):
                    exp_max_tree[i+1][j] = exp_max_tree[i][2*j] if (exp_max_tree[i][2*j] > exp_max_tree[i][2*j+1]) else exp_max_tree[i][2*j+1]
            exp_max_vector = exp_max_tree[tree_level][0]
            o_max_exp = exp_max_vector if exp_max_vector > exp_acc_signed else exp_acc_signed
            '''
            mask = bit(33, bin(0x1ffffffff))
            val0 = bit(33, '0')
            val1 = bit(33, '0')
            val2 = bit(33, '0')
            val3 = bit(33, '0')
            val4 = bit(33, '0')
            val5 = bit(33, '0')
            val6 = bit(33, '0')
            val7 = bit(33, '0')
            for i in range(len(exp_v_signed)):
                val0[i] = exp_v_signed[i][0].bin
                val1[i] = exp_v_signed[i][1].bin
                val2[i] = exp_v_signed[i][2].bin
                val3[i] = exp_v_signed[i][3].bin
                val4[i] = exp_v_signed[i][4].bin
                val5[i] = exp_v_signed[i][5].bin
                val6[i] = exp_v_signed[i][6].bin
                val7[i] = exp_v_signed[i][7].bin
            val0[32] = exp_acc_signed[0].bin
            val1[32] = exp_acc_signed[1].bin
            val2[32] = exp_acc_signed[2].bin
            val3[32] = exp_acc_signed[3].bin
            val4[32] = exp_acc_signed[4].bin
            val5[32] = exp_acc_signed[5].bin
            val6[32] = exp_acc_signed[6].bin
            val7[32] = exp_acc_signed[7].bin

            ans0 = bit(33, '0')
            ans1 = bit(33, '0')
            ans2 = bit(33, '0')
            ans3 = bit(33, '0')
            ans4 = bit(33, '0')
            ans5 = bit(33, '0')
            ans6 = bit(33, '0')
            ans7 = bit(33, '0')
            for i in range(len(exp_v_signed)):
                ans0[i] = (~(mask[i] ^ val7[i]) & (mask[i])).bin
            mask = ans0 if ans0 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans1[i] = (~(mask[i] ^ val6[i]) & (mask[i])).bin
            mask = ans1 if ans1 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans2[i] = (~(mask[i] ^ val5[i]) & (mask[i])).bin
            mask = ans2 if ans2 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans3[i] = (~(mask[i] ^ val4[i]) & (mask[i])).bin
            mask = ans3 if ans3 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans4[i] = (~(mask[i] ^ val3[i]) & (mask[i])).bin
            mask = ans4 if ans4 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans5[i] = (~(mask[i] ^ val2[i]) & (mask[i])).bin
            mask = ans5 if ans5 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans6[i] = (~(mask[i] ^ val1[i]) & (mask[i])).bin
            mask = ans6 if ans6 != bit(1, '0') else mask
            for i in range(len(exp_v_signed)):
                ans7[i] = (~(mask[i] ^ val0[i]) & (mask[i])).bin
            ans7 = mask if ans7 == bit(1, '0') else ans7

            if ans7[0] == bit(1, '1'):
                o_max_exp = exp_v_signed[0]
            elif ans7[1] == bit(1, '1'):
                o_max_exp = exp_v_signed[1]
            elif ans7[2] == bit(1, '1'):
                o_max_exp = exp_v_signed[2]
            elif ans7[3] == bit(1, '1'):
                o_max_exp = exp_v_signed[3]
            elif ans7[4] == bit(1, '1'):
                o_max_exp = exp_v_signed[4]
            elif ans7[5] == bit(1, '1'):
                o_max_exp = exp_v_signed[5]
            elif ans7[6] == bit(1, '1'):
                o_max_exp = exp_v_signed[6]
            elif ans7[7] == bit(1, '1'):
                o_max_exp = exp_v_signed[7]
            elif ans7[8] == bit(1, '1'):
                o_max_exp = exp_v_signed[8]
            elif ans7[9] == bit(1, '1'):
                o_max_exp = exp_v_signed[9]
            elif ans7[10] == bit(1, '1'):
                o_max_exp = exp_v_signed[10]
            elif ans7[11] == bit(1, '1'):
                o_max_exp = exp_v_signed[11]
            elif ans7[12] == bit(1, '1'):
                o_max_exp = exp_v_signed[12]
            elif ans7[13] == bit(1, '1'):
                o_max_exp = exp_v_signed[13]
            elif ans7[14] == bit(1, '1'):
                o_max_exp = exp_v_signed[14]
            elif ans7[15] == bit(1, '1'):
                o_max_exp = exp_v_signed[15]
            elif ans7[16] == bit(1, '1'):
                o_max_exp = exp_v_signed[16]
            elif ans7[17] == bit(1, '1'):
                o_max_exp = exp_v_signed[17]
            elif ans7[18] == bit(1, '1'):
                o_max_exp = exp_v_signed[18]
            elif ans7[19] == bit(1, '1'):
                o_max_exp = exp_v_signed[19]
            elif ans7[20] == bit(1, '1'):
                o_max_exp = exp_v_signed[20]
            elif ans7[21] == bit(1, '1'):
                o_max_exp = exp_v_signed[21]
            elif ans7[22] == bit(1, '1'):
                o_max_exp = exp_v_signed[22]
            elif ans7[23] == bit(1, '1'):
                o_max_exp = exp_v_signed[23]
            elif ans7[24] == bit(1, '1'):
                o_max_exp = exp_v_signed[24]
            elif ans7[25] == bit(1, '1'):
                o_max_exp = exp_v_signed[25]
            elif ans7[26] == bit(1, '1'):
                o_max_exp = exp_v_signed[26]
            elif ans7[27] == bit(1, '1'):
                o_max_exp = exp_v_signed[27]
            elif ans7[28] == bit(1, '1'):
                o_max_exp = exp_v_signed[28]
            elif ans7[29] == bit(1, '1'):
                o_max_exp = exp_v_signed[29]
            elif ans7[30] == bit(1, '1'):
                o_max_exp = exp_v_signed[30]
            elif ans7[31] == bit(1, '1'):
                o_max_exp = exp_v_signed[31]
            else:
                o_max_exp = exp_acc_signed


            # Align shifter
            # Shift amount should be ceil(log2(align shifter width))
            shamt: List[int] = []
            for i in range(len(exp_v_signed)):
                shamt.append(int(o_max_exp - exp_v_signed[i]))
            # Acc shift
            shamt.append(int(o_max_exp - exp_acc_signed))

            #print('exp max tree', exp_max_tree)
            #print(exp_v)
            #print(int(exp_v_signed[0]))
            #print('max exp', int(o_max_exp))
            #print('shamt', shamt)

            # mantissa: h.mmm_mmmm
            # shifted mantissa: x.xxx_xxxx_xxxx_...._xxxx
            #                   < align shifter length  >
            # append zero to LSB amount: align_bitwidth - (mantissa bits + hidden bit + 1)
            # 64 entries -> add 6 bits
            # 2^n entries -> add n bits
            # Extend to align shifter length+C+S+6
            # Extend to align shifter length+C+S+tree_level
            # Extended mantissa: 0000_0000_h.mmm_mmmm0_...._0000
            #                    <   align shifter length+2+tree_level  >
            # Make 2's complement
            # Make mantissa to signed bitstring
            # signed mantissa: SSSS_SSSS_h.mmm_mmmm_...._xxxx
            #                    <   align shifter length+2+tree_level  >
            # Alignment shift
            # added mantissa: CSxx_xxxx_x.xxx_xxxx_...._xxxx
            #                    <   align shifter length+2+tree_level  >
            # Carry bit is not used, so remove it
            align_bit = self.align_bitwidth
            # adder bit: bitwidth of MSB extension bits
            adder_bit = 2 + tree_level
            # sum bit: bitwidth of adder tree
            sum_bit = align_bit + adder_bit
            precision_bit = fp32_config.mantissa_bits + 1
            mant_v_sign: List[sbit] = []
            for i in range(len(sign_v)):
                mant_sign = sbit(sum_bit, f'{adder_bit * "0"}{mant_v_us[i]}{(align_bit - precision_bit) * "0"}')
                if sign_v[i] == bit(1, '1'):
                    mant_v_sign.append(-mant_sign)
                else:
                    mant_v_sign.append(mant_sign)
            # accumulator
            mant_acc_sign = sbit(sum_bit, f'{adder_bit * "0"}{mant_acc_us}{(align_bit - precision_bit) * "0"}')
            if sign_acc == bit(1, '1'):
                mant_v_sign.append(-mant_acc_sign)
            else:
                mant_v_sign.append(mant_acc_sign)
            #print('mant_v_sign', mant_v_sign)
            mant_v_sign_print: List[str] = []
            for i in mant_v_sign:
                mant_v_sign_print.append(hex(int(i.bin, 2)))
            #print('mant_v_sign', mant_v_sign_print)

            # mantissa shift
            # shifted & signed mantissa: Sx.xxx_xxxx_xxxx_...._xxxx
            #                            < align shifter length+1 >
            mant_v_aligned: List[sbit] = []
            for i in range(len(mant_v_sign)):
                mant_aligned = mant_v_sign[i].arith_rshift(shamt[i])
                mant_v_aligned.append(mant_aligned)
            
            #print('mant_v_aligned', mant_v_aligned)
            mant_v_aligned_print: List[str] = []
            for i in mant_v_aligned:
                mant_v_aligned_print.append(hex(int(i.bin, 2)))
            #print('mant_v_aligned', mant_v_aligned_print)


            # Adder tree
            # 64 entries -> 6 bits
            # added mantissa: CSxx_xxxx.xxxx_xxxx_...._xxxx
            #                 <   align shifter length+8   >
            # Carry bit is not used, so remove it
            mant_add = sbit(sum_bit, '0')
            for i in range(len(mant_v_aligned)):
                mant_add = mant_add + mant_v_aligned[i]

            #print('mant_add: ', repr(mant_add))
            #print('mant_add: ', hex(int(mant_add.bin, 2)))

            # Post adder & accumulation
            # Sign bitpos: align shifter length + 7
            mant_add_sign = mant_add[sum_bit-2]
            zero_mant_result = mant_add[sum_bit-3:0] == sbit(sum_bit-2, '0')
            mant_add_nocarry = mant_add[sum_bit-2:0]
            # mantissa result before sign removal: Sxx_xxxx.xxxx_xxxx_...._xxxx
            # mantissa result: xx_xxxx.xxxx_xxxx_...._xxxx
            #                  <  align shifter length+6 >
            mant_add_result_before_sign_remove = ubit(sum_bit-1, f'{(-mant_add_nocarry).bin if mant_add_sign == sbit(1, "1") else mant_add_nocarry.bin}')
            mant_add_result = mant_add_result_before_sign_remove[sum_bit-3:0]

            #print('mant_add_sign', repr(mant_add_sign))
            #print('mant_add_nocarry', repr(mant_add_nocarry))
            #print('inv mant_add_nocarry', repr(-mant_add_nocarry))
            #print('mant_add_before_sign', repr(mant_add_result_before_sign_remove))
            #print('mant_add_result', repr(mant_add_result))

            # Leading zero count for close path
            # to sum_bit - (tree_level + 2)
            # rshamt bitwidth = ceil(log2(tree_level))
            rshamt = 0
            for i in range(tree_level):
                if mant_add_result[sum_bit-(i+3)].bin == '1':
                    rshamt = tree_level - i

            close_path = mant_add_result >> rshamt
        
            # Leading zero count for far path
            # clz_in is from point
            # ex) for 64 entries, last rshamt = 9, which is sum_bit - (tree_level + 3) where tree level = 6
            clz_in = mant_add_result[sum_bit-(tree_level+3):0]
            if hwutil.leading_zero_count(clz_in) < (sum_bit-precision_bit):
                lshamt = hwutil.leading_zero_count(clz_in)
            else:
                lshamt = 0
            #print('rshamt', rshamt)
            #print('lshamt', lshamt)
            
    #        far_path = clz_in << lshamt
            far_path = mant_add_result << lshamt
            rnd_in = far_path if rshamt == 0 else close_path

            #print('close_path', repr(close_path))
            #print('far_path', repr(far_path))

            # Round
            # round in: 1.xxxx_xxx|R_ssss_...._xxxx
            #                  < align shifter length >
            #print('rnd_in: ', repr(rnd_in))
            round_bitpos = 1 + fp32_config.mantissa_bits+1
            rnd = rnd_in[align_bit-round_bitpos]
            sticky = rnd_in[align_bit-round_bitpos-1:0].reduceor()
            round = (rnd & sticky) | (rnd_in[align_bit-round_bitpos+1] & rnd & ~sticky)
            #print('round', repr(round))

            round_up = round.concat(bit(align_bit - round_bitpos, '0'))
            normed = ubit(align_bit+1, (rnd_in + round_up).bin)

            #print('normed', repr(normed))

            # To handle overflow case, increase 1 bit
            if zero_mant_result:
                t_exp = ubit(fp32_config.exponent_bits+1, '0')
            elif rshamt > 0:
                t_exp = ubit(fp32_config.exponent_bits+1, bin(int(o_max_exp) + rshamt))
            else:
                t_exp = ubit(fp32_config.exponent_bits+1, bin(int(o_max_exp) - lshamt))
            
            # Post normalization
            postnorm_flag = normed[align_bit].bin == '1'
            out_sign = bit(1, mant_add_sign.bin)
            out_exp_overflow = t_exp + ubit(1, '1') if postnorm_flag else t_exp
            # 23bits of fraction
            #out_frac_overflow = normed[align_bit-1:align_bit-7] if postnorm_flag else normed[align_bit-2:align_bit-8]
            out_frac_overflow = normed[align_bit-1:align_bit-fp32_config.mantissa_bits] if postnorm_flag else normed[align_bit-2:align_bit-fp32_config.mantissa_bits-1]
            #print('postnorm flag', postnorm_flag)

            # BF16 assumes denormalized number as zero
            # There is no denormalized output in addition
            # Handle infinity exponent
            # You can use out_exp_overflow[Bfloat16.exponent_bits] == sbit(1, '1') also
            if out_exp_overflow >= ubit(fp32_config.exponent_bits+1, bin((1 << fp32_config.exponent_bits) - 1)):
                out_exp = ubit(fp32_config.exponent_bits, bin((fp32_config.exp_max + fp32_config.bias)))
                out_frac = ubit(fp32_config.mantissa_bits, '0')
            else:
                out_exp = out_exp_overflow[fp32_config.exponent_bits-1:0]
                out_frac = out_frac_overflow

            #print('t_exp',t_exp)
        else:
            out_sign = out_sign
            out_exp = out_exp
            out_frac = out_frac

        #print(out_sign, out_exp, out_frac)
        return out_sign, out_exp, out_frac

    def execute_multi_path(self) -> FPBitT:
        # Special cases
        # Need handle accumulator later...
        # input
        isnormal = False
        # Conditions
        # In vector if there's any nan -> nan
        # nan inputs clearance
        # inf + inf = inf
        # -inf + -inf = -inf
        # In vector if there's any inf with same sign
        any_nan = False
        any_inf = False
        inf_num = 0
        neg_inf_num = 0
        inf_sign = 0
        for i in self.vector:
            any_nan |= isnan(i)
            if isinf(i):
                inf_num = inf_num + 1
                if i[0] == bit(1, '1'):    # negative sign
                    neg_inf_num = neg_inf_num + 1

        any_nan |= isnan(self.acc)
        if isinf(self.acc):
            inf_num = inf_num + 1
            if self.acc[0] == bit(1, '1') >> 31:   # negative sign
                neg_inf_num = neg_inf_num + 1

        all_inf_same_sign = False
        if inf_num != 0:
            any_inf = True
            if neg_inf_num == 0:
                all_inf_same_sign = True
                inf_sign = 0
            elif neg_inf_num == inf_num:
                all_inf_same_sign = False
                inf_sign = 1

        # In vector if there's any nan -> nan
        # nan inputs clearance
        if any_nan:
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        # inf + inf = inf
        # -inf + -inf = -inf
        # In vector if there's any inf with same sign
        elif any_inf and all_inf_same_sign:
            ret_sign_sp = bit(1, str(inf_sign))
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, fp32_config.mantissa_bits * '0')
        # -inf + +inf = nan
        # In vector if there's any inf with different sign
        elif any_inf and not all_inf_same_sign:
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(fp32_config.exponent_bits, bin(fp32_config.exp_max + fp32_config.bias))
            ret_mant_sp = bit(fp32_config.mantissa_bits, bin(fp32_config.mant_max))
        else:
            ret_sign_sp = bit(1, '0')
            ret_exp_sp = bit(1, '0')
            ret_mant_sp = bit(1, '0')
            isnormal = True

        # Extract vectors
        # Decompose elements
#        decompsed_vector = (map(bf16.Bfloat16.decompose_bf16, self.vector_elements))
#        sign_v, exp_v, mant_v = zip(*decompsed_vector)
        # make vector of sign, exp, mant
        sign_v, exp_v, mant_v = [list(v) for v in zip(*self.vector)]
        # JMG
        
        sign_acc, exp_acc, mant_nohidden_acc = self.acc
        exp_acc_signed = bit(fp32_config.exponent_bits, f'{exp_acc.bin}')
        # Treat acc as fp32 in module -> mantissa to 24bits
        mant_acc_us = ubit(fp32_config.mantissa_bits + 1, f'{exp_acc_signed.reduceor()}{mant_nohidden_acc.bin}')
        

        # Process elements
        # Exponent to signed bitstring
        # FIX: For 0 input, use 0 as hidden bit
        exp_v_signed: List[bit] = []
        hidden_bit: List[bit] = []
        for i in exp_v:
            exp_v_signed.append(bit(fp32_config.exponent_bits, f'{i}'))
            hidden_bit.append(i.reduceor())
        
        
        # Adjust hidden bit to mantissa
        # FIX: Alignment size = inner sum mantissa bitwidth, 24 + align_bitwidth
        # FIX: for bf16 input, concat 16*'0' + align_bitwidth
        # FIX: For 0 input, use 0 as hidden bit
        mant_v_us: List[ubit] = []
        for i in range(len(mant_v)):
            mant_v_us.append(ubit(fp32_config.mantissa_bits + 1, f'{hidden_bit[i]}{mant_v[i]}'))
        
        

        # Normal case
        #Max tree

        # Find tree level
        element_num = self.vector_element_num
        element_num_shift = element_num
        tree_level = 1
        while element_num_shift != 1:
            element_num_shift = element_num_shift >> 1
            tree_level += 1

        '''
        # Exponent max tree
        # Make zero initialized dual list of tree
        # [[0] * 2**(tree_level), [0] * 2**(tree_level-1), ..., [0]]
        # Initialize first stage with exp_v_signed
        # Compare previous stage (i) and record it to current stage(i+1)
        exp_max_tree: List[List[sbit]] = [[sbit(fp32_config.exponent_bits + 2, '0'*(fp32_config.exponent_bits + 2))] * (2**(tree_level-i)) for i in range(tree_level+1)]
        # Initialize
        for i in range(len(exp_v_signed)):
            exp_max_tree[0][i] = exp_v_signed[i]
        # Compare and record
        for i in range(tree_level):
            for j in range(len(exp_max_tree[i+1])):
                exp_max_tree[i+1][j] = exp_max_tree[i][2*j] if (exp_max_tree[i][2*j] > exp_max_tree[i][2*j+1]) else exp_max_tree[i][2*j+1]
        exp_max_vector = exp_max_tree[tree_level][0]
        o_max_exp = exp_max_vector if exp_max_vector > exp_acc_signed else exp_acc_signed
        '''
        mask = bit(33, bin(0x1ffffffff))
        val0 = bit(33, '0')
        val1 = bit(33, '0')
        val2 = bit(33, '0')
        val3 = bit(33, '0')
        val4 = bit(33, '0')
        val5 = bit(33, '0')
        val6 = bit(33, '0')
        val7 = bit(33, '0')
        for i in range(self.vector_element_num):
            val0[i] = exp_v_signed[i][0].bin
            val1[i] = exp_v_signed[i][1].bin
            val2[i] = exp_v_signed[i][2].bin
            val3[i] = exp_v_signed[i][3].bin
            val4[i] = exp_v_signed[i][4].bin
            val5[i] = exp_v_signed[i][5].bin
            val6[i] = exp_v_signed[i][6].bin
            val7[i] = exp_v_signed[i][7].bin
        val0[32] = exp_acc_signed[0].bin
        val1[32] = exp_acc_signed[1].bin
        val2[32] = exp_acc_signed[2].bin
        val3[32] = exp_acc_signed[3].bin
        val4[32] = exp_acc_signed[4].bin
        val5[32] = exp_acc_signed[5].bin
        val6[32] = exp_acc_signed[6].bin
        val7[32] = exp_acc_signed[7].bin

        ans0 = bit(33, '0')
        ans1 = bit(33, '0')
        ans2 = bit(33, '0')
        ans3 = bit(33, '0')
        ans4 = bit(33, '0')
        ans5 = bit(33, '0')
        ans6 = bit(33, '0')
        ans7 = bit(33, '0')
        for i in range(self.vector_element_num+1):
            ans0[i] = (~(mask[i] ^ val7[i]) & (mask[i])).bin
        mask = ans0 if ans0 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans1[i] = (~(mask[i] ^ val6[i]) & (mask[i])).bin
        mask = ans1 if ans1 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans2[i] = (~(mask[i] ^ val5[i]) & (mask[i])).bin
        mask = ans2 if ans2 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans3[i] = (~(mask[i] ^ val4[i]) & (mask[i])).bin
        mask = ans3 if ans3 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans4[i] = (~(mask[i] ^ val3[i]) & (mask[i])).bin
        mask = ans4 if ans4 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans5[i] = (~(mask[i] ^ val2[i]) & (mask[i])).bin
        mask = ans5 if ans5 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans6[i] = (~(mask[i] ^ val1[i]) & (mask[i])).bin
        mask = ans6 if ans6 != bit(1, '0') else mask
        for i in range(self.vector_element_num+1):
            ans7[i] = (~(mask[i] ^ val0[i]) & (mask[i])).bin
        ans7 = mask if ans7 == bit(1, '0') else ans7

        if ans7[0] == bit(1, '1'):
            o_max_exp = exp_v_signed[0]
        elif ans7[1] == bit(1, '1'):
            o_max_exp = exp_v_signed[1]
        elif ans7[2] == bit(1, '1'):
            o_max_exp = exp_v_signed[2]
        elif ans7[3] == bit(1, '1'):
            o_max_exp = exp_v_signed[3]
        elif ans7[4] == bit(1, '1'):
            o_max_exp = exp_v_signed[4]
        elif ans7[5] == bit(1, '1'):
            o_max_exp = exp_v_signed[5]
        elif ans7[6] == bit(1, '1'):
            o_max_exp = exp_v_signed[6]
        elif ans7[7] == bit(1, '1'):
            o_max_exp = exp_v_signed[7]
        elif ans7[8] == bit(1, '1'):
            o_max_exp = exp_v_signed[8]
        elif ans7[9] == bit(1, '1'):
            o_max_exp = exp_v_signed[9]
        elif ans7[10] == bit(1, '1'):
            o_max_exp = exp_v_signed[10]
        elif ans7[11] == bit(1, '1'):
            o_max_exp = exp_v_signed[11]
        elif ans7[12] == bit(1, '1'):
            o_max_exp = exp_v_signed[12]
        elif ans7[13] == bit(1, '1'):
            o_max_exp = exp_v_signed[13]
        elif ans7[14] == bit(1, '1'):
            o_max_exp = exp_v_signed[14]
        elif ans7[15] == bit(1, '1'):
            o_max_exp = exp_v_signed[15]
        elif ans7[16] == bit(1, '1'):
            o_max_exp = exp_v_signed[16]
        elif ans7[17] == bit(1, '1'):
            o_max_exp = exp_v_signed[17]
        elif ans7[18] == bit(1, '1'):
            o_max_exp = exp_v_signed[18]
        elif ans7[19] == bit(1, '1'):
            o_max_exp = exp_v_signed[19]
        elif ans7[20] == bit(1, '1'):
            o_max_exp = exp_v_signed[20]
        elif ans7[21] == bit(1, '1'):
            o_max_exp = exp_v_signed[21]
        elif ans7[22] == bit(1, '1'):
            o_max_exp = exp_v_signed[22]
        elif ans7[23] == bit(1, '1'):
            o_max_exp = exp_v_signed[23]
        elif ans7[24] == bit(1, '1'):
            o_max_exp = exp_v_signed[24]
        elif ans7[25] == bit(1, '1'):
            o_max_exp = exp_v_signed[25]
        elif ans7[26] == bit(1, '1'):
            o_max_exp = exp_v_signed[26]
        elif ans7[27] == bit(1, '1'):
            o_max_exp = exp_v_signed[27]
        elif ans7[28] == bit(1, '1'):
            o_max_exp = exp_v_signed[28]
        elif ans7[29] == bit(1, '1'):
            o_max_exp = exp_v_signed[29]
        elif ans7[30] == bit(1, '1'):
            o_max_exp = exp_v_signed[30]
        elif ans7[31] == bit(1, '1'):
            o_max_exp = exp_v_signed[31]
        else:
            o_max_exp = exp_acc_signed

        # Align shifter
        # Shift amount should be ceil(log2(align shifter width))
        shamt: List[int] = []
        for i in range(self.vector_element_num):
            shamt.append(int(o_max_exp - exp_v_signed[i]))
        # Acc shift
        shamt_acc = int(o_max_exp - exp_acc_signed)

        
        

        # mantissa: h.mmm_mmmm
        # shifted mantissa: x.xxx_xxxx_xxxx_...._xxxx
        #                   < align shifter length  >
        # append zero to LSB amount: align_bitwidth - (mantissa bits + hidden bit + 1)
        # 64 entries -> add 6 bits
        # 2^n entries -> add n bits
        # Extend to align shifter length+C+S+6
        # Extend to align shifter length+C+S+tree_level
        # Extended mantissa: 0000_0000_h.mmm_mmmm0_...._0000
        #                    <   align shifter length+2+tree_level  >
        # Make 2's complement
        # Make mantissa to signed bitstring
        # signed mantissa: SSSS_SSSS_h.mmm_mmmm_...._xxxx
        #                    <   align shifter length+2+tree_level  >
        # Alignment shift
        # added mantissa: CSxx_xxxx_x.xxx_xxxx_...._xxxx
        #                    <   align shifter length+2+tree_level  >
        # Carry bit is not used, so remove it

        align_bit = self.align_bitwidth
        # adder bit: bitwidth of MSB extension bits
        #adder_bit = 2 + tree_level
        adder_bit = tree_level
        # sum bit: bitwidth of adder tree
        sum_bit = align_bit + adder_bit
        precision_bit = fp32_config.mantissa_bits + 1
        mant_v_sign: List[bit] = []
        for i in range(self.vector_element_num):
            mant_sign = bit(precision_bit + 1, f'0{mant_v_us[i]}')
            if sign_v[i] == bit(1, '1'):
                mant_v_sign.append(-mant_sign)
            else:
                mant_v_sign.append(mant_sign)
        # accumulator
        if sign_acc == bit(1, '1'):
            mant_acc_sign = -bit(precision_bit + 1, f'0{mant_acc_us}')
        else:
            mant_acc_sign = bit(precision_bit + 1, f'0{mant_acc_us}')
        

        # mantissa shift
        # shifted & signed mantissa: Sx.xxx_xxxx_xxxx_...._xxxx
        #                            < align shifter length+1 >
        mant_v_aligned: List[sbit] = []
        for i in range(self.vector_element_num):
            mant_aligned = sbit(align_bit, '01') if shamt[i] >= 32 else sbit(align_bit, f'{mant_v_sign[i]}{(align_bit - precision_bit - 1) * "0"}').arith_rshift(shamt[i])
            mant_v_aligned.append(mant_aligned)
        mant_acc_aligned = sbit(align_bit, '01') if shamt_acc >= 32 else sbit(align_bit, f'{mant_acc_sign}{(align_bit - precision_bit - 1) * "0"}').arith_rshift(shamt_acc)
        


        # Adder tree
        # 64 entries -> 6 bits
        # added mantissa: CSxx_xxxx.xxxx_xxxx_...._xxxx
        #                 <   align shifter length+8   >
        # Carry bit is not used, so remove it
        # 33bit 16 entries
        l1_add_tree = list()
        for i in range(self.vector_element_num//2**1):
            l1_sum = sbit(mant_v_aligned[i].bitwidth+1, (mant_v_aligned[i] + mant_v_aligned[self.vector_element_num//2**1 + i]).bin)
            l1_add_tree.append(l1_sum)
        # 34bit 8 entries
        l2_add_tree = list()
        for i in range(self.vector_element_num//2**2):
            mant_sum = l1_add_tree[i] + l1_add_tree[self.vector_element_num//2**2 + i]
            #l2_sum = sbit(l1_add_tree[i].bitwidth+1, (l1_add_tree[i] + l1_add_tree[self.vector_element_num//2**2 + i]).bin)
            l2_sum = sbit(l1_add_tree[i].bitwidth+1, mant_sum.bin)
            l2_add_tree.append(l2_sum)
        # 35bit 4 entries
        l3_add_tree = list()
        for i in range(self.vector_element_num//2**3):
            l3_sum = sbit(l2_add_tree[i].bitwidth+1, (l2_add_tree[i] + l2_add_tree[self.vector_element_num//2**3 + i]).bin)
            l3_add_tree.append(l3_sum)
        # 36bit 2 entries
        l4_add_tree = list()
        for i in range(self.vector_element_num//2**4):
            l4_sum = sbit(l3_add_tree[i].bitwidth+1, (l3_add_tree[i] + l3_add_tree[self.vector_element_num//2**4 + i]).bin)
            l4_add_tree.append(l4_sum)
        # 37bit mant final sum
        l5_add_tree = sbit(l4_add_tree[0].bitwidth+1, (l4_add_tree[0] + l4_add_tree[1]).bin)
        # 38bit final sum with acc
        mant_add_before_sign = sbit(l5_add_tree.bitwidth+1, (l5_add_tree + mant_acc_aligned).bin)
        mant_add = bit(mant_add_before_sign.bitwidth, mant_add_before_sign.bin)

        mant_add_test = sbit(38, '0')
        for i in range(len(mant_v_aligned)):
            mant_add_test = sbit(38, (mant_add_test + mant_v_aligned[i]).bin)

        

        # Post adder & accumulation
        ret_sign = mant_add[sum_bit-1]
        mant_abs = -mant_add[sum_bit-2:0] if mant_add[sum_bit-1].bin == '1' else mant_add[sum_bit-2:0]
        # mant_abs
        # c cccc c1.ff ffff ffff ffff ffff ffff fppp pppp
        # x xxxx xx.xx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        

        # normalization path
        norm_shamt = hwutil.leading_zero_count(mant_abs[sum_bit-2:align_bit-2])
        # which one is right?
        max_norm_shamt = (sum_bit-2)-(align_bit-1)+1
        
        norm_path_exp = o_max_exp + bit(3, bin(max_norm_shamt - norm_shamt))
        #norm_path_exp = o_max_exp - bit(3, bin(norm_shamt))
        # normed_mant = 1.xx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        # fp32:                                       lrss ssss
        normed_mant = bit(31, (mant_abs[sum_bit-2:align_bit-precision_bit-2] << norm_shamt).bin)
        norm_sticky = bit(1, bin(int(mant_abs[align_bit-precision_bit-3:0]) != 0))
        norm_fp32_lsb = normed_mant[align_bit-precision_bit-1]
        norm_fp32_r = normed_mant[align_bit-precision_bit-2]
        norm_fp32_s = bit(1, bin(int(normed_mant[align_bit-precision_bit-3:0]) !=0)) & norm_sticky
        norm_fp32_rup = (norm_fp32_lsb & norm_fp32_r) | (norm_fp32_r & norm_fp32_s)
        fp32_norm_mant = normed_mant[align_bit-3:align_bit-precision_bit-1]
        norm_fp32_exp = norm_path_exp
        # fp32's normalization and post normalization
        if((int(fp32_norm_mant) == (1<<precision_bit-1)-1) & int(norm_fp32_rup)):
            #fp32_norm_mant = bit(fp32_norm_mant.bitwidth, '0')
            fp32_norm_mant.set_bin('0')
            norm_fp32_exp = norm_fp32_exp + bit(1, '1')
        else:
            fp32_norm_mant = fp32_norm_mant + norm_fp32_rup
        
        
        # normed_mant = 1.xx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        # bf16:                   lrss ssss ssss ssss ssss ssss 
        norm_bf16_lsb = normed_mant[align_bit-(precision_bit-16)-1]
        norm_bf16_r = normed_mant[align_bit-(precision_bit-16)-2]
        norm_bf16_s = bit(1, bin((int(normed_mant[align_bit-(precision_bit-16)-3:0]) !=0))) & norm_sticky
        norm_bf16_rup = (norm_bf16_lsb & norm_bf16_r) | (norm_bf16_r & norm_bf16_s)
        bf16_norm_mant = normed_mant[align_bit-3:align_bit-(precision_bit-16)-1]
        norm_bf16_exp = norm_path_exp
        # bf16's normalization and post normalization
        if((int(bf16_norm_mant) == (1<<precision_bit-1-16)-1) & int(norm_bf16_rup)):
            #bf16_norm_mant = bit(bf16_norm_mant.bitwidth, '0')
            bf16_norm_mant.set_bin('0')
            norm_bf16_exp = norm_bf16_exp + bit(1, '1')
        else:
            bf16_norm_mant = bf16_norm_mant + norm_bf16_rup
        
                
        # norm path output
        norm_path_mant = bf16_norm_mant.concat(bit(16, '0')) if self.mod==3 else fp32_norm_mant
        norm_exp = norm_bf16_exp if self.mod==3 else norm_fp32_exp
        

        # even path: 1.ff ffff ffff ffff ffff ffff fppp pppp
        # fp32:                                    lrss ssss
        even_mant = mant_abs[align_bit-2:align_bit-precision_bit-1]
        even_path_exp = o_max_exp
        even_sticky = bit(1, bin(int(mant_abs[align_bit-precision_bit-2:0]) != 0))
        even_fp32_lsb = mant_abs[align_bit-precision_bit-1]
        even_fp32_r = mant_abs[align_bit-precision_bit-2]
        even_fp32_rup = (even_fp32_lsb & even_fp32_r) | (even_fp32_r & even_sticky)
        fp32_even_mant = even_mant[precision_bit-2:0]
        even_fp32_exp = even_path_exp
        # fp32's normalization and post normalization
        if((int(fp32_even_mant) == (1<<precision_bit-1)-1) & int(even_fp32_rup)):
            #fp32_even_mant = bit(fp32_even_mant.bitwidth, '0')
            fp32_even_mant.set_bin('0')
            even_fp32_exp = even_fp32_exp + bit(1, '1')
        else:
            fp32_even_mant = fp32_even_mant + even_fp32_rup
        
        # even path: 1.ff ffff ffff ffff ffff ffff fppp pppp
        # bf16:                lrss ssss ssss ssss ssss ssss
        even_bf16_lsb = mant_abs[align_bit-(precision_bit-16)-1]
        even_bf16_r = mant_abs[align_bit-(precision_bit-16)-2]
        even_bf16_s = bit(1, bin(int(mant_abs[align_bit-(precision_bit-16)-3:0]) != 0))
        even_bf16_rup = (even_bf16_lsb & even_bf16_r) | (even_bf16_r & even_bf16_s)
        bf16_even_mant = mant_abs[align_bit-3:align_bit-(precision_bit-16)-1]
        even_bf16_exp = even_path_exp
        # bf16's normalization and post normalization
        if((int(bf16_even_mant) == ((1<<precision_bit-1-16)-1)) & int(even_bf16_rup)):
            #bf16_even_mant = bit(bf16_even_mant.bitwidth, '0')
            bf16_even_mant.set_bin('0')
            even_bf16_exp = even_bf16_exp + bit(1, '1')
        else:
            bf16_even_mant = bf16_even_mant + even_bf16_rup
        
        # even path output
        even_path_mant = bf16_even_mant.concat(bit(16, '0')) if self.mod==3 else fp32_even_mant
        even_exp = even_bf16_exp if self.mod==3 else even_fp32_exp
        
        # cancellation path: 1.x xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        # fp32:                                            lrs ssss
        can_shamt = hwutil.leading_zero_count(mant_abs[align_bit-3:0])
        
        # which one is right?
        #can_path_exp = o_max_exp - bit(5, bin(can_shamt))
        can_path_exp = bit(o_max_exp.bitwidth, (o_max_exp - bit(5, bin(can_shamt)) - bit(1, '1')).bin)
        
        can_mant = mant_abs[align_bit-3:0] << can_shamt
        can_fp32_lsb = can_mant[align_bit-precision_bit-2]
        can_fp32_r = can_mant[align_bit-precision_bit-3]
        can_fp32_s = bit(1, bin(int(can_mant[align_bit-precision_bit-4:0]) != 0))
        can_fp32_rup = (can_fp32_lsb & can_fp32_r) | (can_fp32_r & can_fp32_s)
        fp32_can_mant = can_mant[align_bit-4:align_bit-precision_bit-2]
        can_fp32_exp = can_path_exp
        
        # fp32's normalization and post normalization
        if((int(fp32_can_mant) == (1<<precision_bit-1)-1) & int(can_fp32_rup)):
            #fp32_can_mant = bit(fp32_can_mant.bitwidth, '0')
            fp32_can_mant.set_bin('0')
            can_fp32_exp = can_fp32_exp + bit(1, '1')
        else:
            fp32_can_mant = fp32_can_mant + can_fp32_rup
        
        
        
        # cancellation path: 1.x xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        # bf16:                        lrs ssss ssss ssss ssss ssss
        can_bf16_lsb = can_mant[align_bit-(precision_bit-16)-2]
        can_bf16_r = can_mant[align_bit-(precision_bit-16)-3]
        can_bf16_s = bit(1, bin(int(can_mant[align_bit-(precision_bit-16)-4:0]) != 0))
        can_bf16_rup = (can_bf16_lsb & can_bf16_r) | (can_bf16_r & can_fp32_s & can_bf16_s)
        bf16_can_mant = can_mant[align_bit-4:align_bit-(precision_bit-16)-2]
        can_bf16_exp = can_path_exp
        # bf16's normalization and post normalization
        #print('bf16_can_mant', repr(bf16_can_mant))
        if((int(bf16_can_mant) == ((1<<precision_bit-1-16)-1)) & int(can_bf16_rup)):
            #bf16_can_mant = bit(bf16_can_mant.bitwidth, '0')
            bf16_can_mant.set_bin('0')
            can_bf16_exp = can_bf16_exp + bit(1, '1')
        else:
            bf16_can_mant = bf16_can_mant + can_bf16_rup
        
        # cancellation path output
        can_path_mant = bf16_can_mant.concat(bit(16, '0')) if self.mod==3 else fp32_can_mant
        can_exp = can_bf16_exp if self.mod==3 else can_fp32_exp

        # which one is right?
        #if (norm_shamt != 6):
        #if (norm_shamt <= 6):
        if (norm_shamt < 6):
            ret_exp = norm_exp
            ret_mant = norm_path_mant
            #print('NORM')
        elif (mant_abs[align_bit-2].bin == '1'):
            ret_exp = even_exp
            ret_mant = even_path_mant
            #print('EVEN')
        else:
            ret_exp = can_exp
            ret_mant = can_path_mant
            #print('CAN')

        if isnormal:
            ret_sign = ret_sign
            ret_exp = ret_exp
            ret_mant = ret_mant
        else:
            ret_sign = ret_sign_sp
            ret_exp = ret_exp_sp
            ret_mant = ret_mant_sp
        
        # debug prints
        #print(sign_v)
        #print('mant_nohidden_acc', repr(mant_nohidden_acc))
        #print('mant_acc_us', repr(mant_acc_us))
        #print('exp_v', list(map(int, exp_v)))
        #print('mant_v_us', list(map(bit.__hex__, mant_v_us)))
        #print([int(i) for i in exp_v_signed])
        #print('max exp', int(o_max_exp))
        #print('shamt', shamt)
        #print('mant_v_us', mant_v_us)
        #print('mant_v_sign', mant_v_sign)
        #print('mant_v_aligned', list(map(sbit.__hex__, mant_v_aligned)))
        #print('shamt_acc', shamt_acc)
        #print(mant_acc_sign)
        #print('mant_acc_aligned', mant_acc_aligned.__hex__())
        #print(l1_add_tree)
        #print(l2_add_tree)
        #print(l3_add_tree)
        #print(l4_add_tree)
        #print(repr(l5_add_tree))
        #print(list(map(sbit.__hex__, l1_add_tree)))
        #print(list(map(sbit.__hex__, l2_add_tree)))
        #print(list(map(sbit.__hex__, l3_add_tree)))
        #print(list(map(sbit.__hex__, l4_add_tree)))
        #print(l5_add_tree.__hex__())
        #print('mant_add: ', repr(mant_add))
        #print('mant_add: ', mant_add.__hex__())
        #print('mant_abs', repr(mant_abs))
        #print('mant_abs', mant_abs.__hex__())
        #print('max_norm_shamt', max_norm_shamt)
        #print('norm_shamt', norm_shamt)
        #print('normed_mant', repr(normed_mant))
        #print('fp32_norm_mant', repr(fp32_norm_mant))
        #print('normed_mant', normed_mant.__hex__())
        #print('fp32_norm_mant', fp32_norm_mant.__hex__())
        #print('bf16_norm_mant', bf16_norm_mant)
        #print('norm_bf16_exp', int(norm_bf16_exp))
        #print('norm_fp32_exp', int(norm_fp32_exp))
        #print('norm_exp', int(norm_exp))
        #print('mant_abs[align_bit-3:0]', repr(mant_abs[align_bit-3:0]))
        #print('mant_abs[align_bit-3:0]', mant_abs[align_bit-3:0].__hex__())
        #print('can_shamt', can_shamt)
        #print('can_path_exp', int(can_path_exp))
        #print('can_mant', repr(can_mant))
        #print('can_mant', can_mant.__hex__())
        #print('fp32_can_mant', repr(fp32_can_mant))
        #print('can_fp32_rup', repr(can_fp32_rup))
        #print('fp32_can_mant', fp32_can_mant.__hex__())

        #print('ret_sign', repr(ret_sign))
        #print('ret_exp', repr(ret_exp))
        #print('ret_mant', repr(ret_mant))
        return ret_sign, ret_exp, ret_mant
