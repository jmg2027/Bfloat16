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
    def __init__(self, vector: FPBitVectorT, acc: FPBitT) -> None:
        # vector_list = [Element, Element, ...]
        self.vector = vector
        self.acc = acc
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

    def excute(self) -> FPBitT:
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
            print(sign_v)
            sign_acc, exp_acc, mant_nohidden_acc = self.acc
            exp_acc_signed = sbit(fp32_config.exponent_bits + 2, f'0{exp_acc.bin}')
            # Treat acc as fp32 in module -> mantissa to 24bits
            mant_acc_us = ubit(fp32_config.mantissa_bits + 1, f'1{mant_nohidden_acc.bin}')
            print('mant_nohidden_acc', repr(mant_nohidden_acc))
            print('mant_acc_us', repr(mant_acc_us))

            # Process elements
            # Exponent to signed bitstring
            # FIX: For 0 input, use 0 as hidden bit
            exp_v_signed: List[sbit] = []
            hidden_bit: List[bit] = []
            for i in exp_v:
                exp_v_signed.append(sbit(fp32_config.exponent_bits + 2, f'0{i}'))
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

            # Align shifter
            # Shift amount should be ceil(log2(align shifter width))
            shamt: List[int] = []
            for i in range(len(exp_v_signed)):
                shamt.append(int(o_max_exp - exp_v_signed[i]))
            # Acc shift
            shamt.append(int(o_max_exp - exp_acc_signed))

            print('exp max tree', exp_max_tree)
            print(exp_v)
            print(int(exp_v_signed[0]))
            print('max exp', int(o_max_exp))
            print('shamt', shamt)

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
            print('mant_v_sign', mant_v_sign)
            mant_v_sign_print: List[str] = []
            for i in mant_v_sign:
                mant_v_sign_print.append(hex(int(i.bin, 2)))
            print('mant_v_sign', mant_v_sign_print)

            # mantissa shift
            # shifted & signed mantissa: Sx.xxx_xxxx_xxxx_...._xxxx
            #                            < align shifter length+1 >
            mant_v_aligned: List[sbit] = []
            for i in range(len(mant_v_sign)):
                mant_aligned = mant_v_sign[i].arith_rshift(shamt[i])
                mant_v_aligned.append(mant_aligned)
            
            print('mant_v_aligned', mant_v_aligned)
            mant_v_aligned_print: List[str] = []
            for i in mant_v_aligned:
                mant_v_aligned_print.append(hex(int(i.bin, 2)))
            print('mant_v_aligned', mant_v_aligned_print)


            # Adder tree
            # 64 entries -> 6 bits
            # added mantissa: CSxx_xxxx.xxxx_xxxx_...._xxxx
            #                 <   align shifter length+8   >
            # Carry bit is not used, so remove it
            mant_add = sbit(sum_bit, '0')
            for i in range(len(mant_v_aligned)):
                mant_add = mant_add + mant_v_aligned[i]

            print('mant_add: ', repr(mant_add))
            print('mant_add: ', hex(int(mant_add.bin, 2)))

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

            print('mant_add_sign', repr(mant_add_sign))
            print('mant_add_nocarry', repr(mant_add_nocarry))
            print('inv mant_add_nocarry', repr(-mant_add_nocarry))
            print('mant_add_before_sign', repr(mant_add_result_before_sign_remove))
            print('mant_add_result', repr(mant_add_result))

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
            print('rshamt', rshamt)
            print('lshamt', lshamt)
            
    #        far_path = clz_in << lshamt
            far_path = mant_add_result << lshamt
            rnd_in = far_path if rshamt == 0 else close_path

            print('close_path', repr(close_path))
            print('far_path', repr(far_path))

            # Round
            # round in: 1.xxxx_xxx|R_ssss_...._xxxx
            #                  < align shifter length >
            print('rnd_in: ', repr(rnd_in))
            round_bitpos = 1 + fp32_config.mantissa_bits+1
            rnd = rnd_in[align_bit-round_bitpos]
            sticky = rnd_in[align_bit-round_bitpos-1:0].reduceor()
            round = (rnd & sticky) | (rnd_in[align_bit-round_bitpos+1] & rnd & ~sticky)
            print('round', repr(round))

            round_up = round.concat(bit(align_bit - round_bitpos, '0'))
            normed = ubit(align_bit+1, (rnd_in + round_up).bin)

            print('normed', repr(normed))

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
            print('postnorm flag', postnorm_flag)

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

            print('t_exp',t_exp)
        else:
            out_sign = out_sign
            out_exp = out_exp
            out_frac = out_frac

        print(out_sign, out_exp, out_frac)
        return out_sign, out_exp, out_frac