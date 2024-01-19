from ..utils.commonimport import *

class FloatVectorThreshold:
    """
        // mod 3bit
    // case        (f)mod   v<lo, lo<v<hi, v>hi
    // vclip        0 000   0       v       0
    // vsat         0 001   lo      0       hi
    // vclip_bin    0 010   0       1       0
    // vsat_bin     0 011   1       0       1
    // vclip_sat    0 100   lo      v       hi
    // vclip_comp   0 101   v       0       v
     
    // vclip        1 000   0       v       0
    // vsat         1 001   lo      0       hi
    // vclip_bin    1 010   0       1.0     0
    // vsat_bin     1 011   1.0     0       1.0
    // vclip_sat    1 100   lo      v       hi
    // vclip_comp   1 101   v       0       v    
    """
    def __init__(self, v: List[bit], lo: bit, hi: bit, mod: int):
        self.vec = v
        self.lo = lo
        self.hi = hi
        self.mod = mod

    def execute(self) -> List[bit]:
        vec = self.vec
        lo = self.lo
        hi = self.hi
        mod = bit(4, bin(self.mod))
        tans = [bit(32, '0') for i in range(VSIZE)]
        _in = [bit(32, '0') for i in range(VSIZE)]
        _in = vec
        temp = bit(32, '0')
        tin = bit(32, '0')
        for i in range(len(_in)):
            tin = _in[i]
            if mod[3] == bit(1, '1'):
                if (_in[i][31]) == bit(1, '1'):
                    tin[30:0] = ~_in[i][30:0]
            if tin < lo:
                if (mod[2:0] == bit(3, '0')) | (mod[2:0] == bit(3, '2')):
                    temp = bit(32, '0')
                elif (mod[2:0] == bit(3, '1')) | (mod[2:0] == bit(3, '4')):
                    temp = lo
                elif mod[2:0] == bit(3, '3'):
                    temp = bit.from_hex(32, '0x3f80_0000') if mod[3] == bit(1, '1') else bit(32, '1')
                else:
                    temp = tin
            elif tin > hi:
                if (mod[2:0] == bit(3, '0')) | (mod[2:0] == bit(3, '2')):
                    temp = bit(32, '0')
                elif (mod[2:0] == bit(3, '1')) | (mod[2:0] == bit(3, '4')):
                    temp = hi
                elif mod[2:0] == bit(3, '3'):
                    temp = bit.from_hex(32, '0x3f80_0000') if mod[3] == bit(1, '1') else bit(32, '1')
                else:
                    temp = tin
            else:
                if (mod[2:0] == bit(3, '1')) | (mod[2:0] == bit(3, '3')) | (mod[2:0] == bit(3, '5')):
                    temp = bit(32, '0')
                elif mod[2:0] == bit(3, '2'):
                    temp = bit.from_hex(32, '0x3f80_0000') if mod[3] == bit(1, '1') else bit(32, '1')
                else:
                    temp = tin
            tans[i] = temp
        return tans
