from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math


KP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
const bool CONV_${No}_PatternFlag = ${PatternFlag};
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE * CONV_${No}_Kp;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
    ''')

KP_array_cascade = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
KP_Array_Cascade<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
                 CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
                 CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
                 CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT, CONV_${No}_PatternFlag>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')


class KP_Opt_Templates(FP_Opt_Templates):

    ################################################ Processing ################################################
    def weight_reorder(self):
        if self.conv.kpf == 1:
            w = self.conv.w    # [och, ich, kr, kc]
            assert self.conv.och%(self.conv.pe*self.conv.kp) == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}, kp {self.conv.kp}"
            assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

            w = w.reshape(self.conv.och//(self.conv.kp*self.conv.pe), self.conv.pe, self.conv.kp, self.conv.ich//self.conv.simd, self.conv.simd, self.conv.k, self.conv.k)   # [och/(kp*pe), pe, kp, ich/simd, simd, kr, kc]
            w = w.transpose(1, 0, 5, 3, 6, 4, 2)            # [pe, och/(kp*pe), kr, ich/simd, kc, simd, kp]
            w = w[:, :, :, :, ::-1, :, :]
            w = w.reshape(self.conv.pe, -1, self.conv.simd*self.conv.kp)   # [pe, och/(kp*pe) * kr * ich/simd * kc, simd * kp]
            self.conv.w = w

            return f"const ap_uint<{self.conv.wbit * self.conv.kp * self.conv.simd}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="
        else:
            w = self.conv.w    # [och, ich, kr, kc]
            assert self.conv.och%(self.conv.pe*self.conv.kp) == 0, f"conv_{self.conv.n}, och {self.conv.och}, pe {self.conv.pe}, kp {self.conv.kp}"
            assert self.conv.ich%self.conv.simd == 0, f"conv_{self.conv.n}, ich {self.conv.ich}, k {self.conv.k}, simd {self.conv.simd}"

            w = w.reshape(self.conv.och//(self.conv.kp*self.conv.pe), self.conv.pe, self.conv.kp, self.conv.ich//self.conv.simd, self.conv.simd, self.conv.k, self.conv.k)   # [och/(kp*pe), pe, kp, ich/simd, simd, kr, kc]
            w = w.transpose(1, 0, 3, 6, 5, 4, 2)            # [pe, och/(kp*pe), ich/simd, kc, kr, simd, kp]
            w = w[:, :, :, ::-1, :, :, :]
            w = w.reshape(self.conv.pe, -1, self.conv.k*self.conv.simd*self.conv.kp)   # [pe, och/(kp*pe) * ich/simd * kc, kr * simd * kp]
            self.conv.w = w

            return f"const ap_uint<{self.conv.wbit * self.conv.kp * self.conv.simd * self.conv.k}> conv_{self.conv.n}_w[{self.conv.pe}][{self.conv.w.shape[1]}]="

    ################################################ HLS Template ################################################
    def find_CASCADE(self):
        cascade = 1
        upper = min((2**self.conv.gb), self.conv.simd*self.conv.kpf)
        for factor in range(1, int(upper) + 1):
            if (self.conv.simd*self.conv.kpf)%factor == 0:
                cascade = factor
        return cascade

    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'
        if self.conv.pack_flag:
            PatternFlag = 'true'
        else:
            PatternFlag = 'false'
        return KP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                  SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd))),
                                  CASCADE=str(self.find_CASCADE()), PatternFlag=PatternFlag)

    def gen_conv_array(self):
        return KP_array_cascade.substitute(No=str(self.conv.n))