from string import Template
from ConvOpt_KP import KP_Opt_Templates
import math
import pickle
import numpy as np


KP_LUT_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE * CONV_${No}_Kp;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
    ''')

KP_LUT_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
KP_Array<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
         CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
         CONV_${No}_Np, CONV_${No}_M_BIT, CONV_${No}_SIMD_BIT, ${kc_counter_bw}, ${kich_counter_bw}, ${och_offset_bw}>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')

class KP_LUT_Opt_Templates(KP_Opt_Templates):
	def gen_conv_para(self):
        if self.conv.n == 0:
            in_assign = 'conv0_in'
        else:
            in_assign = f'conv_{self.conv.n-1}_layer_out'
            
        return KP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                  SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd))))

    def gen_conv_array(self):
        INFOLD = self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)
        OUTPENUM = self.conv.och // (self.conv.pe * self.conv.kp)

        kc_counter_bw = self.ceil_width(self.conv.k)
        kich_counter_bw = self.ceil_width(self.conv.k * INFOLD)
        och_offset_bw = self.ceil_width(OUTPENUM * self.conv.k * INFOLD)

        return KP_LUT_array.substitute(No=str(self.conv.n), kc_counter_bw=str(kc_counter_bw), kich_counter_bw=str(kich_counter_bw), och_offset_bw=str(och_offset_bw))