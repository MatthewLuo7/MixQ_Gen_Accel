from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math
import pickle
import numpy as np

FP_LUT_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
    ''')

FP_LUT_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
FP_Array_lut<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
         	 CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
         	 CONV_${No}_Np, CONV_${No}_M_BIT, CONV_${No}_SIMD_BIT, ${k_counter_bw}, ${infold_counter_bw}, ${res_offset_bw}, ${add_offset_bw}>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')


class FP_LUT_Opt_Templates(FP_Opt_Templates):

################################################ HLS Template ################################################
    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'

        return FP_LUT_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                   SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd * min(self.conv.kp, self.conv.np)))))

    def gen_conv_array(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (self.conv.simd * self.conv.kpf)
        OUTPENUM = self.conv.och // self.conv.pe

        k_counter_bw = self.ceil_width(KNUM)
        infold_counter_bw = self.ceil_width(INFOLD, min_BW=2)
        res_offset_bw = self.ceil_width(self.conv.kp * KNUM)
        add_offset_bw = self.ceil_width(OUTPENUM * INFOLD)

        return FP_LUT_array.substitute(No=str(self.conv.n), k_counter_bw=str(k_counter_bw), infold_counter_bw=str(infold_counter_bw),
                                  	   res_offset_bw=str(res_offset_bw), add_offset_bw=str(add_offset_bw))