from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math
import pickle
import numpy as np


FP_para = Template('''//--------------------Conv ${No}: Parameters--------------------
stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv_${No}_in = ${in_assign_last};
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
const unsigned CONV_${No}_OCH_PF = CONV_${No}_PE;
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_ROW_LEN;
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_OCH_PF) * CONV_${No}_IN_W * (CONV_${No}_OCH_PF / CONV_${No}_ACTP);
const bool CONV_${No}_Sep_Flag = ${Sep_Flag};
    ''')


FP_array = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_OCH_PF * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
FP_Array_sep<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
             CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD * CONV_${No}_KPF, CONV_${No}_PE, CONV_${No}_Kp,
             CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
             CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT, CONV_${No}_Sep_Flag>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')


class FP_Sep_Opt_Templates(FP_Opt_Templates):
	################################################ Search ################################################
    def dsp_operations(self):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (1 * 1)
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1
        OUTPENUM = self.conv.och // 1

        dsp_operations = KNUM * INFOLD * ROW_LEN * OUTPENUM * self.conv.irow

        return dsp_operations

    def get_actp(self, simd, pe, kpf):
        KNUM = (self.conv.k - 1) // self.conv.kp + 1
        INFOLD = self.conv.k * self.conv.ich // (simd * kpf)
        OUT_PF = self.conv.np * pe
        min_actp = OUT_PF // (KNUM * INFOLD)

        valid_flag = False
        best_actp = OUT_PF

        actp_p_max = math.floor(math.log2(pe))
        for actp_p in range(0, actp_p_max + 1):
            actp = 2 ** actp_p
            if actp >= min_actp:
                valid_flag = True
                best_actp = actp
                break

        return valid_flag, best_actp

    # def check_constraints(self, inpe, simd, pe, kpf):
    #     flag = self.reshape_buffer_constraints(inpe=inpe, simd=simd, pe=pe, kpf=kpf)
    #     flag = flag and (self.conv.och % (pe * self.conv.kp) == 0)

    #     return flag

    # def get_feature(self, simd, pe, actp, kpf):
    #     features_list = [simd, pe, actp, kpf]
    #     for idx, (k, v) in enumerate(name_mapping_KP.items()):
    #         if idx < 4:
    #             continue
    #         features_list.append(getattr(self.conv, k))

    #     features_list.append(simd * pe * kpf + actp)
    #     float_features = list(map(float, features_list))

    #     return float_features

    # def Load_Model(self):
    #     self.pred_models = {}
    #     targets = ['wns', 'dsp', 'lut', 'bram', 'II']
    #     for tar in targets:
    #         predictor_file = f'E:/Projects/DeepBurning_MixQ/MixQ_Gen_Accel/Generator/operators/ConvOpt_KP/predictors/BRR_{tar}.pkl'
    #         with open(predictor_file, 'rb') as file:
    #             self.pred_models[tar] = pickle.load(file)


    ################################################ HLS Template ################################################
    def gen_conv_para(self):
        if self.conv.n == 0:
                in_assign = 'conv0_in'
        else:
                in_assign = f'conv_{self.conv.n-1}_layer_out'

        return FP_para.substitute(No=str(self.conv.n), in_assign_last=in_assign, EX_M_BIT=str(math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))),
                                  SIMD_BIT=str(math.ceil(math.log2(self.conv.kpf * self.conv.simd * min(self.conv.kp, self.conv.np)))), CASCADE=str(self.find_CASCADE()),
                                  Sep_Flag=conv.Sep_Flag)

    def gen_conv_array(self):
        return FP_array.substitute(No=str(self.conv.n))

