from string import Template
from ConvOpt_FP import FP_Opt_Templates
import math
import pickle
import numpy as np


class FP_Sep_Opt_Templates(FP_Opt_Templates):
	################################################ Search ################################################
    def dsp_operations(self):
        K = self.conv.k
        INFOLD = self.conv.k * self.conv.ich // (1 * 1)
        ROW_LEN = (self.conv.icol + self.conv.k - 2) // self.conv.np + 1
        OUTPENUM = self.conv.och // (1 * self.conv.kp);

        dsp_operations = K * INFOLD * ROW_LEN * OUTPENUM * self.conv.irow

        return dsp_operations

    def get_actp(self, simd, pe, kpf):
        K = self.conv.k
        INFOLD = self.conv.k * self.conv.ich // (simd * kpf)
        OUT_PF = self.conv.np * pe * self.conv.kp
        min_actp = OUT_PF // (K * INFOLD)

        valid_flag = False
        best_actp = OUT_PF

        actp_p_max = math.floor(math.log2(pe * self.conv.kp))
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

