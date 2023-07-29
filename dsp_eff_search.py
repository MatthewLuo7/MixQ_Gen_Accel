import math
import numpy as np
import json

class DSP_Config_Search:
    def __init__(self, PortA, PortB, wb_thres):
        self.PortA = PortA
        self.PortB = PortB
        self.wb_thres = wb_thres

    def extra_wb(self, wb):
        # Signed range: [-2^(wb-1), 2^(wb-1) - 1]. For this asymmetric quantization, packing -2^(wb-1) can lead to overflow, one bit must left to prevent this.
        # Usually, symetric quantization, i.e. [-2^(wb-1) + 1, 2^(wb-1) - 1], is used, so there is no need to preserve the extra one bit.
        # However, to maintain accuracy for ultra-low quantization (wb <= wb_thres), we still support asymmetric quantization.

        if wb <= self.wb_thres:
            return 1
        else:
            return 0


    def Filter_Packing(self, K=3, wb=4, ab=4, overlap=0):
        # initialization
        Kp = 1             # the number of packed weights
        Np = 1             # the number of packed activations
        gb = -1            # the guard bits
        T_mul = 1          # multiplication throughput

        N_Switch = 1 if self.PortA == self.PortB else 2    # if PortA != PortB, switch the two ports for thorough exploration
        for i in range(N_Switch):
            if i == 0:
                PA = self.PortA
                PW = self.PortB
            else:
                PA = self.PortB
                PW = self.PortA

            PA = PA - 1      # leave one bit for the unsigned activation port to emulate signed multiplication
            Npmax = math.floor((PA - ab) / (wb + ab - overlap)) + 1                                             # upper-bound of Np
            Kpmax = min(K, math.floor((PW - wb - self.extra_wb(wb)) / (wb + ab - overlap)) + 1)       # upper-bound of Mp

            # traverse all possible Np and Kp configurations
            for Npi in range(1, Npmax + 1):
                for Kpi in range(1, Kpmax + 1):
                    gbi_min = math.ceil(math.log2(min(Kpi, Npi))) - overlap   # the minimum guard bits requirement   

                    # calculating the upper-bound of guard bits                
                    if Kpi > 1:                                               # the maximum guard bits for weight packing
                        gW = ((PW - wb - self.extra_wb(wb)) // (Kpi - 1)) - wb - ab
                    else:
                        gW = PW + PA - wb - ab                 
                    if Npi > 1:                                               # the maximum guard bits for activation packing
                        gA = ((PA - ab) // (Npi - 1)) - wb - ab
                    else:
                        gA = PW + PA - wb - ab
                    gbi_max = min(gW, gA)                                      # the maximum guard bits considering both activation and weight packing

                    # for the legal combinations
                    if gbi_min <= gbi_max:
                        gbi = gbi_max
                        GKi = math.ceil(K / Kpi)

                        C1 = ((K * Npi / GKi) > T_mul)                                                            # higher multiplication throughput
                        C2 = (((K * Npi / GKi) == T_mul) and (                                                    # same multiplication throughput but support more accumulations
                                math.floor((2 ** gbi) / min(Kpi, Npi)) > math.floor((2 ** gb) / min(Kpi, Npi))))
                        if C1 or C2:
                            Kp = Kpi
                            Np = Npi
                            gb = gbi
                            T_mul = K * Npi / GKi

        return Kp, Np, gb, T_mul


    def Kernel_Packing(self, wb=4, ab=4, overlap=0):
        # assume there are two ports E and D, and PortE >= PortD
        PortE = max(self.PortA, self.PortB)
        PortD = min(self.PortA, self.PortB)
        
        pack_flag = 0     # pack weight(s) on Port E when pack_flag == 0, otherwise pack activation(s) on Port E
        Ep = 1            # the number of weights (or activations) packed on Port E
        Dp = 1            # the number of activations (or weights) packed on Port D
        gb = -1           # the number of guard bits
        T_mul = 1         # multiplication throughput

        N_Switch = 1 if PortE == PortD else 2    # if PortE != PortD, switch the two ports for thorough exploration
        for pflag in range(N_Switch):
            if pflag == 0:    # pack weight(s) on Port E and pack activation(s) on Port D
                PortE = max(self.PortA, self.PortB)
                PortD = min(self.PortA, self.PortB) - 1  # leave one bit to emulate signed multiplication
                eb_bias = wb + self.extra_wb(wb)
                db_bias = ab
                Epmax = (PortE - eb_bias) // (wb + ab - overlap) + 1
                Dpmax = (PortD - db_bias) // (wb + ab - overlap) + 1
                
            else:    # pack activation(s) on Port E and pack weight(s) on Port D
                PortE = max(self.PortA, self.PortB) - 1  # leave one bit to emulate signed multiplication
                PortD = min(self.PortA, self.PortB)
                eb_bias = ab
                db_bias = wb + self.extra_wb(wb)
                Epmax = (PortE - eb_bias) // (wb + ab - overlap) + 1
                Dpmax = (PortD - db_bias) // (wb + ab - overlap) + 1

            for Epi in range(1, Epmax + 1):
                for Dpi in range(1, Dpmax + 1):
                    gbi_min = -overlap   # the minimum guard bits requirement 

                    # calculating the upper-bound of guard bits
                    if Dpi > 1:
                        gD = ((PortD - db_bias) // (Dpi - 1)) - wb - ab
                    else:
                        gD = PortE + PortD - wb - ab
                    if Epi > 1:
                        gE = ((PortE - eb_bias) // (Dpi * (Epi - 1))) - wb - ab
                    else:
                        gE = PortE + PortD - wb - ab
                    gbi_max = min(gE, gD)

                    # for the legal combinations
                    if gbi_min <= gbi_max:
                        gbi = gbi_max

                        C1 = (Dpi * Epi > T_mul)                                          # higher multiplication throughput
                        C2 = ((Dpi * Epi == T_mul) and (gbi > gb))                        # same multiplication throughput but support more accumulations
                        if C1 or C2:
                            pack_flag = pflag
                            Ep = Epi
                            Dp = Dpi
                            gb = gbi
                            T_mul = Dpi * Epi

        return pack_flag, Ep, Dp, gb, T_mul

    def Packing_Exploration(self, K=3, overlap=0, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True):
        DSP_Config_Lookup = {}
        for wb in range(wbmin, wbmax + 1):
            for ab in range(abmin, abmax + 1):
                DSP_Config_Dic = {'Packing_Type': 'Default', 'T_mul': 1, 'gb': self.PortA + self.PortB - wb - ab}

                if Filter_Packing_EN:
                    for wsep in [1, 2]:
                        for asep in [1, 2]:
                            wb_sep = math.ceil(wb / wsep)
                            ab_sep = math.ceil(ab / asep)

                            Kp, Np, gb, T_mul = self.Filter_Packing(K, wb_sep, ab_sep, overlap)
                            T_mul /= (wsep * asep)
                            C1 = T_mul > DSP_Config_Dic['T_mul']
                            C2 = (T_mul == DSP_Config_Dic['T_mul']) and (gb > DSP_Config_Dic['gb'])

                            if C1 or C2:
                                DSP_Config_Dic = {'Packing_Type': 'Filter_Packing', 'wsep': wsep, 'asep': asep, 'Kp': Kp, 'Np': Np, 'T_mul': T_mul, 'gb': gb}

                if Kernel_Packing_EN:
                    for wsep in [1, 2]:
                        for asep in [1, 2]:
                            wb_sep = math.ceil(wb / wsep)
                            ab_sep = math.ceil(ab / asep)

                            pack_flag, Ep, Dp, gb, T_mul = self.Kernel_Packing(wb_sep, ab_sep, overlap)
                            T_mul /= (wsep * asep)
                            C1 = T_mul > DSP_Config_Dic['T_mul']
                            C2 = (T_mul == DSP_Config_Dic['T_mul']) and (gb > DSP_Config_Dic['gb'])
                            if C1 or C2:
                                DSP_Config_Dic = {'Packing_Type': 'Kernel_Packing', 'Pack_Flag': pack_flag, 'wsep': wsep, 'asep': asep, 'Ep': Ep, 'Dp': Dp, 'T_mul': T_mul, 'gb': gb}

                precision_str = 'w'+str(int(wb))+'a'+str(int(ab))
                DSP_Config_Lookup[precision_str] = DSP_Config_Dic

        return DSP_Config_Lookup
        

    def Save_Lookup_Table(self, DSP_Config_Lookup, K=3, overlap=0):
        save_data = {
            "K" : K,
            "overlap" : overlap,
            "Lookup" : DSP_Config_Lookup
        }

        with open('./Lookup.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
        return


#------------------------------------------------------------------------------------------------------------------------------------------

def ref_kernel_search(M, overlap):
    Mp_1 = np.zeros((7, 7))
    Np_1 = np.zeros((7, 7))
    gb_1 = np.zeros((7, 7))
    Eff_1 = np.zeros((7, 7))

    Mp_2 = np.zeros((7, 7))
    Np_2 = np.zeros((7, 7))
    gb_2 = np.zeros((7, 7))
    Eff_2 = np.zeros((7, 7))

    DSP_Explorer = DSP_Config_Search(27, 18, 3)
    for wb in range(2, 9):
        for ab in range(2, 9):
            Mp_1[wb - 2, ab - 2], Np_1[wb - 2, ab - 2], gb_1[wb - 2, ab - 2], Eff_1[wb - 2, ab - 2] = max_DSP_mul_search(M, wb, ab, 27, 18, overlap)               
            Mp_2[wb - 2, ab - 2], Np_2[wb - 2, ab - 2], gb_2[wb - 2, ab - 2], Eff_2[wb - 2, ab - 2] = DSP_Explorer.Filter_Packing(M, wb, ab, overlap)

    return Mp_1, Np_1, gb_1, Eff_1, Mp_2, Np_2, gb_2, Eff_2


def ref_HL_sep_search_eff_gb(GM, Mp, Np, gb, Eff):
    GM_opt = np.copy(GM)
    Mp_opt = np.copy(Mp)
    Np_opt = np.copy(Np)
    gb_opt = np.copy(gb)
    Eff_opt = np.copy(Eff)
    wb_s = np.ones((7, 7))
    ab_s = np.ones((7, 7))
    for Wb in range(2, 9):
        for Ab in range(2, 9):
            Eff_t = Eff[Wb - 2, Ab - 2]
            gb_t = gb[Wb - 2, Ab - 2]
            in_acc_t = min(Mp[Wb - 2, Ab - 2], Np[Wb - 2, Ab - 2])
            for wscale in [1, 2]:
                for ascale in [1, 2]:
                    Wbs = math.ceil(Wb / wscale)
                    Abs = math.ceil(Ab / ascale)
                    if (Wbs in range(2, 9)) and (Abs in range(2, 9)):
                        # C1 = ((Eff[Wbs-2,Abs-2] / (wscale * ascale)) > Eff_opt[Wb-2,Ab-2])
                        # C2 = (((Eff[Wbs-2,Abs-2] / (wscale * ascale)) == Eff_opt[Wb-2,Ab-2]) and (gb[Wbs-2,Abs-2] > gb_opt[Wb-2,Ab-2]))
                        C1 = ((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) > Eff_t)
                        # C2 = (((Eff[Wbs-2,Abs-2] / (wscale * ascale)) == Eff_t) and (gb[Wbs-2,Abs-2] > gb_t))
                        C2 = (((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) == Eff_t) and (math.floor(
                            (2 ** gb[Wbs - 2, Abs - 2]) / min(Mp[Wbs - 2, Abs - 2], Np[Wbs - 2, Abs - 2])) > math.floor(
                            (2 ** gb_t) / in_acc_t)))
                        if C1 or C2:
                            GM_opt[Wb - 2, Ab - 2] = GM[Wbs - 2, Abs - 2]
                            Mp_opt[Wb - 2, Ab - 2] = Mp[Wbs - 2, Abs - 2]
                            Np_opt[Wb - 2, Ab - 2] = Np[Wbs - 2, Abs - 2]
                            gb_opt[Wb - 2, Ab - 2] = gb[Wbs - 2, Abs - 2]
                            Eff_opt[Wb - 2, Ab - 2] = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            wb_s[Wb - 2, Ab - 2] = wscale
                            ab_s[Wb - 2, Ab - 2] = ascale
                            Eff_t = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            gb_t = gb[Wbs - 2, Abs - 2]
                            in_acc_t = min(Mp[Wbs - 2, Abs - 2], Np[Wbs - 2, Abs - 2])

    return GM_opt, Mp_opt, Np_opt, gb_opt, Eff_opt, wb_s, ab_s


#------------------------------------------------------------------------------------------------------------------------------------------


def get_acc(gb, Mp, Np):
    temp = np.concatenate((Mp.reshape(7, 7, 1), Np.reshape(7, 7, 1)), axis=2)
    MNmin = np.min(temp, axis=2)
    # return np.floor((2**gb) / MNmin)
    return (gb - np.ceil(np.log2(MNmin)))
    # return gb
    # return


def extraWP(Wb):
    if Wb <= 3:
        a = 1
    else:
        a = 0
    return a


def max_DSP_mul_search(M, wb, ab, PortA, PortB, overlap):
    GM = M
    Mp = 1
    Np = 1
    gb = -1
    Eff = 1
    for i in range(2):
        if i == 0:
            PA = PortA
            PW = PortB
        else:
            PA = PortB
            PW = PortA

        PA = PA - 1   # leave one bit for the activation port to emulate signed multiplication
        Npmax = math.floor((PA - ab) / (wb + ab - overlap)) + 1
        Mpmax = min(M, math.floor((PW - wb - extraWP(wb)) / (wb + ab - overlap)) + 1)
        for Mpi in range(1, Mpmax + 1):
            GMi = math.ceil(M / Mpi)
            for Npi in range(1, Npmax + 1):
                gbimin = math.ceil(math.log2(min(Mpi, Npi))) - overlap
                if Mpi > 1:
                    gW = ((PW - wb - extraWP(wb)) // (Mpi - 1)) - wb - ab
                else:
                    gW = PW - wb - extraWP(wb)
                if Npi > 1:
                    gA = ((PA - ab) // (Npi - 1)) - wb - ab
                else:
                    gA = PA - ab
                gbimax = min(gW, gA)
                if gbimin <= gbimax:
                    gbi = gbimax
                    pbi = wb + ab + gbi
                    C1 = ((M * Npi / GMi) > Eff)
                    C2 = (((M * Npi / GMi) == Eff) and (
                                math.floor((2 ** gbi) / min(Mpi, Npi)) > math.floor((2 ** gb) / min(Mpi, Npi))))
                    if C1 or C2:
                        GM = GMi
                        Mp = Mpi
                        Np = Npi
                        gb = gbi
                        Eff = M * Npi / GMi

    return Mp, Np, gb, Eff


def kernel_search(M, overlap):
    Mp_1 = np.zeros((7, 7))
    Np_1 = np.zeros((7, 7))
    gb_1 = np.zeros((7, 7))
    Eff_1 = np.zeros((7, 7))

    Mp_2 = np.zeros((7, 7))
    Np_2 = np.zeros((7, 7))
    gb_2 = np.zeros((7, 7))
    Eff_2 = np.zeros((7, 7))

    DSP_Explorer = DSP_Config_Search(27, 18, 3)
    for wb in range(2, 9):
        for ab in range(2, 9):
            Mp_1[wb - 2, ab - 2], Np_1[wb - 2, ab - 2], gb_1[wb - 2, ab - 2], Eff_1[wb - 2, ab - 2] = max_DSP_mul_search(M, wb, ab, 27, 18, overlap)               
            Mp_2[wb - 2, ab - 2], Np_2[wb - 2, ab - 2], gb_2[wb - 2, ab - 2], Eff_2[wb - 2, ab - 2] = DSP_Explorer.Filter_Packing(M, wb, ab, overlap)

    return Mp_1, Np_1, gb_1, Eff_1, Mp_2, Np_2, gb_2, Eff_2


def max_DSP_mul_search_1x1(wb, ab, overlap):
    Mp = 1
    Np = 1
    gb = -1
    Eff = 1
    flag = -1
    for i in range(2):
        if i == 0:
            N1 = (18 - wb - extraWP(wb)) // (wb + ab - overlap) + 1
            pb_1 = (18 - wb - extraWP(wb)) // (N1 - 1) if N1 > 1 else 45
            N2 = (27 - 1 - ab) // (N1 * (wb + ab - overlap)) + 1                        # - 1 aims to leave one bit for the activation port to emulate signed multiplication
            pb_2 = (27 - 1 - ab) // ((N2 - 1)*N1) if N2 > 1 else 45
        else:
            N1 = (18 - 1 - ab) // (wb + ab - overlap) + 1
            pb_1 = (18 - 1 - ab) // (N1 - 1) if N1 > 1 else 45
            N2 = (27 - wb - extraWP(wb)) // (N1 * (wb + ab - overlap)) + 1
            pb_2 = (27 - wb - extraWP(wb)) // ((N2 - 1)*N1) if N2 > 1 else 45

        gb_cur = min(pb_1, pb_2) - ab - wb
        if ((N1 * N2) > Eff) or (((N1 * N2) == Eff) and (gb_cur > gb)):
            Eff = N1 * N2
            if i == 0:
                Mp = N1
                Np = N2
            else:
                Mp = N2
                Np = N1
            gb = gb_cur
            flag = i

        if flag == 0:
            WITV_BIT = (wb+ab+gb-overlap) if Mp > 1 else 0
            AITV_BIT = (wb+ab+gb-overlap)*Mp
        else:
            AITV_BIT = (wb+ab+gb-overlap) if Np > 1 else 0
            WITV_BIT = (wb+ab+gb-overlap)*Np

    return flag, WITV_BIT, AITV_BIT, Mp, Np, gb, Eff


def max_DSP_mul_search_1x1_p2(wb, ab, overlap):
    Mp = 1
    Np = 1
    gb = -1
    Eff = 1
    flag = -1
    for i in range(2):
        if i == 0:
            N1 = 2**math.floor(math.log2((18 - wb - extraWP(wb)) // (wb + ab - overlap) + 1))
            pb_1 = (18 - wb - extraWP(wb)) // (N1 - 1) if N1 > 1 else 45
            N2 = (27 - 1 - ab) // (N1 * (wb + ab - overlap)) + 1
            pb_2 = (27 - 1 - ab) // ((N2 - 1) * N1) if N2 > 1 else 45
        else:
            N1 = (18 - 1 - ab) // (wb + ab - overlap) + 1
            pb_1 = (18 - 1 - ab) // (N1 - 1) if N1 > 1 else 45
            N2 = 2**math.floor(math.log2((27 - wb - extraWP(wb)) // (N1 * (wb + ab - overlap)) + 1))
            pb_2 = (27 - wb - extraWP(wb)) // ((N2 - 1) * N1) if N2 > 1 else 45

        gb_cur = min(pb_1, pb_2) - ab - wb
        if ((N1 * N2) > Eff) or (((N1 * N2) == Eff) and (gb_cur > gb)):
            Eff = N1 * N2
            if i == 0:
                Mp = N1
                Np = N2
            else:
                Mp = N2
                Np = N1
            gb = gb_cur
            flag = i

    return flag, Mp, Np, gb, Eff


# def kernel_search_1x1(overlap):
#     eff = np.zeros((7, 7))
#     gb = np.zeros((7, 7))
#     Mp = np.zeros((7, 7))
#     Np = np.zeros((7, 7))
#     flag = np.zeros((7, 7)) - 1
#     WITV_BIT = np.zeros((7, 7))
#     AITV_BIT = np.zeros((7, 7))
#     for wb in range(2, 9):
#         for ab in range(2, 9):
#             flag[wb - 2, ab - 2], WITV_BIT[wb - 2, ab - 2], AITV_BIT[wb - 2, ab - 2], Mp[wb - 2, ab - 2], Np[wb - 2, ab - 2], gb[wb - 2, ab - 2], eff[
#                 wb - 2, ab - 2] = max_DSP_mul_search_1x1(wb, ab, overlap)

#     return flag, WITV_BIT, AITV_BIT, Mp, Np, gb, eff

def kernel_search_1x1(overlap):
    eff_1 = np.zeros((7, 7))
    gb_1 = np.zeros((7, 7))
    Mp_1 = np.zeros((7, 7))
    Np_1 = np.zeros((7, 7))
    flag_1 = np.zeros((7, 7)) - 1
    WITV_BIT_1 = np.zeros((7, 7))
    AITV_BIT_1 = np.zeros((7, 7))

    eff_2 = np.zeros((7, 7))
    gb_2 = np.zeros((7, 7))
    Mp_2 = np.zeros((7, 7))
    Np_2 = np.zeros((7, 7))
    flag_2 = np.zeros((7, 7)) - 1

    DSP_Explorer = DSP_Config_Search(27, 18, 3)

    for wb in range(2, 9):
        for ab in range(2, 9):
            flag_1[wb - 2, ab - 2], WITV_BIT_1[wb - 2, ab - 2], AITV_BIT_1[wb - 2, ab - 2], Mp_1[wb - 2, ab - 2], Np_1[wb - 2, ab - 2], gb_1[wb - 2, ab - 2], eff_1[
                wb - 2, ab - 2] = max_DSP_mul_search_1x1(wb, ab, overlap)

            flag_2[wb - 2, ab - 2], Mp_2[wb - 2, ab - 2], Np_2[wb - 2, ab - 2], gb_2[wb - 2, ab - 2], eff_2[wb - 2, ab - 2] = DSP_Explorer.Kernel_Packing(wb, ab, overlap)


    return eff_1, eff_2


def HL_sep_search_1x1_gb(Eff, gb):
    gb_opt = np.copy(gb)
    Eff_opt = np.copy(Eff)
    for Wb in range(2, 9):
        for Ab in range(2, 9):
            Eff_t = Eff[Wb - 2, Ab - 2]
            gb_t = gb[Wb - 2, Ab - 2]
            for wscale in [1, 2]:
                for ascale in [1, 2]:
                    Wbs = math.ceil(Wb / wscale)
                    Abs = math.ceil(Ab / ascale)
                    if (Wbs in range(2, 9)) and (Abs in range(2, 9)):
                        C1 = ((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) > Eff_t)
                        C2 = (((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) == Eff_t)) and (gb[Wbs - 2, Abs - 2] > gb_t)
                        if C1 or C2:
                            gb_opt[Wb - 2, Ab - 2] = gb[Wbs - 2, Abs - 2]
                            Eff_opt[Wb - 2, Ab - 2] = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            Eff_t = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            gb_t = gb[Wbs - 2, Abs - 2]

    return gb_opt, Eff_opt


def HL_sep_search(eff):
    max_eff = np.zeros_like(eff)
    for Wb in range(2, 9):
        for Ab in range(2, 9):
            Wbh = math.ceil(Wb / 2)
            Abh = math.ceil(Ab / 2)
            eff1 = 0
            eff2 = 0
            eff3 = 0
            if Wbh >= 2:
                eff1 = eff[Wbh - 2, Ab - 2] / 2
            if Abh >= 2:
                eff2 = eff[Wb - 2, Abh - 2] / 2
            if Wbh >= 2 and Abh >= 2:
                eff3 = eff[Wbh - 2, Abh - 2] / 4
            max_eff[Wb - 2, Ab - 2] = max([eff[Wb - 2, Ab - 2], eff1, eff2, eff3])

    return max_eff


def HL_sep_search_eff_gb(GM, Mp, Np, gb, Eff):
    GM_opt = np.copy(GM)
    Mp_opt = np.copy(Mp)
    Np_opt = np.copy(Np)
    gb_opt = np.copy(gb)
    Eff_opt = np.copy(Eff)
    wb_s = np.ones((7, 7))
    ab_s = np.ones((7, 7))
    for Wb in range(2, 9):
        for Ab in range(2, 9):
            Eff_t = Eff[Wb - 2, Ab - 2]
            gb_t = gb[Wb - 2, Ab - 2]
            in_acc_t = min(Mp[Wb - 2, Ab - 2], Np[Wb - 2, Ab - 2])
            for wscale in [1, 2]:
                for ascale in [1, 2]:
                    Wbs = math.ceil(Wb / wscale)
                    Abs = math.ceil(Ab / ascale)
                    if (Wbs in range(2, 9)) and (Abs in range(2, 9)):
                        # C1 = ((Eff[Wbs-2,Abs-2] / (wscale * ascale)) > Eff_opt[Wb-2,Ab-2])
                        # C2 = (((Eff[Wbs-2,Abs-2] / (wscale * ascale)) == Eff_opt[Wb-2,Ab-2]) and (gb[Wbs-2,Abs-2] > gb_opt[Wb-2,Ab-2]))
                        C1 = ((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) > Eff_t)
                        # C2 = (((Eff[Wbs-2,Abs-2] / (wscale * ascale)) == Eff_t) and (gb[Wbs-2,Abs-2] > gb_t))
                        C2 = (((Eff[Wbs - 2, Abs - 2] / (wscale * ascale)) == Eff_t) and (math.floor(
                            (2 ** gb[Wbs - 2, Abs - 2]) / min(Mp[Wbs - 2, Abs - 2], Np[Wbs - 2, Abs - 2])) > math.floor(
                            (2 ** gb_t) / in_acc_t)))
                        if C1 or C2:
                            GM_opt[Wb - 2, Ab - 2] = GM[Wbs - 2, Abs - 2]
                            Mp_opt[Wb - 2, Ab - 2] = Mp[Wbs - 2, Abs - 2]
                            Np_opt[Wb - 2, Ab - 2] = Np[Wbs - 2, Abs - 2]
                            gb_opt[Wb - 2, Ab - 2] = gb[Wbs - 2, Abs - 2]
                            Eff_opt[Wb - 2, Ab - 2] = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            wb_s[Wb - 2, Ab - 2] = wscale
                            ab_s[Wb - 2, Ab - 2] = ascale
                            Eff_t = Eff[Wbs - 2, Abs - 2] / (wscale * ascale)
                            gb_t = gb[Wbs - 2, Abs - 2]
                            in_acc_t = min(Mp[Wbs - 2, Abs - 2], Np[Wbs - 2, Abs - 2])

    return GM_opt, Mp_opt, Np_opt, gb_opt, Eff_opt, wb_s, ab_s



if __name__ == '__main__':
    DSP_Explorer = DSP_Config_Search(27, 18, 3)
    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=1, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True)
    DSP_Explorer.Save_Lookup_Table(DSP_Config_Lookup, K=3, overlap=0)
    print(DSP_Explorer.Kernel_Packing(7, 5, 1))

    # Mp_1, Np_1, gb_1, Eff_1, Mp_2, Np_2, gb_2, Eff_2 = kernel_search(7, 1)
    # print(Eff_1)
    # print(Eff_2)
    # print(Eff_1 - Eff_2)

    # DSP_Explorer = DSP_Config_Search(27, 18, 3)
    # M = 3
    # overlap = 1
    # print(max_DSP_mul_search(M, 2, 5, 27, 18, overlap))
    # print(DSP_Explorer.Filter_Packing(M, 2, 5, overlap))

    # eff_1, eff_2 = kernel_search_1x1(0)
    # print(eff_1)
    # print(eff_2)
    # print(eff_1 - eff_2)

    # DSP_Explorer = DSP_Config_Search(27, 18, 3)
    # overlap = 1
    # flag_1, _, _, Mp_1, Np_1, gb_1, eff_1 = max_DSP_mul_search_1x1(2, 3, overlap)
    # print(flag_1, Mp_1, Np_1, gb_1, eff_1)
    # print(DSP_Explorer.Kernel_Packing(2, 3, overlap))

    # DSP_Explorer = DSP_Config_Search(27, 18, 3)
    # print(DSP_Explorer.Filter_Packing(K=3, wb=2, ab=2, overlap=1))
    # print(DSP_Explorer.Kernel_Packing(wb=2, ab=2, overlap=1))


# if __name__ == '__main__':
#     GM, Mp, Np, gb, Eff = kernel_search(3, 1)
#     GM_opt, Mp_opt, Np_opt, gb_opt, Eff_opt, wb_s, ab_s = HL_sep_search_eff_gb(GM, Mp, Np, gb, Eff)
#     print(Eff_opt)
#     # flag, WITV_BIT, AITV_BIT, Mp, Np, gb, eff = kernel_search_1x1(1)
#     # print("efficiency:")
#     # print(eff)
#     # print("PEP:")
#     # print(Mp)
#     # print("Np:")
#     # print(Np)
#     # print("WITV_BIT:")
#     # print(WITV_BIT)
#     # print("AITV_BIT:")
#     # print(AITV_BIT)