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

        pack_flag = bool(pack_flag)
        Kp = Dp if pack_flag else Ep
        Np = Ep if pack_flag else Dp

        return pack_flag, Kp, Np, gb, T_mul

    def Packing_Exploration(self, K=3, overlap=0, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True):
        DSP_Config_Lookup = {}
        for wb in range(wbmin, wbmax + 1):
            for ab in range(abmin, abmax + 1):
                DSP_Config_Dic = {'Packing_Type': 'Default', 'T_mul': 1, 'gb': self.PortA + self.PortB - wb - ab}

                if Filter_Packing_EN:
                    for wsep in [1, 2]:
                        for asep in [1, 2]:
                            if wsep == 2 and asep == 2:
                                continue
                            wb_sep = math.ceil(wb / wsep)
                            ab_sep = math.ceil(ab / asep)

                            Kp, Np, gb, T_mul = self.Filter_Packing(K, wb_sep, ab_sep, overlap)
                            T_mul /= (wsep * asep)
                            C1 = T_mul > DSP_Config_Dic['T_mul']
                            C2 = (T_mul == DSP_Config_Dic['T_mul']) and (gb > DSP_Config_Dic['gb'])

                            if C1 or C2:
                                if wsep == 1 and asep == 1:
                                    DSP_Config_Dic = {'Packing_Type': 'Filter_Packing', 'kp': Kp, 'np': Np, 'T_mul': T_mul, 'gb': gb}
                                else:
                                    Sep_Flag = True if asep == 2 else False
                                    DSP_Config_Dic = {'Packing_Type': 'Filter_Packing_Sep', 'Sep_Flag': Sep_Flag, 'wsep': wsep, 'asep': asep, 'kp': Kp, 'np': Np, 'T_mul': T_mul, 'gb': gb}
                                

                if Kernel_Packing_EN:
                    for wsep in [1, 2]:
                        for asep in [1, 2]:
                            if wsep == 2 and asep == 2:
                                continue
                            wb_sep = math.ceil(wb / wsep)
                            ab_sep = math.ceil(ab / asep)

                            pack_flag, Kp, Np, gb, T_mul = self.Kernel_Packing(wb_sep, ab_sep, overlap)
                            T_mul /= (wsep * asep)
                            C1 = T_mul > DSP_Config_Dic['T_mul']
                            C2 = (T_mul == DSP_Config_Dic['T_mul']) and (gb > DSP_Config_Dic['gb'])

                            if C1 or C2:
                                if wsep == 1 and asep == 1:
                                    DSP_Config_Dic = {'Packing_Type': 'Kernel_Packing', 'Pack_Flag': pack_flag, 'kp': Kp, 'np': Np, 'T_mul': T_mul, 'gb': gb}
                                else:
                                    Sep_Flag = True if asep == 2 else False
                                    DSP_Config_Dic = {'Packing_Type': 'Kernel_Packing_Sep', 'Sep_Flag': Sep_Flag, 'Pack_Flag': pack_flag, 'wsep': wsep, 'asep': asep, 'kp': Kp, 'np': Np, 'T_mul': T_mul, 'gb': gb}

                precision_str = 'w'+str(int(wb))+'a'+str(int(ab))
                DSP_Config_Lookup[precision_str] = DSP_Config_Dic

        return DSP_Config_Lookup
        

    def Save_Lookup_Table(self, DSP_Config_Lookup, K=3, overlap=0):
        save_data = {
            "K" : K,
            "overlap" : overlap,
            "Lookup" : DSP_Config_Lookup
        }

        with open(f'./DSP_Lookup/Lookup_K{K}.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
        return



if __name__ == '__main__':
    DSP_Explorer = DSP_Config_Search(27, 18, 8)
    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=3, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True)
    DSP_Explorer.Save_Lookup_Table(DSP_Config_Lookup, K=3, overlap=1)