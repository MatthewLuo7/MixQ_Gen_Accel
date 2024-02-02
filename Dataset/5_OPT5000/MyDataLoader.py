import os
import json
import numpy as np
import math
from sklearn.model_selection import train_test_split

# name_mapping = {
#     'simd': 'SIMD',
#     'pe': 'PE',
#     'actp': 'ACTP',
#     'kpf': 'KPF',
#     'icol': 'IN_W',
#     'irow': 'IN_H',
#     'ich': 'IN_CH',
#     'och': 'OUT_CH',
#     'abit': 'IN_BIT',
#     'wbit': 'W_BIT',
#     'obit': 'OUT_BIT',
#     'kp': 'Kp',
#     'np': 'Np',
#     'gb': 'GUARD_BIT',
#     'incbit': 'INC_BIT',
#     'biasbit': 'BIAS_BIT',
#     'max_pool': 'MAX_POOL'
#     }

attr_names = ['k', 'icol', 'irow', 'ich', 'och', 'max_pool', 'abit', 'wbit', 'obit', 'kp',
             'np', 'gb', 'w_sep', 'a_sep', 'simd', 'pe', 'actp', 'kpf', 'lshift', 'incbit',
             'biasbit', 'pack_flag']   # removed in_pe

def MAE(y_pred, y):
    abs_diff = np.abs(y_pred.reshape(-1) - y.reshape(-1))
    avg_diff = np.average(abs_diff)

    return avg_diff

def get_features(opt_config):
    features_list = []
    for attr in attr_names:
        if attr in opt_config.keys():
            features_list.append(opt_config[attr])

    features_list.append(opt_config['w_sep'] * opt_config['a_sep'] * opt_config['simd'] * opt_config['pe'] * opt_config['kpf'] + opt_config['actp'])

    float_features = list(map(float, features_list))

    return float_features


def load_dataset(dataset_path, Packing, LUT, DW, Include_II_violation=False):
    features = []
    R_DSP = []
    R_LUT = []
    R_BRM = []
    T_WNS = []
    II_flag = []

    with open(os.path.normpath(dataset_path), 'r') as file:
        opt_dicts = json.load(file)
    for (opt_name, opt_val) in opt_dicts.items():
        # if (opt_val['Type']['Packing'] == Packing) and (opt_val['Type']['LUT'] == LUT) and (opt_val['Type']['DW'] == DW) and (opt_val['Type']['Latency'] == 4.0):
        #     if (opt_val['Syn_Res'] != None) and (Include_II_violation or int(opt_val['Syn_Res']['timing']['PipelineII']) == 1) and (opt_val['Config']['k'] == 3):
        if (opt_val['Type']['Packing'] == Packing) and (opt_val['Type']['LUT'] == LUT) and (opt_val['Type']['DW'] == DW):
            if (opt_val['Syn_Res'] != None) and (Include_II_violation or int(opt_val['Syn_Res']['timing']['PipelineII']) == 1):
                features.append(get_features(opt_val['Config']))
                R_DSP.append([opt_val['Syn_Res']['resources']['DSP']])
                R_LUT.append([opt_val['Syn_Res']['resources']['LUT']])
                R_BRM.append([opt_val['Syn_Res']['resources']['BRAM']])
                T_WNS.append([opt_val['Syn_Res']['timing']['AchievedClockPeriod']])
                II_flag.append([int(opt_val['Syn_Res']['timing']['PipelineII']) == 1])

                # # debug
                # opt_config = opt_val['Config']
                # est = opt_config['w_sep'] * opt_config['a_sep'] * opt_config['simd'] * opt_config['pe'] * opt_config['kpf'] + opt_config['actp']
                # if int(opt_val['Syn_Res']['resources']['DSP']) == 73 and est == 37:
                #     print(opt_name)

    return features, R_DSP, R_LUT, R_BRM, T_WNS, II_flag

class HLS_Dataloader:
    def __init__(self, dataset_path, test_size, Packing, LUT, DW, Include_II_violation=False):
        # import dataset
        features, R_DSP, R_LUT, R_BRM, T_WNS, II_flag = load_dataset(dataset_path, Packing, LUT, DW)
        self.samp_num = len(features)
        
        X = np.array(features)
        y_dsp = np.array(R_DSP)
        y_lut = np.array(R_LUT)
        y_brm = np.array(R_BRM)
        y_wns = np.array(T_WNS)
        y_IIf = np.array(II_flag)
        Y = np.concatenate((y_dsp, y_lut, y_brm, y_wns, y_IIf), axis=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    def get_datasets(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test