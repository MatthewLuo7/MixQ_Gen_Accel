import os
import json
import numpy as np
import math
from sklearn.model_selection import train_test_split

name_mapping = {
    'simd': 'SIMD',
    'pe': 'PE',
    'actp': 'ACTP',
    'kpf': 'KPF',
    'icol': 'IN_W',
    'irow': 'IN_H',
    'ich': 'IN_CH',
    'och': 'OUT_CH',
    'abit': 'IN_BIT',
    'wbit': 'W_BIT',
    'obit': 'OUT_BIT',
    'kp': 'Kp',
    'np': 'Np',
    'gb': 'GUARD_BIT',
    'incbit': 'INC_BIT',
    'biasbit': 'BIAS_BIT',
    'max_pool': 'MAX_POOL',
    'pack_flag': 'PACK_FLAG'
    }

def MAE(y_pred, y):
    abs_diff = np.abs(y_pred.reshape(-1) - y.reshape(-1))
    avg_diff = np.average(abs_diff)

    return avg_diff

def get_features(opt_config):
    features_list = []
    for k, v in name_mapping.items():
        features_list.append(opt_config[v])

    features_list.append(opt_config['SIMD'] * opt_config['PE'] * opt_config['KPF'] + opt_config['ACTP'])

    float_features = list(map(float, features_list))

    return float_features


def load_dataset(dataset_path, features, R_DSP, R_LUT, R_BRM, T_WNS):
    with open(os.path.normpath(dataset_path), 'r') as file:
        opt_dicts = json.load(file)
    for (opt_name, opt_val) in opt_dicts.items():
        if int(opt_val['Syn_Res']['timing']['PipelineII']) == 1:
            features.append(get_features(opt_val['Config']))
            R_DSP.append([opt_val['Syn_Res']['resources']['DSP']])
            R_LUT.append([opt_val['Syn_Res']['resources']['LUT']])
            R_BRM.append([opt_val['Syn_Res']['resources']['BRAM']])
            T_WNS.append([opt_val['Syn_Res']['timing']['AchievedClockPeriod']])

class HLS_Dataloader:
    def __init__(self, test_size):
        # import dataset
        features = []
        R_DSP = []
        R_LUT = []
        R_BRM = []
        T_WNS = []

        load_dataset('./dataset/dataset.json', features, R_DSP, R_LUT, R_BRM, T_WNS)
        self.samp_num = len(features)
        
        X = np.array(features)
        y_wns = np.array(T_WNS)
        y_dsp = np.array(R_DSP)
        y_lut = np.array(R_LUT)
        y_brm = np.array(R_BRM)
        Y = np.concatenate((y_wns, y_dsp, y_lut, y_brm), axis=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    def get_datasets(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test