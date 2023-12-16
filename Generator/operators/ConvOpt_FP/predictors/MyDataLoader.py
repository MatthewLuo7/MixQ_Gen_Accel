import os
import json
import numpy as np
import math

def MSE(y_pred, y):
    y_diff = y_pred.reshape(-1) - y.reshape(-1)
    y_sqr = np.power(y_diff, 2)
    avg_mse = np.average(y_sqr) / 2.0

    return avg_mse

def Abs_Diff(y_pred, y):
    abs_diff = np.abs(y_pred.reshape(-1) - y.reshape(-1))
    avg_diff = np.average(abs_diff)

    return avg_diff

name_mapping = {
    'icol': 'IN_W',
    'irow': 'IN_H',
    'ich': 'IN_CH',
    'och': 'OUT_CH',
    'abit': 'IN_BIT',
    'wbit': 'W_BIT',
    'obit': 'OUT_BIT',
    'in_pe': 'IN_PE',
    'simd': 'SIMD',
    'pe': 'PE',
    'actp': 'ACTP',
    'kp': 'Kp',
    'np': 'Np',
    'gb': 'GUARD_BIT',
    'incbit': 'INC_BIT',
    'biasbit': 'BIAS_BIT',
    'kpf': 'KPF',
    'max_pool': 'MAX_POOL'
    }

def get_features(opt_config):
    features_list = []
    for k, v in name_mapping.items():
        features_list.append(opt_config[v])

    features_list.append(opt_config['SIMD'] * opt_config['PE'] * opt_config['KPF'] + opt_config['ACTP'])

    WROM_BW = opt_config['K'] * opt_config['SIMD'] * opt_config['KPF'] * opt_config['W_BIT']
    WROM_BW_num = math.ceil(WROM_BW / 72)
    WROM_BW_wth = math.floor(WROM_BW / WROM_BW_num)
    WROM_CP = (opt_config['K'] * opt_config['IN_CH'] / (opt_config['SIMD'] * opt_config['KPF'])) * (opt_config['OUT_CH'] / opt_config['PE'])

    S2P_BW = opt_config['IN_PE'] * opt_config['IN_BIT'] * opt_config['Np'] if opt_config['SIMD'] >= opt_config['IN_PE'] else opt_config['SIMD'] * opt_config['IN_BIT'] * opt_config['Np']
    S2P_BW_num = math.ceil(S2P_BW / 36)
    S2P_BW_wth = math.floor(S2P_BW / S2P_BW_num)
    S2P_DIM1 = opt_config['SIMD'] / opt_config['IN_PE'] if opt_config['SIMD'] >= opt_config['IN_PE'] else opt_config['IN_PE'] / opt_config['SIMD']
    K = 3
    S2P_DIM2 = 1 if opt_config['KPF'] == 1 else (K+1)
    ROW_LEN = (opt_config['IN_W'] + K - 2) // opt_config['Np'] + 1
    S2P_CP = ROW_LEN * (opt_config['IN_CH'] / opt_config['SIMD']) if opt_config['SIMD'] >= opt_config['IN_PE'] else ROW_LEN * (opt_config['IN_CH'] / opt_config['IN_PE'])
    S2P_CP_num = math.ceil(S2P_BW_wth * S2P_CP / (18 * 1024))

    features_list.append(WROM_BW_wth * WROM_CP / (18 * 1024) + S2P_BW_num * S2P_DIM1 * S2P_DIM2 * S2P_CP_num + 2*opt_config['ACTP'])

    float_features = list(map(float, features_list))

    return float_features


def load_dataset(dataset_path, features, R_DSP, R_LUT, R_BRM, T_WNS):
    with open(os.path.normpath(dataset_path), 'r') as file:
        opt_dicts = json.load(file)
    for (opt_name, opt_val) in opt_dicts.items():
        if opt_val['Config']['Opt_Type'] == 'FP_Opt' and int(opt_val['Syn_Res']['timing']['PipelineII']) == 1:
            features.append(get_features(opt_val['Config']))
            R_DSP.append([opt_val['Syn_Res']['resources']['DSP']])
            R_LUT.append([opt_val['Syn_Res']['resources']['LUT']])
            R_BRM.append([opt_val['Syn_Res']['resources']['BRAM']])
            T_WNS.append([opt_val['Syn_Res']['timing']['AchievedClockPeriod']])

class HLS_Dataloader:
    def __init__(self, train_ratio, val_ratio):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - self.train_ratio - self.val_ratio

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

        self.X_mean = np.mean(X, axis=0)
        self.Y_mean = np.mean(Y, axis=0)

        self.X_std = np.std(X, axis=0)
        self.Y_std = np.std(Y, axis=0)

        X = (X - self.X_mean) / self.X_std
        Y = (Y - self.Y_mean) / self.Y_std

        X = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)

        self.train_num = round(self.samp_num * self.train_ratio)
        self.val_num = round(self.samp_num * self.val_ratio)
        self.test_num = self.samp_num - self.train_num - self.val_num

        # shuffle dataset
        np.random.seed(314)
        np.random.shuffle(X)
        np.random.seed(314)
        np.random.shuffle(Y)

        self.X_train = X[:self.train_num]
        self.Y_train = Y[:self.train_num]

        self.X_val = X[self.train_num:self.train_num + self.val_num]
        self.Y_val = Y[self.train_num:self.train_num + self.val_num]

        self.X_test = X[self.train_num + self.val_num:]
        self.Y_test = Y[self.train_num + self.val_num:]

    def get_datasets(self):
        return self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test