import argparse
from random import choice, sample
import json
import pathlib
import math
import numpy as np
import sys
sys.path.append('../DSP_Explorer/')
sys.path.append('../Opt_Definition/')
from dsp_eff_search import DSP_Config_Search
import time

from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_LUT import FP_LUT_Opt_Templates
from ConvOpt_FP_DW import FP_DW_Opt_Templates
from ConvOpt_FP_DW_LUT import FP_DW_LUT_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_KP_LUT import KP_LUT_Opt_Templates

candidate_latency = [10.0, 6.667, 4.667, 4.0, 3.333, 2.667]     # lacency (ns)

candidate_K = [1, 3, 5, 7]
# candidate_W = list(np.arange(8, 640 + 1, 2))
candidate_W = [8, 16, 32, 64, 128, 256, 512, 10, 20, 40, 80, 160, 320, 640, 14, 28, 56, 112, 224, 448]
# candidate_CH = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 384, 512]   # 768 and 1024 seem too large
candidate_CH = [1, 2, 4, 6, 8, 12] + [3, 16, 24, 36, 384, 512] * 16 + [32, 48, 64, 96, 128, 192, 256] * 32
candidate_BIT = list(np.arange(2, 8 + 1, 1))
candidate_POOL = [True, False]
candidate_lshift = list(np.arange(4, 16 + 1, 1))
candidate_Packing = ['FP', 'KP']
candidate_overlap = [0, 1]
candidate_LUT = [False] * 8 + [True] * 2
candidate_DW = [False] * 8 + [True] * 2
DSP_Explorer = DSP_Config_Search(27, 18)

attr_names = ['k', 'icol', 'irow', 'ich', 'och', 'max_pool', 'abit', 'wbit', 'obit', 'kp',
			 'np', 'gb', 'w_sep', 'a_sep', 'simd', 'pe', 'actp', 'kpf', 'in_pe', 'lshift',
			 'incbit', 'biasbit', 'pack_flag']

def attr_to_tuple(conv, latency):
	attr_list = []
	for elem in attr_names[:19]:                     # lshift, incbit, biasbit are not important
		attr_list.append(getattr(conv, elem))

	attr_list.append(latency)

	return tuple(attr_list)

class ConvParam: ...

def sample_from_list(candidate_list):
    return choice(candidate_list)

def get_factors(m):
    factors = []
    for i in range(1, m + 1):
        if m % i == 0:
            factors.append(i)

    return factors

def get_pf(m):
    low_freq = []
    high_freq = []
    for i in range(1, m + 1):
        if m % i == 0:
            if(i < 4) or (i >= 64):
                low_freq.append(i)
            else:
                high_freq.append(i)

    factors = low_freq + high_freq * 8
    return factors

def Random_Opt():
    conv = ConvParam()
    # packing
    Packing = sample_from_list(candidate_Packing)
    if Packing == 'FP':
        # DW = sample_from_list([True, False])
        DW = sample_from_list(candidate_DW)
    else:
        DW = False
    LUT = sample_from_list(candidate_LUT)

    # sample architecture
    conv.k = sample_from_list(candidate_K)
    conv.icol = sample_from_list(candidate_W)
    # conv.irow = sample_from_list(candidate_W)
    irow_list = []
    for irow in candidate_W:
        if ((conv.icol / irow) >= (1.0 / 2.5) and ((conv.icol / irow) <= 2.5)):
            irow_list.append(irow)
    conv.irow = sample_from_list(irow_list)
    # conv.ich = 1 if DW else sample_from_list(candidate_CH)
    conv.och = sample_from_list(candidate_CH)
    ich_list = []
    for ich in candidate_CH:
        if ((conv.och / ich) >= (1.0 / 8.0) and ((conv.och / ich) <= 8.0)):
            ich_list.append(ich)
    conv.ich = 1 if DW else sample_from_list(ich_list)
    conv.max_pool = sample_from_list(candidate_POOL)
    acc_num = conv.k if DW else conv.k * conv.ich

    # sample bit-width
    conv.abit = sample_from_list(candidate_BIT)
    conv.wbit = sample_from_list(candidate_BIT)
    conv.obit = sample_from_list(candidate_BIT)

    conv.lshift = sample_from_list(candidate_lshift)
    T = conv.abit + conv.wbit + conv.lshift - 1
    skew = min(8, T - 2)
    inc_bias_skew = list(np.arange(-skew, skew + 1) + T)
    conv.incbit = min(sample_from_list(inc_bias_skew), 27)
    conv.biasbit = min(sample_from_list(inc_bias_skew), 40)

    # dsp search
    overlap = sample_from_list(candidate_overlap)
    if Packing == 'FP':
        DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8,
                                                             Filter_Packing_EN=True, Kernel_Packing_EN=False, acc_num=acc_num,
                                                             och=conv.och)
    else:
        DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8,
                                                             Filter_Packing_EN=False, Kernel_Packing_EN=True, acc_num=acc_num,
                                                             och=conv.och)
    conv.kp = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['kp']
    conv.np = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['np']
    conv.gb = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['gb']
    # if not LUT:
    conv.w_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['w_sep']
    conv.a_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['a_sep']

    # sample parallelism factors
    candidate_ICH_PF = get_pf(conv.ich)
    conv.simd = sample_from_list(candidate_ICH_PF)
    candidate_OCH_PF = get_pf(conv.och // conv.kp) if Packing == 'KP' else get_pf(conv.och)
    conv.pe = sample_from_list(candidate_OCH_PF)
    candidate_ACTP = get_factors(conv.pe * conv.kp) if Packing == 'KP' else get_factors(conv.pe)
    conv.actp = sample_from_list(candidate_ACTP)
    conv.kpf = sample_from_list([conv.k, 1])
    candidate_INPE = get_pf(conv.och) if DW else get_pf(conv.ich)
    conv.in_pe = sample_from_list(candidate_INPE)

    # latency (or frequency)
    latency = sample_from_list(candidate_latency)

    # constraints for parallelism factor
    pf_constraints = (conv.w_sep * conv.a_sep) * conv.simd * conv.kpf * conv.pe <= 150

    # constraints for on-chip buffer storage
    s2p_buffer = ((conv.k + 1) * conv.icol * conv.och * conv.abit / (8 * 1024)) if DW else ((conv.k + 1) * conv.icol * conv.ich * conv.abit / (8 * 1024))
    weight_rom = conv.k * conv.k * conv.ich * conv.och * conv.wbit / (8 * 1024)
    para_rom = (conv.biasbit + conv.incbit) * conv.och / (8 * 1024)
    storage_constraints = (s2p_buffer + weight_rom + para_rom) < 600

    if Packing == 'FP':
    	if DW:
    		opt = FP_DW_LUT_Opt_Templates(conv) if LUT else FP_DW_Opt_Templates(conv)
    	else:
    		opt = FP_LUT_Opt_Templates(conv) if LUT else FP_Opt_Templates(conv)
    else:
    	conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']
    	opt = KP_LUT_Opt_Templates(conv) if LUT else KP_Opt_Templates(conv)

    dim_constraints = max(max(opt.weight_shape()), max(opt.S2P_buffer_shape()), conv.och // conv.actp) <= 4096     # 8192

    constraint_flag =  pf_constraints and storage_constraints and opt.opt_constraints_() and dim_constraints
    attr_vec = attr_to_tuple(conv, latency)

    return constraint_flag, conv, Packing, DW, LUT, attr_vec, latency


class Opt_Sampling:
	def __init__(self, require_num, attempt_num):
		self.sample_tuple = ()
		self.find_num = 0
		self.sample_dict = {}
		self.require_num = require_num
		self.attempt_num = attempt_num

	def Sampling(self, save_path):
		for i in range(attempt_num):
			constraint_flag, conv, Packing, DW, LUT, attr_vec, latency = Random_Opt()

			C1 = constraint_flag
			C2 = not attr_vec in self.sample_tuple

			if C1 and C2:
				temp_list = list(self.sample_tuple)
				temp_list.append(attr_vec)
				self.sample_tuple = tuple(temp_list)

				dict_elem = {'Type': {'Packing': Packing, 'LUT': LUT, 'DW': DW, 'Latency': latency}}
				Config = {}
				for attr in attr_names:
					if hasattr(conv, attr):
						val = getattr(conv, attr)
						if isinstance(val, np.integer):
							val = int(val)
						Config[attr] = val
						
				dict_elem['Config'] = Config

				self.sample_dict['Sample_' + str(self.find_num)] = dict_elem

				self.find_num += 1
				if self.find_num == self.require_num:
					print(f"Successfully collected {self.require_num} samples within {i + 1} attempts!")
					break

		if (self.find_num == self.require_num):
			with open(save_path, 'w', encoding='utf-8') as f:
				json.dump(self.sample_dict, f, indent=4)
		else:
			print(f"Failed to collect required samples within {self.attempt_num} attempts, only collected {self.find_num} samples. Sorry!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rn', '--require-num', help='required number of samples')
    parser.add_argument('-an', '--attempt_num', help='number of sampling attenpts')
    parser.add_argument('-f', '--file-name', help='file name for saving generated samples')
    opt = parser.parse_args()

    sample_dir = pathlib.Path('./Samples')
    if not sample_dir.is_dir():
        sample_dir.mkdir()

    attempt_num = int(opt.attempt_num)
    require_num = int(opt.require_num)
    save_path = sample_dir / str(opt.file_name)
    opt_samp = Opt_Sampling(require_num=require_num, attempt_num=attempt_num)

    print(save_path)

    t1 = time.time()
    opt_samp.Sampling(save_path)
    t2 = time.time()
    print(f'Spent {t2 - t1} seconds in total.')