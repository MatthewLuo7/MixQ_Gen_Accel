from random import choice, sample
import json
import pathlib
import math
import numpy as np
import sys
sys.path.append('../DSP_Explorer/')
sys.path.append('../Opt_Definition/ConvOpt_FP/')
sys.path.append('../Opt_Definition/ConvOpt_FP_DW/')
sys.path.append('../Opt_Definition/ConvOpt_FP_DW_LUT/')
sys.path.append('../Opt_Definition/ConvOpt_FP_LUT/')
sys.path.append('../Opt_Definition/ConvOpt_KP/')
sys.path.append('../Opt_Definition/ConvOpt_KP_LUT/')
from dsp_eff_search import DSP_Config_Search

from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_LUT import FP_LUT_Opt_Templates
from ConvOpt_FP_DW import FP_DW_Opt_Templates
from ConvOpt_FP_DW_LUT import FP_DW_LUT_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_KP_LUT import KP_LUT_Opt_Templates

candidate_K = [1, 3, 5, 7]
candidate_W = list(np.arange(8, 640 + 1, 2))
candidate_CH = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
candidate_BIT = list(np.arange(2, 8 + 1, 1))
candidate_POOL = [True, False]
candidate_lshift = list(np.arange(4, 16 + 1, 1))
candidate_Packing = ['FP', 'KP']
candidate_LUT = [False, True]
DSP_Explorer = DSP_Config_Search(27, 18, 8)

attr_names = ['k', 'icol', 'irow', 'ich', 'och', 'max_pool', 'abit', 'wbit', 'obit', 'kp',
			 'np', 'gb', 'w_sep', 'a_sep', 'simd', 'pe', 'actp', 'kpf', 'lshift', 'incbit',
			 'biasbit', 'pack_flag']

def attr_to_tuple(conv):
	attr_list = []
	for elem in attr_names[:18]:                     # lshift, incbit, biasbit are not important
		attr_list.append(getattr(conv, elem))

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

def Random_Opt():
    conv = ConvParam()
    # packing
    Packing = sample_from_list(candidate_Packing)
    if Packing == 'FP':
        DW = sample_from_list([True, False])
    else:
        DW = False
    LUT = sample_from_list(candidate_LUT)

    # sample architecture
    conv.k = sample_from_list(candidate_K)
    conv.icol = sample_from_list(candidate_W)
    conv.irow = sample_from_list(candidate_W)
    conv.ich = 1 if DW else sample_from_list(candidate_CH)
    conv.och = sample_from_list(candidate_CH)
    conv.max_pool = sample_from_list(candidate_POOL)
    acc_num = conv.k if DW else conv.k * conv.ich

    # sample bit-width
    conv.abit = sample_from_list(candidate_BIT)
    conv.wbit = sample_from_list(candidate_BIT)
    conv.obit = sample_from_list(candidate_BIT)

    conv.lshift = sample_from_list(candidate_lshift)
    T = conv.abit + conv.wbit + conv.lshift - 1
    skew = min(12, T - 2)
    inc_bias_skew = list(np.arange(skew, skew + 1))
    conv.incbit = sample_from_list(inc_bias_skew)
    conv.biasbit = sample_from_list(inc_bias_skew)

    # dsp search
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
    candidate_ICH_PF = get_factors(conv.ich)
    conv.simd = sample_from_list(candidate_ICH_PF)
    candidate_OCH_PF = get_factors(conv.och // conv.kp) if Packing == 'KP' else get_factors(conv.och)
    conv.pe = sample_from_list(candidate_OCH_PF)
    candidate_ACTP = get_factors(conv.pe * conv.kp) if Packing == 'KP' else get_factors(conv.pe)
    conv.actp = sample_from_list(candidate_ACTP)
    conv.kpf = sample_from_list([conv.k, 1])
    candidate_INPE = get_factors(conv.och) if DW else get_factors(conv.ich)
    conv.inpe = sample_from_list(candidate_INPE)

    if Packing == 'FP':
    	if DW:
    		opt = FP_DW_LUT_Opt_Templates(conv) if LUT else FP_DW_Opt_Templates(conv)
    	else:
    		opt = FP_LUT_Opt_Templates(conv) if LUT else FP_Opt_Templates(conv)
    elif 'KP':
    	conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']
    	opt = KP_LUT_Opt_Templates(conv) if LUT else KP_Opt_Templates(conv)

    constraint_flag = opt.opt_constraints_()
    attr_vec = attr_to_tuple(conv)

    return constraint_flag, conv, Packing, DW, LUT, attr_vec


class Opt_Sampling:
	def __init__(self, sample_dir, require_num, attempt_num):
		self.sample_tuple = ()
		self.find_num = 0
		self.sample_dict = {}
		self.sample_dir = sample_dir
		self.require_num = require_num
		self.attempt_num = attempt_num

	def Sampling(self):
		for i in range(attempt_num):
			constraint_flag, conv, Packing, DW, LUT, attr_vec = Random_Opt()

			C1 = constraint_flag
			C2 = not attr_vec in self.sample_tuple

			if C1 and C2:
				temp_list = list(self.sample_tuple)
				temp_list.append(attr_vec)
				self.sample_tuple = tuple(temp_list)

				dict_elem = {'Type': {'Packing': Packing, 'LUT': LUT, 'DW': DW}}
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
			if not sample_dir.is_dir():
				sample_dir.mkdir()
			with open(self.sample_dir / 'opt_samples.json', 'w', encoding='utf-8') as f:
				json.dump(self.sample_dict, f, indent=4)
		else:
			print(f"Failed to collect required samples within {self.attempt_num} attempts, only collected {self.find_num} samples. Sorry!")


if __name__ == '__main__':
	attempt_num = 2000     # 50000
	require_num = 500       #5000
	sample_dir = pathlib.Path('./Samples')
	opt_samp = Opt_Sampling(sample_dir=sample_dir, require_num=require_num, attempt_num=attempt_num)
	opt_samp.Sampling()