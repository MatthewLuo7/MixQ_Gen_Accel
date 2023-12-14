import random
from random import choice, sample
import json
from dsp_eff_search import DSP_Config_Search
import math

name_mapping = {
    'opt_type': 'Opt_Type',
    'n': 'No',
    'k': 'K',
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
    'lshift': 'L_SHIFT',
    'kpf': 'KPF',
    'max_pool': 'MAX_POOL',
    'T_mul': 'T_mul'
    # 'pack_flag': 'PACK_FLAG'
    }

class ConvParam: ...

class FP_Opt_Sample:
	def __init__(self, conv):
		self.conv = conv

	def check_attr(self):
		for k, v in name_mapping.items():
			if (not hasattr(self.conv, k)) or (self.conv.k is None):
				print(f'Check error! There is no {str(k)} in opt or the value is none.')
				return False

		return True

	def gen_vec(self):
		config_list = []
		for k, v in name_mapping.items():
			config_list.append(getattr(self.conv, k))
		config_vec = tuple(config_list)

		return config_vec


	def check_constraints(self):
		if self.check_attr():
			C1 = self.conv.k <= self.conv.icol
			C2 = self.conv.k <= (self.conv.icol / 2)
			C3 = self.conv.in_pe <= self.conv.ich
			C4 = (self.conv.in_pe % self.conv.simd == 0) if self.conv.in_pe > self.conv.simd else (self.conv.simd % self.conv.in_pe == 0)
			C5 = self.conv.simd <= self.conv.ich
			C6 = self.conv.ich % self.conv.simd == 0
			C7 = self.conv.pe <= self.conv.och
			C8 = self.conv.och % self.conv.pe == 0
			C9 = self.conv.pe % self.conv.actp == 0
			mbit = self.conv.abit + self.conv.wbit + math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))
			C10 = self.conv.in_pe * self.conv.abit <= 1024
			C11 = self.conv.np * self.conv.simd * self.conv.kpf * self.conv.abit <= 1024
			C12 = self.conv.np * self.conv.pe * mbit <= 1024
			C13 = self.conv.pe * self.conv.obit <= 1024
			C14 = self.conv.k * self.conv.simd * self.conv.kpf * self.conv.wbit <= 1024
			C15 = self.conv.simd * self.conv.pe * self.conv.kpf <= 192
			C16 = self.conv.simd * self.conv.kpf <= 48

			return C1 and C2 and C3 and C4 and C5 and C6 and C7 and C8 and C9 and C10 and C11 and C12 and C13 and C14 and C15 and C15 and C16

		else:
			return False

class KP_Opt_Sample(FP_Opt_Sample):

	def check_constraints(self):
		if self.check_attr():
			C1 = self.conv.k <= self.conv.icol
			C2 = self.conv.k <= (self.conv.icol / 2)
			C3 = self.conv.in_pe <= self.conv.ich
			C4 = (self.conv.in_pe % self.conv.simd == 0) if self.conv.in_pe > self.conv.simd else (self.conv.simd % self.conv.in_pe == 0)
			C5 = self.conv.simd <= self.conv.ich
			C6 = self.conv.ich % self.conv.simd == 0
			C7 = (self.conv.pe * self.conv.kp) <= self.conv.och
			C8 = self.conv.och % (self.conv.pe * self.conv.kp) == 0
			C9 = (self.conv.pe * self.conv.kp) % self.conv.actp == 0
			mbit = self.conv.abit + self.conv.wbit + math.ceil(math.log2(self.conv.k * self.conv.k * self.conv.ich))
			C10 = self.conv.in_pe * self.conv.abit <= 1024
			C11 = self.conv.np * self.conv.simd * self.conv.kpf * self.conv.abit <= 1024
			C12 = self.conv.np * self.conv.pe * self.conv.kp * mbit <= 1024
			C13 = self.conv.pe * self.conv.kp * self.conv.obit <= 1024
			C14 = self.conv.kp * self.conv.simd * self.conv.kpf * self.conv.wbit <= 1024
			C15 = self.conv.simd * self.conv.pe * self.conv.kpf <= 192
			C16 = self.conv.simd * self.conv.kpf <= 48

			return C1 and C2 and C3 and C4 and C5 and C6 and C7 and C8 and C9 and C10 and C11 and C12 and C13 and C14 and C15 and C15

		else:
			return False


class sample_opt:
	def __init__(self, require_num, attempt_num):
		# self.set_opt_type = ['FP_Opt', 'KP_Opt']
		self.set_opt_type = ['FP_Opt']
		self.set_K = [3]
		self.set_OPT_ARCH = [(320, 160, 3, 16, True), (160, 80, 16, 32, True), (80, 40, 32, 64, True), (40, 20, 64, 64, True), (20, 10, 64, 64, False)]
		self.set_PF = [1, 2, 3, 4, 8, 16, 32]      # self.set_PF = [1, 2, 4, 8, 16, 32, 64]
		self.set_BIT = [2, 3, 4, 5, 6, 7, 8]
		self.set_KPF = [1, 3]
		# self.set_opt_type = ['FP_Opt']
		# self.set_K = [3]
		# self.set_OPT_ARCH = [(20, 10, 64, 64, False)]
		# self.set_PF = [2, 16]      # self.set_PF = [1, 2, 4, 8, 16, 32, 64]
		# self.set_BIT = [2, 7]
		# self.set_KPF = [1]
		self.require_num = require_num
		self.attempt_num = attempt_num
		self.cur_samples = ()
		self.prev_samples = ()
		self.find_num = 0
		self.sample_dict = {}

	def dict_to_vec(self, config):
		config_list = []
		for k, v in name_mapping.items():
			if config[v] is None:
				print(f'Failed to load previous operator. There is no feature {str(v)}')
				exit(0)

			config_list.append(config[v])
		config_vec = tuple(config_list)

		return config_vec

	def load_prev_samples(self, prev_samples_json):
		with open(prev_samples_json, 'r') as fdict:
			prev_dataset = json.load(fdict)
		
		for (opt_key, opt_val) in prev_dataset.items():
			config = opt_val

			# add features
			config['Opt_Type'] = 'FP_Opt'
			config['KPF'] = 1
			config['No'] = 0

			config_vec = self.dict_to_vec(config)
	
			temp_list = list(self.prev_samples)
			temp_list.append(config_vec)
			self.prev_samples = tuple(temp_list)


	def random_sampling(self):
		prev_samples_json = './opt_samples_4.json'
		self.load_prev_samples(prev_samples_json)
		for i in range(self.attempt_num):
			conv = ConvParam()
			conv.opt_type = choice(self.set_opt_type)
			conv.n = 0
			conv.k = 3                                # choice(self.set_K)
			conv.icol, conv.irow, conv.ich, conv.och, conv.max_pool = choice(self.set_OPT_ARCH)

			conv.abit = choice(self.set_BIT)
			conv.wbit = choice(self.set_BIT)
			conv.obit = choice(self.set_BIT)
			if conv.icol == 320:
				conv.in_pe = 3
			else:
				conv.in_pe = choice(self.set_PF)
			conv.simd = choice(self.set_PF)
			conv.pe = choice(self.set_PF)
			conv.actp = choice(self.set_PF)
			conv.lshift = 8
			conv.kpf = choice(self.set_KPF)

			po = DSP_Config_Search(27, 18, 8)
			# conv.incbit = conv.wbit + conv.abit + random.randint(5, 10)
			# conv.biasbit = conv.wbit + conv.abit + random.randint(10, 20)
			conv.incbit = conv.wbit + conv.abit + 6
			conv.biasbit = conv.wbit + conv.abit + 15
			if conv.opt_type == 'FP_Opt':
				conv.kp, conv.np, conv.gb, conv.T_mul = po.Filter_Packing(conv.k, conv.abit, conv.wbit, 0)
				conv.pack_flag = False
				opt = FP_Opt_Sample(conv)
			elif conv.opt_type == 'KP_Opt':
				pack_flag, Ep, Dp, conv.gb, conv.T_mul = po.Kernel_Packing(conv.wbit, conv.abit, 0)
				if pack_flag:
					conv.np = Ep
					conv.kp = Dp
					conv.pack_flag = True
				else:
					conv.kp = Ep
					conv.np = Dp
					conv.pack_flag = False
				opt = KP_Opt_Sample(conv)

			vec = opt.gen_vec()

			if vec in self.prev_samples:
				print('in prev')
			if vec in self.cur_samples:
				print("in current")
			if opt.check_constraints() and (not vec in self.cur_samples) and (not vec in self.prev_samples):
				temp_list = list(self.cur_samples)
				temp_list.append(vec)
				self.cur_samples = tuple(temp_list)
				self.find_num += 1

				if self.find_num == self.require_num:
					print(f"Successfully collected {self.require_num} samples within {i + 1} attempts!")
					break

		if (self.find_num < self.require_num):
			print(f"Failed to collect required samples within {self.attempt_num} attempts, only collected {self.find_num} samples. Sorry!")
			return False
		else:
			return True

	def save_sample(self):
		for idx, vec in enumerate(self.cur_samples):
			dict_elem = {}
			for i, (k, v) in enumerate(name_mapping.items()):
				dict_elem[v] = vec[i]

			self.sample_dict['OPT_' + str(idx)] = dict_elem

		with open('./opt_samples_5.json', 'w', encoding='utf-8') as f:
			json.dump(self.sample_dict, f, indent=4)

		return



Sample_NUM = 10000
Require_NUM = 1000
sop = sample_opt(Require_NUM, Sample_NUM)
if sop.random_sampling():
	sop.save_sample()
