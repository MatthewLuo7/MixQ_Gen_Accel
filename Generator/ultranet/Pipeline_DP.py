import numpy as np
import math

class DP_node:
    def __init__(self, Init_Lat=99999999999):
        self.Lat = Init_Lat
        self.SIMD = []
        self.PE = []
        self.ACTP = []
        self.KPF = []


class Pipeline_Allocation:
	def __init__(self, model_opt, DSP_max):
		self.model_opt = model_opt
		self.n_layers = len(model_opt)
		self.DSP_max = DSP_max
		self.DPT = [[[DP_node(self.model_opt[k].dsp_operations()) for i in range(2)] for j in range(self.DSP_max + 1)] for k in range(self.n_layers)]

	def translate_kpf(self, kpf_flag, conv):
		kpf = 1 if kpf_flag == 0 else conv.k
		return kpf

	def check_constraint(self, inpe, simd, pe, kpf, conv):
		# if kpf * simd >= inpe:
		# 	flag = (kpf * simd % inpe == 0)
		# else:
		# 	flag = (inpe % (kpf * simd) == 0)

		flag = (kpf * simd >= inpe) and (kpf * simd % inpe == 0)
		flag = flag and (conv.ich % (kpf * simd) == 0) and (conv.och % pe == 0)

		return flag

	def Traverse_Solutions(self, layer, opt, DSP_aval, kpf_flag):
		conv = opt.conv
		kpf = self.translate_kpf(kpf_flag, conv)
		if DSP_aval == 0:
			return
		simd_p_max = math.floor(math.log2(min(DSP_aval, conv.ich) / kpf))
		pe_p_max = math.floor(math.log2(min(DSP_aval, conv.och)))
		if (simd_p_max < 0) or (pe_p_max < 0):
			return
		for simd_p in range(0, simd_p_max + 1):
			for pe_p in range(0, pe_p_max + 1):

				simd = 2 ** simd_p
				pe = 2 ** pe_p
				actp = 0

				cur_dsp = kpf * simd * pe + actp
				cur_Lat = opt.dsp_operations() / (kpf * simd * pe)

				cur_Node = self.DPT[layer][DSP_aval][kpf_flag]
				if (cur_dsp > DSP_aval):
					continue

				if layer == 0:
					if not self.check_constraint(3, simd, pe, kpf, conv):
						continue
					if cur_Lat < cur_Node.Lat:
						cur_Node.Lat = cur_Lat
						cur_Node.SIMD = [simd]
						cur_Node.PE = [pe]
						cur_Node.ACTP = [actp]
						cur_Node.KPF = [kpf]

				else:
					for last_kpf_flag in range(2):
						prev_Node = self.DPT[layer - 1][DSP_aval - cur_dsp][last_kpf_flag]
						# debug
						if len(prev_Node.PE) == 0:
							continue
							# print(layer, DSP_aval, last_kpf_flag)
						inpe = prev_Node.PE[-1]
						if not self.check_constraint(inpe, simd, pe, kpf, conv):
							continue

						cur_Lat = max(cur_Lat, prev_Node.Lat)
	
						if cur_Lat < cur_Node.Lat:
							cur_Node.Lat = cur_Lat
							cur_Node.SIMD = prev_Node.SIMD.copy()
							cur_Node.SIMD.extend([simd])
							cur_Node.PE = prev_Node.PE.copy()
							cur_Node.PE.extend([pe])
							cur_Node.ACTP = prev_Node.ACTP.copy()
							cur_Node.ACTP.extend([actp])
							cur_Node.KPF = prev_Node.KPF.copy()
							cur_Node.KPF.extend([kpf])


	def DP_Search(self):
		for layer in range(self.n_layers):
			for DSP_aval in range(0, self.DSP_max + 1):
				for kpf_flag in range(2):
					self.Traverse_Solutions(layer, self.model_opt[layer], DSP_aval, kpf_flag)

		best_kpf_flag = None
		best_Lat = 99999999999
		for kpf_flag in range(2):
			cur_Lat = self.DPT[self.n_layers - 1][self.DSP_max][kpf_flag].Lat

			if cur_Lat < best_Lat:
				best_Lat = cur_Lat
				best_kpf_flag = kpf_flag

		best_Node = self.DPT[self.n_layers - 1][self.DSP_max][best_kpf_flag]
		SIMD_list = best_Node.SIMD
		PE_list = best_Node.PE
		ACTP_list = best_Node.ACTP
		KPF_list = best_Node.KPF

		return best_Lat, SIMD_list, PE_list, ACTP_list, KPF_list


