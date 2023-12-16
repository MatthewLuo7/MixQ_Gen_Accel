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
		self.DPT = [[DP_node(self.model_opt[k].dsp_operations()) for j in range(self.DSP_max + 1)] for k in range(self.n_layers)]

	def translate_kpf(self, kpf_flag, conv):
		kpf = 1 if kpf_flag == 0 else conv.k
		return kpf

	def get_factors(self, m):
		factors = []
		for i in range(1, m + 1):
			if m % i == 0:
				factors.append(i)

		return factors

	def Traverse_Solutions(self, layer, opt, DSP_aval, simd_aval, pe_aval):
		conv = opt.conv
		if DSP_aval == 0:
			return

		for kpf in [1, conv.k]:
			for simd in simd_aval:
				for pe in pe_aval:
					valid_flag, actp = opt.get_actp(simd=simd, pe=pe, kpf=kpf)
					if not valid_flag:
						continue
	
					cur_dsp = kpf * simd * pe + actp
					cur_Lat = opt.dsp_operations() / (kpf * simd * pe)
					if (cur_dsp > DSP_aval):
						continue
	
					cur_Node = self.DPT[layer][DSP_aval]
					if layer == 0:
						if not opt.check_constraints(inpe=3, simd=simd, pe=pe, kpf=kpf):
							continue
						if cur_Lat < cur_Node.Lat:
							cur_Node.Lat = cur_Lat
							cur_Node.SIMD = [simd]
							cur_Node.PE = [pe]
							cur_Node.ACTP = [actp]
							cur_Node.KPF = [kpf]
	
					else:
						prev_Node = self.DPT[layer - 1][DSP_aval - cur_dsp]
						if len(prev_Node.PE) == 0:
							continue
						inpe = prev_Node.PE[-1]
						if not opt.check_constraints(inpe=inpe, simd=simd, pe=pe, kpf=kpf):
							continue
	
						new_Lat = max(cur_Lat, prev_Node.Lat)
		
						if new_Lat < cur_Node.Lat:
							cur_Node.Lat = new_Lat
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
			simd_aval = self.get_factors(self.model_opt[layer].conv.ich)
			pe_aval = self.get_factors(self.model_opt[layer].conv.och)
			for DSP_aval in range(0, self.DSP_max + 1):
				self.Traverse_Solutions(layer, self.model_opt[layer], DSP_aval,
										simd_aval, pe_aval)

		best_Node = self.DPT[self.n_layers - 1][self.DSP_max]
		SIMD_list = best_Node.SIMD
		PE_list = best_Node.PE
		ACTP_list = best_Node.ACTP
		KPF_list = best_Node.KPF
		best_Lat = best_Node.Lat

		return best_Lat, SIMD_list, PE_list, ACTP_list, KPF_list


