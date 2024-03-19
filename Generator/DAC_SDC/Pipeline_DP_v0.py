import numpy as np
import math
import pathlib
from threading import Thread

class DP_node:
    def __init__(self, Init_Lat=99999999999):
        self.Lat = Init_Lat
        self.SIMD = []
        self.PE = []
        self.ACTP = []
        self.KPF = []


class Pipeline_Allocation:
	def __init__(self, model_opt, thread_num, DSP_max, LUT_max, BRAM_max, DSP_step=5, LUT_step=2000, BRAM_step=20, cycle=4.0):
		self.model_opt = model_opt
		self.n_layers = len(model_opt)
		# dsp
		self.DSP_max = DSP_max
		self.DSP_step = DSP_step
		self.DSP_max_step = math.ceil(self.DSP_max / self.DSP_step)
		# lut
		self.LUT_max = LUT_max
		self.LUT_step = LUT_step
		self.LUT_max_step = math.ceil(self.LUT_max / self.LUT_step)
		# bram
		self.BRAM_max = BRAM_max
		self.BRAM_step = BRAM_step
		self.BRAM_max_step = math.ceil(self.BRAM_max / self.BRAM_step)

		self.cycle = cycle
		self.Init_Lat = max(tuple(self.model_opt[l].dsp_operations() for l in range(self.n_layers)))
		self.DPT = [[[[DP_node(self.Init_Lat) for i in range(0, self.DSP_max_step + 1)] for j in range(0, self.LUT_max_step + 1)] for k in range(0, self.BRAM_max_step + 1)] for l in range(self.n_layers)]

		self.thread_num = thread_num

	def get_factors(self, m):
		factors = []
		for i in range(1, m + 1):
			if m % i == 0:
				factors.append(i)

		return factors

	def Traverse_Solutions(self, layer, opt, DSP_aval, LUT_aval, BRAM_aval, simd_aval, pe_aval):
		conv = opt.conv

		for kpf in [1, conv.k]:
			for pe in pe_aval:
				for simd in simd_aval:
					# get best actp, if none skip
					actp = opt.get_actp(simd=simd, pe=pe, kpf=kpf)
					if actp is None:
						continue

					# predict timing
					# con_II = bool(opt.predict(simd, pe, actp, kpf, 'II'))
					con_II = True
					con_wns = bool(opt.predict(simd, pe, actp, kpf, 'wns') < self.cycle)
					if not (con_II and con_wns):
						continue

					# predict resources	
					# cur_dsp = kpf * simd * pe + actp
					cur_dsp = opt.predict(simd, pe, actp, kpf, 'dsp')
					cur_lut = opt.predict(simd, pe, actp, kpf, 'lut')
					cur_bram = opt.predict(simd, pe, actp, kpf, 'bram')
					cur_dsp = round(cur_dsp / self.DSP_step)
					cur_lut = round(cur_lut / self.LUT_step)
					cur_bram = round(cur_bram / self.BRAM_step)
					if (cur_dsp > DSP_aval) or (cur_lut > LUT_aval) or (cur_bram > BRAM_aval):
						continue

					# current latency and DP node
					cur_Lat = opt.dsp_operations() / (kpf * simd * pe)
					cur_Node = self.DPT[layer][BRAM_aval][LUT_aval][DSP_aval]

					# DP recursion
					if layer == 0:
						if not opt.opt_constraints(inpe=3, simd=simd, kpf=kpf, pe=pe, actp=actp):
							continue

						if cur_Lat < cur_Node.Lat:
							cur_Node.Lat = cur_Lat
							cur_Node.SIMD = [simd]
							cur_Node.PE = [pe]
							cur_Node.ACTP = [actp]
							cur_Node.KPF = [kpf]
	
					else:
						prev_Node = self.DPT[layer - 1][BRAM_aval - cur_bram][LUT_aval - cur_lut][DSP_aval - cur_dsp]
						if len(prev_Node.PE) == 0:
							continue
						inpe = prev_Node.PE[-1]

						if not opt.opt_constraints(inpe=inpe, simd=simd, kpf=kpf, pe=pe, actp=actp):
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

	def Fuse_Search_Loops(self, layer, simd_aval, pe_aval, thread_idx):
		total_iter = (self.BRAM_max_step + 1) * (self.LUT_max_step + 1) * (self.DSP_max_step + 1)
		opt = self.model_opt[layer]
		opt.conv.cycle = self.cycle
		simd_aval_cp = simd_aval.copy()
		pe_aval_cp = pe_aval.copy()
		for iter_idx in range(0, total_iter, self.thread_num):
			actual_iter_idx = iter_idx + thread_idx
			if actual_iter_idx >= total_iter:
				continue

			DSP_aval = actual_iter_idx % (self.DSP_max_step + 1)
			lut_iter = actual_iter_idx // (self.DSP_max_step + 1)
			LUT_aval = lut_iter % (self.LUT_max_step + 1)
			bram_iter = lut_iter // (self.LUT_max_step + 1)
			BRAM_aval = bram_iter % (self.BRAM_max_step + 1)

			self.Traverse_Solutions(layer, opt, DSP_aval, LUT_aval, BRAM_aval, simd_aval_cp, pe_aval_cp)

	# def Fuse_Search_Loops(self, layer, simd_aval, pe_aval, thread_idx):
	# 	total_iter = (self.BRAM_max_step + 1) * (self.LUT_max_step + 1) * (self.DSP_max_step + 1)
	# 	batch_size = math.ceil(total_iter / self.thread_num)
	# 	for iter_idx in range(0, batch_size):
	# 		actual_iter_idx = iter_idx + thread_idx * batch_size
	# 		if actual_iter_idx >= total_iter:
	# 			continue

	# 		DSP_aval = actual_iter_idx % (self.DSP_max_step + 1)
	# 		lut_iter = actual_iter_idx // (self.DSP_max_step + 1)
	# 		LUT_aval = lut_iter % (self.LUT_max_step + 1)
	# 		bram_iter = lut_iter // (self.LUT_max_step + 1)
	# 		BRAM_aval = bram_iter % (self.BRAM_max_step + 1)

	# 		self.Traverse_Solutions(layer, self.model_opt[layer], DSP_aval, LUT_aval, BRAM_aval, simd_aval, pe_aval)

	def DP_Search(self):
		for layer in range(self.n_layers):
			Packing = self.model_opt[layer].opt_type()['Packing']
			DW = self.model_opt[layer].opt_type()['DW']
			LUT = self.model_opt[layer].opt_type()['LUT']

			simd_aval = [1] if DW else self.get_factors(self.model_opt[layer].conv.ich)
			pe_aval = self.get_factors(self.model_opt[layer].conv.och)
			path = pathlib.Path(f'E:/Projects/DeepBurning_MixQ/MixQ_Gen_Accel/Dataset/6_OPT_5000/{Packing}_{DW}_{LUT}/')
			self.model_opt[layer].Load_Model(path)

			thread_list = []
			for thread_idx in range(0, self.thread_num):
				thrd = Thread(target=self.Fuse_Search_Loops, args=(layer, simd_aval, pe_aval, thread_idx), daemon=True)
				thread_list.append(thrd)
				thrd.start()

			for thrd in thread_list:
				thrd.join()

			print(f'layer {layer} finished!')

		# search best node
		dsp_const = 0
		lut_const = 0
		bram_const = 0
		best_Lat = self.Init_Lat
		print("Init_Lat:", self.Init_Lat)
		for cur_dsp in range(0, self.DSP_max_step + 1):
			for cur_lut in range(0, self.LUT_max_step + 1):
				for cur_bram in range(0, self.BRAM_max_step + 1):
					cur_Lat = self.DPT[self.n_layers - 1][cur_bram][cur_lut][cur_dsp].Lat
					if cur_Lat < best_Lat:
						dsp_const = cur_dsp
						lut_const = cur_lut
						bram_const = cur_bram
						best_Lat = cur_Lat

		best_Node = self.DPT[self.n_layers - 1][bram_const][lut_const][dsp_const]
		SIMD_list = best_Node.SIMD
		PE_list = best_Node.PE
		ACTP_list = best_Node.ACTP
		KPF_list = best_Node.KPF

		print(bram_const, lut_const, dsp_const)

		return best_Lat, SIMD_list, PE_list, ACTP_list, KPF_list


