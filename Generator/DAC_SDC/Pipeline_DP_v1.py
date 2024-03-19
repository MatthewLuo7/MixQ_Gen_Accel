import numpy as np
import math
import pathlib
from threading import Thread
from dsp_eff_search import DSP_Config_Search

from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_LUT import FP_LUT_Opt_Templates
from ConvOpt_FP_DW import FP_DW_Opt_Templates
from ConvOpt_FP_DW_LUT import FP_DW_LUT_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_KP_LUT import KP_LUT_Opt_Templates
from ConvOpt_1x1 import Conv1x1_Opt_Templates

import json

def resolve_opt(conv, Packing, DW=False, LUT=False, Last=False):
    if Last:
        return Conv1x1_Opt_Templates(conv)
    elif Packing == 'Filter_Packing':
        if DW:
            opt = FP_DW_LUT_Opt_Templates(conv) if LUT else FP_DW_Opt_Templates(conv)
            return opt
        else:
            opt = FP_LUT_Opt_Templates(conv) if LUT else FP_Opt_Templates(conv)
            return opt
    elif Packing == 'Kernel_Packing':
        if DW:
            raise TypeError(f"Kernel_Packing operator cannot be used for Depth-wise Convolution!")
        else:
            opt = KP_LUT_Opt_Templates(conv) if LUT else KP_Opt_Templates(conv)
            return opt
    else:
        raise TypeError(f"Operator {str(Packing)} is not defined!")

class DP_node:
    def __init__(self, Init_Lat=99999999999):
        self.Lat = Init_Lat
        self.SIMD = []
        self.PE = []
        self.ACTP = []
        self.KPF = []
        self.LUT = []


class Pipeline_Allocation:
	def __init__(self, model_param, thread_num, DSP_max, LUT_max, BRAM_max, pred_path, DSP_step=5, LUT_step=2000, BRAM_step=20, cycle=4.0):
		self.model_param = model_param
		self.n_layers = len(model_param)
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
		self.Init_Lat = max(tuple(9999999999.0 for l in range(self.n_layers)))
		self.DPT = [[[[DP_node(self.Init_Lat) for i in range(0, self.DSP_max_step + 1)] for j in range(0, self.LUT_max_step + 1)] for k in range(0, self.BRAM_max_step + 1)] for l in range(self.n_layers)]

		self.thread_num = thread_num
		self.pred_path = pred_path

	def get_factors(self, m):
		factors = []
		for i in range(1, m + 1):
			if m % i == 0:
				factors.append(i)

		return factors

	def Traverse_Solutions(self, layer, opt, DSP_aval, LUT_aval, BRAM_aval, simd_aval, pe_aval, LUT=False, Last=False):
		conv = opt.conv

		for kpf in [1, conv.k]:
			for pe in pe_aval:
				for simd in simd_aval:
					# get best actp, if none skip
					actp = pe if Last else opt.get_actp(simd=simd, pe=pe, kpf=kpf)
					if actp is None:
						continue

					# predict timing
					# con_II = bool(opt.predict(simd, pe, actp, kpf, 'II'))
					con_II = True
					con_wns = max(bool(opt.predict(simd, pe, actp, kpf, 'wns') < self.cycle), 0.0)
					if not (con_II and con_wns):
						continue

					# predict resources	
					# cur_dsp = kpf * simd * pe + actp
					cur_dsp = max(opt.predict(simd, pe, actp, kpf, 'dsp'), 0.0)
					cur_lut = max(opt.predict(simd, pe, actp, kpf, 'lut'), 0.0)
					cur_bram = max(opt.predict(simd, pe, actp, kpf, 'bram'), 0.0)
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
							cur_Node.LUT = [LUT]
	
					else:
						# # debug
						# if layer == 11:
						# 	C1 = ((layer - 1) < 0) or ((layer - 1) >= self.n_layers)
						# 	C2 = ((BRAM_aval - cur_bram) < 0) or ((BRAM_aval - cur_bram) > self.BRAM_max_step)
						# 	C3 = ((LUT_aval - cur_lut) < 0) or ((LUT_aval - cur_lut) > self.LUT_max_step)
						# 	C4 = ((DSP_aval - cur_dsp) < 0) or ((DSP_aval - cur_dsp) > self.DSP_max_step)
						# 	if C1 or C2 or C3 or C4:
						# 		print('error')
						# 		print(layer - 1, self.n_layers)
						# 		print(BRAM_aval - cur_bram, BRAM_aval, cur_bram, self.BRAM_max_step)
						# 		print(LUT_aval - cur_lut, LUT_aval, cur_lut, self.LUT_max_step)
						# 		print(DSP_aval - cur_dsp, DSP_aval, cur_dsp, self.DSP_max_step)
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
							cur_Node.LUT = prev_Node.LUT.copy()
							cur_Node.LUT.extend([LUT])

	def Fuse_Search_Loops(self, conv, Packing, DW, LUT, layer, simd_aval, pe_aval, thread_idx):
		Last = (layer == (len(self.model_param) - 1))
		opt = resolve_opt(conv, Packing, DW, LUT, last=Last)
		if Packing == 'Filter_Packing':
			PK = 'FP'
		elif Packing == 'Kernel_Packing':
			PK = 'KP'
		else:
			raise NotImplementedError(f'{Packing} is not implemented!')
		path = self.pred_path / f'{PK}_{DW}_{LUT}'
		opt.Load_Model(path)

		total_iter = (self.BRAM_max_step + 1) * (self.LUT_max_step + 1) * (self.DSP_max_step + 1)
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

			self.Traverse_Solutions(layer, opt, DSP_aval, LUT_aval, BRAM_aval, simd_aval_cp, pe_aval_cp, LUT, Last)

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
		DSP_Explorer = DSP_Config_Search(27, 18)
		Packing_list = []
		DW_list = []
		for idx, conv in enumerate(self.model_param):
			# check depth-wise
			if conv.w.shape[1] == 1 and conv.ich != 1:   # depth-width
				acc_num = conv.k
				DW = True
			else:
				acc_num = conv.k * conv.ich
				DW = False

			# DSP-packing search
			if idx == (len(self.model_param) - 1):
			    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=False, Kernel_Packing_EN=True, acc_num=acc_num, och=conv.och)
			elif conv.w.shape[1] == 1:        # depth-wise conv
			    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=False, acc_num=acc_num, och=conv.och)
			else:
			    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True, acc_num=acc_num, och=conv.och)

			conv.kp = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['kp']
			conv.np = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['np']
			conv.gb = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['gb']
			conv.w_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['w_sep']
			conv.a_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['a_sep']
		
			Packing = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Packing_Type']
			if Packing == 'Kernel_Packing':
				conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']

			simd_aval = [1] if DW else self.get_factors(conv.ich)
			if idx == (len(self.model_param) - 1):
				conv.kp = 1
				conv.np = 2
				conv.gb = 2
				conv.w_sep = 1
				conv.a_sep = 1
				pe_aval = [2]
			else:
				pe_aval = self.get_factors(conv.och)

			Packing_list.append(Packing)
			DW_list.append(DW)

			if (conv.abit <= 4) and (conv.wbit <= 4):
				print(f'Try to implement LUT-based operators in layer {idx}.')
				LUT = True
				thread_list = []
				for thread_idx in range(0, self.thread_num):
					thrd = Thread(target=self.Fuse_Search_Loops, args=(conv, Packing, DW, LUT, idx, simd_aval, pe_aval, thread_idx), daemon=True)
					thread_list.append(thrd)
					thrd.start()
	
				for thrd in thread_list:
					thrd.join()
	
				print(f'Layer {idx} finished (LUT)!')

			LUT = False
			thread_list = []
			for thread_idx in range(0, self.thread_num):
				thrd = Thread(target=self.Fuse_Search_Loops, args=(conv, Packing, DW, LUT, idx, simd_aval, pe_aval, thread_idx), daemon=True)
				thread_list.append(thrd)
				thrd.start()
	
			for thrd in thread_list:
				thrd.join()
	
			print(f'Layer {idx} finished!')


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
		print(f'Best Node: [{self.n_layers- 1}][{bram_const}][{lut_const}][{dsp_const}], Latency: {best_Node.Lat}')

		SIMD_list = best_Node.SIMD
		PE_list = best_Node.PE
		ACTP_list = best_Node.ACTP
		KPF_list = best_Node.KPF
		LUT_list = best_Node.LUT

		if len(SIMD_list) == 0:
			print(f'Failed to find a solution! Exit!')
			print_dict = {}
			for l in range(self.n_layers):
				for k in range(0, self.BRAM_max_step + 1):
					for j in range(0, self.LUT_max_step + 1):
						for i in range(0, self.DSP_max_step + 1):
							cur_Node = self.DPT[l][k][j][i]
							print_dict[f'{l}_{k}_{j}_{i}'] = {'Lat': cur_Node.Lat, f'SIMD': cur_Node.SIMD, 'PE': cur_Node.PE, 'ACTP': cur_Node.ACTP, 'KPF': cur_Node.KPF, 'LUT': cur_Node.LUT}

			with open('./DP_debug/DP_table.json', 'w', encoding='utf-8') as f:
				json.dump(print_dict, f, indent=4)
			exit(0)

		print(bram_const, lut_const, dsp_const)

		return best_Lat, Packing_list, DW_list, SIMD_list, PE_list, ACTP_list, KPF_list, LUT_list


	# def DP_Search(self):
	# 	for layer in range(self.n_layers):
	# 		Packing = self.model_opt[layer].opt_type()['Packing']
	# 		DW = self.model_opt[layer].opt_type()['DW']
	# 		LUT = self.model_opt[layer].opt_type()['LUT']

	# 		simd_aval = [1] if DW else self.get_factors(self.model_opt[layer].conv.ich)
	# 		pe_aval = self.get_factors(self.model_opt[layer].conv.och)
	# 		path = pathlib.Path(f'E:/Projects/DeepBurning_MixQ/MixQ_Gen_Accel/Dataset/6_OPT_5000/{Packing}_{DW}_{LUT}/')
	# 		self.model_opt[layer].Load_Model(path)

	# 		thread_list = []
	# 		for thread_idx in range(0, self.thread_num):
	# 			thrd = Thread(target=self.Fuse_Search_Loops, args=(layer, simd_aval, pe_aval, thread_idx), daemon=True)
	# 			thread_list.append(thrd)
	# 			thrd.start()

	# 		for thrd in thread_list:
	# 			thrd.join()

	# 		print(f'layer {layer} finished!')

	# 	# search best node
	# 	dsp_const = 0
	# 	lut_const = 0
	# 	bram_const = 0
	# 	best_Lat = self.Init_Lat
	# 	print("Init_Lat:", self.Init_Lat)
	# 	for cur_dsp in range(0, self.DSP_max_step + 1):
	# 		for cur_lut in range(0, self.LUT_max_step + 1):
	# 			for cur_bram in range(0, self.BRAM_max_step + 1):
	# 				cur_Lat = self.DPT[self.n_layers - 1][cur_bram][cur_lut][cur_dsp].Lat
	# 				if cur_Lat < best_Lat:
	# 					dsp_const = cur_dsp
	# 					lut_const = cur_lut
	# 					bram_const = cur_bram
	# 					best_Lat = cur_Lat

	# 	best_Node = self.DPT[self.n_layers - 1][bram_const][lut_const][dsp_const]
	# 	SIMD_list = best_Node.SIMD
	# 	PE_list = best_Node.PE
	# 	ACTP_list = best_Node.ACTP
	# 	KPF_list = best_Node.KPF

	# 	print(bram_const, lut_const, dsp_const)

	# 	return best_Lat, SIMD_list, PE_list, ACTP_list, KPF_list


