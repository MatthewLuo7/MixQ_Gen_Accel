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

import time
import json

def get_factors(m):
	factors = []
	for i in range(1, m + 1):
		if m % i == 0:
			factors.append(i)

	return factors

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

def resolve_packing(conv_org, DSP_Explorer: DSP_Config_Search, Last=False):
	conv = conv_org

	# check if Depth-wise
	DW = conv.w.shape[1] == 1 and conv.ich != 1
	acc_num = conv.k if DW else conv.k * conv.ich

	# DSP-packing search
	if Last:
	    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=False, Kernel_Packing_EN=True, acc_num=acc_num, och=conv.och)
	elif conv.w.shape[1] == 1:        # depth-wise conv
	    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=False, acc_num=acc_num, och=conv.och)
	else:
	    DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True, acc_num=acc_num, och=conv.och)

	# set packing parameters
	conv.kp = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['kp']
	conv.np = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['np']
	conv.gb = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['gb']
	conv.w_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['w_sep']
	conv.a_sep = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['a_sep']
	
	Packing = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Packing_Type']
	if Packing == 'Kernel_Packing':
		conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']

	simd_aval = [1] if DW else get_factors(conv.ich)

	# special setting for the last layer (need to output)
	# TO BE OPTIMIZED
	if Last:
		conv.kp = 1
		conv.np = 2
		conv.gb = 2
		conv.w_sep = 1
		conv.a_sep = 1
		pe_aval = [2]
	else:
		pe_aval = get_factors(conv.och)

	return conv, Packing, DW, simd_aval, pe_aval

class DP_node_new:
    def __init__(self, Init_Lat=99999999999):
        self.Lat = Init_Lat
        self.SIMD = []
        self.PE = []
        self.ACTP = []
        self.KPF = []
        self.Packing = []
        self.DW = []
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
		self.DPT = [[[[DP_node_new(self.Init_Lat) for i in range(0, self.DSP_max_step + 1)] for j in range(0, self.LUT_max_step + 1)] for k in range(0, self.BRAM_max_step + 1)] for l in range(self.n_layers)]

		self.thread_num = thread_num
		self.pred_path = pred_path

		self.DSP_Explorer = DSP_Config_Search(27, 18)

	# return a matrix containing possible features for resource/timing estimation
	def get_feature_vectors(self, opt, simd_aval, pe_aval, Last=False):
		kpf_aval = [1, opt.conv.k]
		feature_list = []
		PF_config_list = []
		for kpf in kpf_aval:
			for pe in pe_aval:
				for simd in simd_aval:
					actp = pe if Last else opt.get_actp(simd=simd, pe=pe, kpf=kpf)
					if actp is None:
						continue
					features = opt.get_feature(simd, pe, actp, kpf)
					feature_list.append(features)
					PF_config_list.append([simd, kpf, pe, actp])

		return np.array(feature_list), np.array(PF_config_list)


	def Traverse_Solutions(self, items, layer_idx, opt, DSP_aval, LUT_aval, BRAM_aval, Packing, DW, LUT=False):
		# current latency and DP node
		cur_Node = self.DPT[layer_idx][BRAM_aval][LUT_aval][DSP_aval]

		for (cur_Lat, dsp, lut, bram, _, simd, kpf, pe, actp) in items:
			cur_dsp, cur_lut, cur_bram, simd, kpf, pe, actp = int(dsp), int(lut), int(bram), int(simd), int(kpf), int(pe), int(actp)
			# DP recursion
			if layer_idx == 0:
				if not opt.opt_constraints(inpe=3, simd=simd, kpf=kpf, pe=pe, actp=actp):
					continue
	
				if cur_Lat < cur_Node.Lat:
					cur_Node.Lat = cur_Lat
					cur_Node.SIMD = [simd]
					cur_Node.PE = [pe]
					cur_Node.ACTP = [actp]
					cur_Node.KPF = [kpf]
					cur_Node.Packing = [Packing]
					cur_Node.DW = [DW]
					cur_Node.LUT = [LUT]
		
			else:
				prev_Node = self.DPT[layer_idx - 1][BRAM_aval - cur_bram][LUT_aval - cur_lut][DSP_aval - cur_dsp]
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
					cur_Node.Packing = prev_Node.Packing.copy()
					cur_Node.Packing.extend([Packing])
					cur_Node.DW = prev_Node.DW.copy()
					cur_Node.DW.extend([DW])
					cur_Node.LUT = prev_Node.LUT.copy()
					cur_Node.LUT.extend([LUT])

	def Fuse_Search_Loops(self, opt, layer_idx, Pred_Lookup, thread_idx, Packing, DW, LUT=False):
		total_iter = (self.BRAM_max_step + 1) * (self.LUT_max_step + 1) * (self.DSP_max_step + 1)
		for iter_idx in range(0, total_iter, self.thread_num):
			actual_iter_idx = iter_idx + thread_idx
			if actual_iter_idx >= total_iter:
				continue

			# decode DSP_aval, LUT_aval, and BRAM_aval, from iteration index
			DSP_aval = actual_iter_idx % (self.DSP_max_step + 1)
			lut_iter = actual_iter_idx // (self.DSP_max_step + 1)
			LUT_aval = lut_iter % (self.LUT_max_step + 1)
			bram_iter = lut_iter // (self.LUT_max_step + 1)
			BRAM_aval = bram_iter % (self.BRAM_max_step + 1)

			# filter out unsatisfied items
			# [Lat, dsp, lut, bram, wns, simd, kpf, pe, actp]
			items_sat_wns = Pred_Lookup[Pred_Lookup[:, 4] < self.cycle]
			items_sat_dsp = items_sat_wns[items_sat_wns[:, 1] < DSP_aval]
			items_sat_lut = items_sat_dsp[items_sat_dsp[:, 2] < LUT_aval]
			items_sat_all = items_sat_lut[items_sat_lut[:, 3] < BRAM_aval]

			if items_sat_all is not None:
				self.Traverse_Solutions(items_sat_all, layer_idx, opt, DSP_aval, LUT_aval, BRAM_aval, Packing, DW, LUT)


	def Search_Layer(self, layer_idx, conv, Packing, simd_aval, pe_aval, DW, LUT=False, Last=False):
		# predict resources/timing with pre-trained predictors in batch mode (expect CUDA acceleration)
		opt = resolve_opt(conv, Packing, DW, LUT, Last=Last)

		if Packing == 'Filter_Packing':
			PK = 'FP'
		elif Packing == 'Kernel_Packing':
			PK = 'KP'
		else:
			raise NotImplementedError(f'{Packing} is not implemented!')
		path = self.pred_path / f'{PK}_{DW}_{LUT}'
		opt.Load_Model(path)

		possible_features, possible_configs = self.get_feature_vectors(opt, simd_aval, pe_aval, Last=Last)
		pred_res_raw = opt.predict_batch(possible_features)   # [dsp, lut, bram, wns]
		round_steps = np.array([self.DSP_step, self.LUT_step, self.BRAM_step])
		pred_res_round = np.round(pred_res_raw[:, :3] / round_steps)
		pred_res = np.concatenate((pred_res_round, pred_res_raw[:, 3].reshape(-1,1)), axis=1)

		Lats = np.array([opt.dsp_operations()]) / (possible_configs[:, 0] * possible_configs[:, 1] * possible_configs[:, 2]).reshape(-1, 1)
		Pred_Lookup = np.concatenate((Lats, pred_res, possible_configs), axis=1)    # [Lat, dsp, lut, bram, wns, simd, kpf, pe, actp]

		thread_list = []
		for thread_idx in range(0, self.thread_num):
			Pred_Lookup_cp = Pred_Lookup.copy()           # do not different thread read the same area of memory, better performance?
			opt_cp = resolve_opt(conv, Packing, DW, LUT, Last=Last)
			thrd = Thread(target=self.Fuse_Search_Loops, args=(opt_cp, layer_idx, Pred_Lookup_cp, thread_idx, Packing, DW, LUT), daemon=True)
			thread_list.append(thrd)
			thrd.start()
	
		for thrd in thread_list:
				thrd.join()


	def DP_Search(self):
		t1 = time.time()
		for layer_idx, conv_org in enumerate(self.model_param):
			# check if last layer
			Last = (layer_idx == (len(self.model_param) - 1))

			# resolve DSP packing for current layer
			conv, Packing, DW, simd_aval, pe_aval = resolve_packing(conv_org, self.DSP_Explorer, Last=Last)
			conv.cycle = self.cycle

			# try to use LUT-based opt first
			if (conv.abit <= 4) and (conv.wbit <= 4):
				print(f'Try to implement LUT-based operators in layer {layer_idx}.')
				self.Search_Layer(layer_idx, conv, Packing, simd_aval, pe_aval, DW, LUT=True, Last=Last)
				print(f'Layer {layer_idx} finished (LUT)!')

			print(f'Try to implement DSP-based operators in layer {layer_idx}.')
			self.Search_Layer(layer_idx, conv, Packing, simd_aval, pe_aval, DW, LUT=False, Last=Last)
			print(f'Layer {layer_idx} finished (DSP)!')

		t2 = time.time()
		print(f'Finished DP-search! Spent {t2 - t1} seconds in total!')

	def DP_Results(self, Save=False):
		# search best node
		dsp_const = 0
		lut_const = 0
		bram_const = 0
		best_Lat = self.Init_Lat

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
		Packing_list = best_Node.Packing
		DW_list = best_Node.DW
		LUT_list = best_Node.LUT

		if len(SIMD_list) == 0 or Save:
			print_dict = {}
			for l in range(self.n_layers):
				for k in range(0, self.BRAM_max_step + 1):
					for j in range(0, self.LUT_max_step + 1):
						for i in range(0, self.DSP_max_step + 1):
							cur_Node = self.DPT[l][k][j][i]
							print_dict[f'{l}_{k}_{j}_{i}'] = {'Lat': cur_Node.Lat, f'SIMD': cur_Node.SIMD, 'PE': cur_Node.PE, 'ACTP': cur_Node.ACTP,
															  'KPF': cur_Node.KPF, 'Packing': cur_Node.Packing, 'DW': cur_Node.DW, 'LUT': cur_Node.LUT}

			with open('./DP_debug/DP_table.json', 'w', encoding='utf-8') as f:
				json.dump(print_dict, f, indent=4)
		
		if len(SIMD_list) == 0:
			print(f'Failed to find a solution! Exit!')	
			exit(0)

		print(bram_const, lut_const, dsp_const)

		return best_Lat, Packing_list, DW_list, SIMD_list, PE_list, ACTP_list, KPF_list, LUT_list

