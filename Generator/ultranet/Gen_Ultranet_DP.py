import argparse
import time
from typing import Dict, List
import torch
import numpy as np
import sys
import os
import math

import sys
sys.path.append('..')
sys.path.append('../operators')
sys.path.append('../operators/ConvOpt_KP')
sys.path.append('../operators/ConvOpt_KP/predictors')
sys.path.append('../operators/ConvOpt_FP')
sys.path.append('../operators/ConvOpt_FP/predictors')
sys.path.append('../operators/ConvOpt_FP_Sep')
sys.path.append('../operators/ConvOpt_1x1')
import mymodel
from utils.view_pt import select_weight_file
from quant_dorefa import activation_quantize_fn
from quant_module import HWGQ, QuantConv2d, ImageInputQ

from Front_Back import get_front, get_back
from Opt_Templates import Gen_Opt_Templates
from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_Sep import FP_Sep_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_1x1 import Conv1x1_Opt_Templates

from Pipeline_DP import Pipeline_Allocation
from dsp_eff_search import DSP_Config_Search


class ConvParam: ...

def write_hls_config(model_param, path):
    name_mapping = {
        'k': 'K',
        'ich': 'IN_CH',
        'irow': 'IN_H',
        'icol': 'IN_W',
        'och': 'OUT_CH',
        'abit': 'IN_BIT',
        'wbit': 'W_BIT',
        'incbit': 'INC_BIT',
        'biasbit': 'BIAS_BIT',
        'obit': 'OUT_BIT',
        'simd': 'SIMD',
        'pe': 'PE',
        'lshift': 'L_SHIFT',
        'actp': 'ACTP',
        'kp': 'Kp',
        'np': 'Np',
        'gb': 'GUARD_BIT',
        'in_pe': 'IN_PE',
        'kpf': 'KPF'
    }
    content = f'''/********************************************************************************
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

'''
    for n, conv in enumerate(model_param):
        content += f'// conv_{n}\n'
        for k, v in name_mapping.items():
            if hasattr(conv, k): # e.g. conv_last has no incbit
                content += f'#define CONV_{n}_{v} {getattr(conv, k)}\n'
        content += '\n'
    content += '#endif'

    with open(path + 'config.h', 'w') as f:
        print(content, file=f)

def extract_model(in_shape):
    model_param: List[ConvParam] = []
    feature_map_shape = in_shape
    conv_cnt = 0
    conv_cur = None
    for sub_module in model.modules():
        # expect [QAct] -> [Pooling] -> Conv -> [BN] -> [Pooling], state machine mode
        if isinstance(sub_module, HWGQ) or isinstance(sub_module, ImageInputQ) or isinstance(sub_module, activation_quantize_fn):
            print('  Detected ActQ Layer', end='')
            if conv_cur is None: conv_cur = ConvParam()
            if isinstance(sub_module, HWGQ) or isinstance(sub_module, ImageInputQ):
                conv_cur.abit = sub_module.bit
                conv_cur.astep = sub_module.step
            else:
                conv_cur.abit = sub_module.a_bit
                conv_cur.astep = 1/2**conv_cur.abit
            
            conv_cur.actq_class = type(sub_module).__name__
            print(f', abit {conv_cur.abit}, astep {conv_cur.astep}, class {conv_cur.actq_class}')

            if conv_cnt: # previous.obit = cur.abit
                model_param[conv_cnt-1].obit = conv_cur.abit
                model_param[conv_cnt-1].ostep = conv_cur.astep
            
        elif isinstance(sub_module, torch.nn.Conv2d):
            if conv_cur is None: conv_cur = ConvParam()
            conv_cur.n = conv_cnt
            print('Extract conv_%d'%conv_cnt, end='')

            conv_cur.k = sub_module.kernel_size[0]
            conv_cur.s = sub_module.stride[0]
            conv_cur.p = sub_module.padding[0]
            conv_cur.ich = sub_module.in_channels
            conv_cur.och = sub_module.out_channels
            conv_cur.irow = feature_map_shape[1]
            conv_cur.icol = feature_map_shape[2]
            
            feature_map_shape[0] = sub_module.out_channels
            feature_map_shape[1] = (feature_map_shape[1] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            feature_map_shape[2] = (feature_map_shape[2] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            conv_cur.orow = feature_map_shape[1]
            conv_cur.ocol = feature_map_shape[2]

            if sub_module.bias is not None:
                conv_cur.convbias = sub_module.bias.detach().numpy()
                print(', +bias', end='')

            if isinstance(sub_module, QuantConv2d): # New quant
                conv_cur.wbit = sub_module.bit
                conv_cur.w, conv_cur.wstep = sub_module.export_quant() # wstep is not QuantConv2d.step becuause of alpha

            elif type(sub_module).__name__ == 'Conv2d_Q': # Old dorefa quant
                conv_cur.wbit = sub_module.w_bit
                conv_cur.wstep = 1/2**(conv_cur.wbit-1)
                weight = np.tanh(sub_module.weight.detach().numpy())
                weight = weight / np.max(np.abs(weight))
                n = 2**(conv_cur.wbit-1)
                weight_q = weight * n
                weight_q = np.clip(np.round(weight_q),-n, n-1)
                weight_q = weight_q.astype(np.int32)
                conv_cur.w = weight_q
            else:
                raise NotImplementedError(sub_module)
            print(', ich {ich}, och {och}, irow {irow}, icol {icol}, ksp {k}{s}{p}, wbit {wbit}, wstep {wstep}'.format(**vars(conv_cur)))

            conv_cur.max_pool = False
            
            model_param.append(conv_cur)
            conv_cur = None
            conv_cnt += 1
        
        elif isinstance(sub_module, torch.nn.BatchNorm2d):
            print('  Detected BatchNorm2d')
            gamma = sub_module.weight
            beta = sub_module.bias
            mean = sub_module.running_mean
            var = sub_module.running_var
            eps = sub_module.eps
            
            model_param[-1].bn_w = (gamma / (torch.sqrt(var + eps))).detach().numpy()
            model_param[-1].bn_b = (beta - (mean / (torch.sqrt(var + eps)) * gamma)).detach().numpy()

        elif isinstance(sub_module, torch.nn.MaxPool2d):
            feature_map_shape[1] = feature_map_shape[1] // sub_module.kernel_size
            feature_map_shape[2] = feature_map_shape[2] // sub_module.kernel_size
            model_param[-1].max_pool = True
    
    if not hasattr(model_param[0], 'abit'): # train code rescaled [0,255] to [0,1) by /256 default
        model_param[0].abit = 8
    if not hasattr(model_param[0], 'astep'):
        model_param[0].astep = 1/256

    return model_param

def process_batchnorm(model_param):
    '''process_batchnorm(model_param)
    Merge wstep, astep, ostep scale into batchnorm, then quantize. 

    Method:
    Define MAC = Conv(w, a), out = MAC*BN_w + BN_b,
    wq = w/wstep, aq = a/astep, MACq = MAC/MACstep, outq = out/ostep.

    outq = (MAC*BN_w + BN_b) / ostep
         = MACq * (MACstep/ostep)*BN_w + BN_b/ostep
         = MACq *     inc_raw          + bias_raw
    next layer activation a' = ActQ(out), i.e. a'q = clip(round(outq))

    Quantiaztion of inc_raw & bias_raw: 
    outq_real = round((MACq*round(inc_raw*scale) + round(bias_raw*scale)) / scale)         ; where scale=2**T
              = (MACq*round(inc_raw*scale) + round(bias_raw*scale) + 0.5 * scale) // scale ; div floor
              = (MACq*        inc          +         bias          +  2**(T-1)  ) >> T     ; [!] the 2**(T-1) bias is done by hls code

    Params:
    T = (wbit-1)+abit+lshift  # This comes from dorefa quant, not optimal
    MBIT = wbit+abit+ceil(log2(sum_number))
    incbit = len(bit(inc)); biasbit = len(bit(bias))
    larger lshift is better, but MBIT+incbit<48
    '''
    lshift = 8

    for conv in model_param[:-1]:
        print(f'Process bn_{conv.n}, shape {conv.bn_w.shape},', end = ' ')

        # Merge step to BN
        conv.lshift = lshift
        MACstep = conv.wstep * conv.astep
        ostep = conv.ostep
        inc_raw = conv.bn_w * MACstep / ostep
        bias_raw = conv.bn_b / ostep
        conv.inc_raw = inc_raw
        conv.bias_raw = bias_raw

        # Quantization
        T = lshift+conv.wbit+conv.abit-1
        conv.inc = np.round(inc_raw * 2**T).astype(np.int64)
        conv.bias = np.round(bias_raw * 2**T).astype(np.int64)
        conv.lshift_T = T
        # Get bitlength
        bitlength = lambda x: 1 + int(np.abs(x).max()).bit_length()
        conv.incbit = bitlength(conv.inc)
        conv.biasbit = bitlength(conv.bias)
        print(f'incbit {conv.incbit}, biasbit {conv.biasbit}, lshift_T {conv.lshift_T}')
    
    conv_last = model_param[-1] # process lastbias
    conv_last.inc = None
    conv_last.div = 1/(conv_last.wstep * conv_last.astep)
    conv_last.bias = np.round(conv_last.convbias * conv_last.div).astype(np.int64)
    conv_last.bias_raw = conv_last.convbias * conv_last.div
    conv_last.biasbit = bitlength(conv_last.bias)
    print(f'conv_last biasbit {conv_last.biasbit}, div {conv_last.div}')


def print_ndarray_recursion(arr, str_func=str, file=sys.stdout, stop=0):
    if not hasattr(arr, '__iter__') or len(arr.shape) == stop:
        print(str_func(arr), file=file, end='')
        return
    ends = '' if (len(arr.shape)==stop+1) else '\n'
    print('{', file=file, end='')
    for i, item in enumerate(arr):
        print_ndarray_recursion(item, str_func, file, stop)
        if i!=len(arr)-1: print(',', file=file, end=ends)
    print(ends+'}', file=file, end='')

def write_hls_weights(model_opt, path):
    '''write_hls_weights(model_param, path)
    Write hls weights+inc+bias array code according to numpy shape.
    '''
    f = open(path + 'weights.hpp', 'w')

    print(f'''/********************************************************************************
********************************************************************************/

#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_
#include <ap_int.h>
''', file=f)

    for opt in model_opt:
        print(f"Write conv_{opt.conv.n} weight, pe {opt.conv.pe}, simd {opt.conv.simd}, wbit {opt.conv.wbit}")
        content = opt.write_weights()
        print(content, file=f, end='')
    
    print('#endif', file=f)
    f.close()

def adjust_weight(model_param):
    # special_wa_bit = ((4,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(7,2),(7,3)) 
    special_wa_bit = []
    # These packing can't quantize to -2**(wbit-1)
    for conv in model_param:
        if (conv.wbit, conv.abit) in special_wa_bit:
            print(f'Adjust conv_{conv.n} wbit={conv.wbit}')
            conv.w = np.maximum(conv.w, -2**(conv.wbit-1)+1)

def write_hls_accel(model_opt, path):

    content = get_front()

    for opt in model_opt:
        content += opt.gen_operator()
    
    content += get_back()

    with open(path + 'accelerator.cpp', 'w') as f:
        print(content, file=f)


# def gen_opts(model_param, array_config):
#     for conv, extra_para in zip(model_param, array_config[:, :8]):
#         conv.kp = extra_para[0] 
#         conv.np = extra_para[1]
#         conv.gb = extra_para[2]  
#         # conv.max_pool = extra_para[3]

#     model_opt = []
#     for conv, opt_type in zip(model_param, array_config[:, 4]):
#         if opt_type == 0:
#             conv.pack_flag = True        # to be modified
#             model_opt.append(KP_Opt_Templates(conv))
#         elif opt_type == 1:
#             model_opt.append(FP_Opt_Templates(conv))
#         elif opt_type == 2:
#             model_opt.append(Conv1x1_Opt_Templates(conv))
#         else:
#             raise ValueError(f"Operator {str(opt_type)} is not defined!")

#     return model_opt

def gen_opts(model_param, array_config):
    DSP_Explorer = DSP_Config_Search(27, 18, 8)
    model_opt = []
    for idx, conv in enumerate(model_param):
        DSP_Config_Lookup = DSP_Explorer.Packing_Exploration(K=conv.k, overlap=1, wbmin=2, wbmax=8, abmin=2, abmax=8, Filter_Packing_EN=True, Kernel_Packing_EN=True)
        conv.kp = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['kp']
        conv.np = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['np']
        conv.gb = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['gb']
        # conv.kp = extra_para[0] 
        # conv.np = extra_para[1]
        # conv.gb = extra_para[2]  
        # conv.max_pool = extra_para[3]

        packing_type = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Packing_Type']
        print(f'Layer {idx} type: {packing_type}')
        if packing_type == 'Filter_Packing':
            model_opt.append(FP_Opt_Templates(conv))
        elif packing_type == 'Filter_Packing_Sep':
            conv.Sep_Flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Sep_Flag']
            model_opt.append(FP_Sep_Opt_Templates(conv))
        elif packing_type == 'Kernel_Packing':
            conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']
            model_opt.append(KP_Opt_Templates(conv))
        elif packing_type == 'Kernel_Packing_Sep':
            print('Have not defined Kernel_Packing_Sep yet!')
            exit(0)
        elif idx == (len(model_param) - 1):
            model_opt.append(Conv1x1_Opt_Templates(conv))
        else:
            raise ValueError(f"Operator {str(opt_type)} is not defined!")

    # model_opt = []
    # for conv, opt_type in zip(model_param, array_config[:, 4]):
    #     # if opt_type == 0:
    #     #     conv.pack_flag = True        # to be modified
    #     #     model_opt.append(KP_Opt_Templates(conv))
    #     # elif opt_type == 1:
    #     #     model_opt.append(FP_Opt_Templates(conv))
    #     # elif opt_type == 2:
    #     #     model_opt.append(Conv1x1_Opt_Templates(conv))
    #     # else:
    #     #     raise ValueError(f"Operator {str(opt_type)} is not defined!")

    return model_opt


def set_parallelism(model_opt, DSP_max, LUT_max, DSP_step, LUT_step):
    pipel_alloc = Pipeline_Allocation(model_opt[:-1], DSP_max=DSP_max, LUT_max=LUT_max, DSP_step=DSP_step, LUT_step=LUT_step)
    print('Begin searching parallelism!')
    Lat, SIMD_list, PE_list, ACTP_list, KPF_list = pipel_alloc.DP_Search()

    print(f'Finished searching! Overall latency is {Lat}')
    print('SIMD, PE, ACTP, KPF, Latency:')
    for i in range(len(model_opt[:-1])):
        model_opt[i].conv.simd = SIMD_list[i]
        model_opt[i].conv.pe = PE_list[i]
        model_opt[i].conv.actp = ACTP_list[i]
        model_opt[i].conv.kpf = KPF_list[i]

        cur_Lat = model_opt[i].dsp_operations() / (SIMD_list[i] * PE_list[i] * KPF_list[i])
        print(f'{SIMD_list[i]}, {PE_list[i]}, {ACTP_list[i]}, {KPF_list[i]}, {cur_Lat}')

    model_opt[-1].conv.simd = 4
    model_opt[-1].conv.pe = 2
    model_opt[-1].conv.actp = 2
    model_opt[-1].conv.kpf = 1
    model_opt[-1].conv.pack_flag = 0

    for n in range(len(model_opt)):
        if n == 0:
            opf = 3
        elif array_config[n-1, 4] == 0:
            opf = model_opt[n-1].conv.pe * model_opt[n-1].conv.kp
        else:
            opf = model_opt[n-1].conv.pe

        model_opt[n].conv.in_pe = opf

    return model_opt


if __name__=='__main__':
    model_name = 'UltraNet_ismart'
    weight = 'ultra_4w4a'
    config_simd_pe = '4w4a_8fl_dp_testflow'

    array_config = np.loadtxt('hls/'+config_simd_pe+'.txt', dtype=int, skiprows=1)
    dir_output = 'hls/' + config_simd_pe + '/'
    if not os.path.exists(dir_output): os.makedirs(dir_output)

    # load model and state_dict
    ptfile:Dict = torch.load('weights/' + weight + '.pt', map_location='cpu')
    model = getattr(mymodel, model_name)(**ptfile.setdefault('model_params', {}))
    model.load_state_dict(ptfile['model'])

    # processs
    model_param = extract_model([1, 160, 320])
    adjust_weight(model_param)
    process_batchnorm(model_param) # get bn param before write hls config
    torch.save(model_param, dir_output + 'model_param.pkl')

    model_opt = gen_opts(model_param, array_config)
    t1 = time.time()
    model_opt = set_parallelism(model_opt, DSP_max=330, LUT_max=56400, DSP_step=5, LUT_step=10000)
    t2 = time.time()
    print(f'Parallelism factor search spent {(t2 - t1) / 60} minutes in total.')
    
    write_hls_config(model_param, dir_output)
    write_hls_weights(model_opt, dir_output)
    write_hls_accel(model_opt, dir_output)
