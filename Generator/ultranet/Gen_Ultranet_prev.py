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
import mymodel
from utils.view_pt import select_weight_file
from quant_dorefa import activation_quantize_fn
from quant_module import HWGQ, QuantConv2d, ImageInputQ
from Front_Back import get_front, get_back
from Opt_Templates import Gen_Opt_Templates
from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates

Opt_list = [1, 2, 3, 4, 5, 6, 7]
Opt_list_KRowP = []
Opt_list_KP = []
Opt_list_KP_KRowP = [0]


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
        'in_pe': 'IN_PE'
    }
    content = f'''/********************************************************************************
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

'''
    for n in range(len(model_param)):
        model_param[n].in_pe = model_param[n-1].pe if n else 3

        if n == 1:
            model_param[n].in_pe = model_param[n-1].pe * model_param[n-1].kp                               # special, attention!!!

    for n, conv_param in enumerate(model_param):
        content += f'// conv_{n}\n'
        for k, v in name_mapping.items():
            if hasattr(conv_param, k): # e.g. conv_last has no incbit
                content += f'#define CONV_{n}_{v} {getattr(conv_param, k)}\n'
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

def reorder_weight(model_param, layers_simd, layers_pe, layers_actp, layers_kp, layer_np, layer_gb):
    '''reorder_weight(model_param)
    Reorder array for hlscode.
    '''

    for conv, simd, pe, actp, kp, np, gb in zip(model_param, layers_simd, layers_pe, layers_actp, layers_kp, layer_np, layer_gb):
        print(f'Reorder conv_{conv.n}, w {conv.w.shape}', end='')
        conv.simd = simd
        conv.pe = pe
        conv.actp = actp
        conv.kp = kp 
        conv.np = np
        conv.gb = gb

        # process batchnorm
        if conv.inc is not None:
            conv.inc = conv.inc.reshape(conv.och//conv.actp, conv.actp).T
        if hasattr(conv, 'bias') and conv.bias is not None:
            conv.bias = conv.bias.reshape(conv.och//conv.actp, conv.actp).T

        # process conv weight
        if conv.k == 1:
            w = conv.w    # [och, ich, kr, kc]
            g_ich = w.shape[1]
            assert conv.och%(conv.pe) == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}"
            assert g_ich%simd == 0, f"conv_{conv.n}, ich {g_ich}, simd {conv.simd}"
            w = w.reshape(conv.och//(conv.pe), conv.pe, g_ich//simd, simd) # [och / pe, pe, ich / simd, simd]
            w = w.transpose(1,0,2,3)  #[pe, och / pe, ich / simd, simd]
            w = w.reshape(conv.pe, -1, g_ich//simd, simd) # [pe, och / pe, ich / simd, simd]
            w = w.reshape(conv.pe, -1, simd)   # hls format [pe, och/pe * ich/simd, simd]
        elif conv.n in Opt_list:
            w = conv.w    # [och, ich, kr, kc]
            assert conv.och%conv.pe == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}"
            assert conv.k*conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

            w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
            w = w.reshape(conv.och//conv.pe, conv.pe, conv.k, conv.k*conv.ich//simd, simd)  # [och/pe, pe, kc, kr*ich/simd, simd]
            w = w.transpose(1, 0, 3, 2, 4) # [pe, och/pe, kr*ich/simd, kc, simd]
            w = w[:, :, :, ::-1, :]
            w = w.reshape(conv.pe, -1, conv.k*simd)  # [pe, och/pe*kr*ich/simd, kc*simd]
        elif conv.n in Opt_list_KRowP:
            w = conv.w    # [och, ich, kr, kc]
            assert conv.och%conv.pe == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}"
            assert conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

            w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
            w = w.reshape(conv.och//conv.pe, conv.pe, conv.k, conv.k, conv.ich//simd, simd)  # [och/pe, pe, kc, kr, ich/simd, simd]
            w = w.transpose(1, 0, 4, 2, 3, 5) # [pe, och/pe, ich/simd, kc, kr, simd]
            w = w[:, :, :, ::-1, :, :]
            w = w.reshape(conv.pe, -1, conv.k*conv.k*simd)  # [pe, och/pe*ich/simd, kc*kr*simd]
        elif conv.n in Opt_list_KP:
            w = conv.w    # [och, ich, kr, kc]
            assert conv.och%(conv.pe*conv.kp) == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}, kp {conv.kp}"
            assert conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

            w = w.reshape(conv.och//(conv.kp*conv.pe), conv.pe, conv.kp, conv.ich//conv.simd, conv.simd, conv.k, conv.k)   # [och/(kp*pe), pe, kp, ich/simd, simd, kr, kc]
            w = w.transpose(1, 0, 5, 3, 6, 4, 2)            # [pe, och/(kp*pe), kr, ich/simd, kc, simd, kp]
            w = w[:, :, :, :, ::-1, :, :]
            w = w.reshape(conv.pe, -1, conv.simd*conv.kp)   # [pe, och/(kp*pe) * kr * ich/simd * kc, simd * kp]

        elif conv.n in Opt_list_KP_KRowP:
            w = conv.w    # [och, ich, kr, kc]
            assert conv.och%(conv.pe*conv.kp) == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}, kp {conv.kp}"
            assert conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

            w = w.reshape(conv.och//(conv.kp*conv.pe), conv.pe, conv.kp, conv.ich//conv.simd, conv.simd, conv.k, conv.k)   # [och/(kp*pe), pe, kp, ich/simd, simd, kr, kc]
            w = w.transpose(1, 0, 3, 6, 5, 4, 2)            # [pe, och/(kp*pe), ich/simd, kc, kr, simd, kp]
            w = w[:, :, :, ::-1, :, :, :]
            w = w.reshape(conv.pe, -1, conv.k*conv.simd*conv.kp)   # [pe, och/(kp*pe) * ich/simd * kc, kr * simd * kp]
        else:
            w = conv.w    # [och, ich, kr, kc]
            assert conv.och%conv.pe == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}"
            assert conv.k*conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

            if conv.n == 0:
                w = w.transpose(0, 2, 3, 1) # [och, kc, kr, ich]
            else:
                w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]

            w = w.reshape(conv.och//conv.pe, conv.pe, conv.k, conv.k*conv.ich//simd, simd)
            w = w.transpose(1,2,0,3,4) # [pe, k, och/pe, k*ich/simd, simd]
            w = w.reshape(conv.pe, conv.k, -1, simd) # hls format [pe, k, och/pe*k*ich/simd, simd]

        print(' ->', w.shape)
        conv.w = w

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

def write_hls_weights(model_param, path):
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

    for conv in model_param:
        n = conv.n
        print(f"Write conv_{n} weight, pe {conv.pe}, simd {conv.simd}, wbit {conv.wbit}")
        print(f"// layer: {n}, PE: {conv.pe}, SIMD: {conv.simd}, wbit: {conv.wbit}", file=f)

        # print conv weight,  merge [SIMD] value into one ap_uint
        if conv.k == 1:
            print(f"const ap_uint<{conv.wbit * conv.simd}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        elif conv.n in Opt_list:
            print(f"const ap_uint<{conv.k * conv.wbit * conv.simd}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        elif conv.n in Opt_list_KRowP:
            print(f"const ap_uint<{conv.k * conv.wbit * conv.k * conv.simd}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        elif conv.n in Opt_list_KP:
            print(f"const ap_uint<{conv.wbit * conv.kp * conv.simd}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        elif conv.n in Opt_list_KP_KRowP:
            print(f"const ap_uint<{conv.wbit * conv.kp * conv.simd * conv.k}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        else:
            print(f"const ap_uint<{conv.wbit * conv.simd}> conv_{n}_w[{conv.pe}][{conv.k}][{conv.w.shape[2]}]=", file=f)
            
        hex_str = lambda x: '"' + hex(x) + '"'
        def pack1d_str(arr): # x: 1d-array
            x = 0
            for v in arr[::-1]: # [!] reverse simd pack, it is related to hls implemention
                v = int(v) # use python bignumber, not np.int
                assert -1<<conv.wbit-1 <= v < 1<<conv.wbit-1, f'got v={v} while wbit={conv.wbit}'
                x=(x<<conv.wbit) + (v&(2**conv.wbit-1))
            return hex_str(x)
        print_ndarray_recursion(conv.w, pack1d_str, f, stop=1)
        print(';', file=f)

        # print inc, bias
        if conv.inc is not None:
            print(f"const ap_int<{conv.incbit}> conv_{n}_inc[{conv.actp}][{conv.och//conv.actp}]=", file=f)
            print_ndarray_recursion(conv.inc, hex_str, f)
            print(';', file=f)
        if conv.bias is not None:
            print(f"const ap_int<{conv.biasbit}> conv_{n}_bias[{conv.actp}][{conv.och//conv.actp}]=", file=f)
            print_ndarray_recursion(conv.bias, hex_str, f)
            print(';', file=f)
    
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

def write_hls_accel(model_param, path):

    content = get_front()

    for conv in model_param:
        if conv.n <= 7:
            if conv.n > 0:
                conv.kpf = 1
                conv.max_pool = True if conv.n < 4 else False
                Opt_Temp = FP_Opt_Templates(conv)
                content += Opt_Temp.gen_operator()
            else:
                conv.kpf = conv.k
                conv.max_pool = True if conv.n < 4 else False
                Opt_Temp = KP_Opt_Templates(conv)
                content += Opt_Temp.gen_operator()

    
    content += get_back()

    with open(path + 'accelerator.cpp', 'w') as f:
        print(content, file=f)



if __name__=='__main__':
    model_name = 'UltraNet_ismart'
    weight = 'ultra_4w4a'
    config_simd_pe = '4w4a_8fl'

    array_config = np.loadtxt('hls/'+config_simd_pe+'.txt', dtype=int, skiprows=1)
    dir_output = 'hls/' + weight + '/'
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
    
    # in_pe = np.concatenate((np.array([3]), array_config[:-1, 5]), axis=0)
    reorder_weight(model_param, array_config[:,0], array_config[:,1], array_config[:,2], array_config[:,3], array_config[:,4], array_config[:,5])
    write_hls_config(model_param, dir_output)
    write_hls_weights(model_param, dir_output)
    write_hls_accel(model_param, dir_output)
