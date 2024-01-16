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
sys.path.append('../../DSP_explorer/')
sys.path.append('../../Opt_Definition')
sys.path.append('../../Opt_Definition/ConvOpt_KP')
sys.path.append('../../Opt_Definition/ConvOpt_KP/predictors')
sys.path.append('../../Opt_Definition/ConvOpt_KP_LUT')
sys.path.append('../../Opt_Definition/ConvOpt_FP')
sys.path.append('../../Opt_Definition/ConvOpt_FP/predictors')
sys.path.append('../../Opt_Definition/ConvOpt_FP_LUT')
sys.path.append('../../Opt_Definition/ConvOpt_FP_DW')
sys.path.append('../../Opt_Definition/ConvOpt_FP_DW_LUT')
sys.path.append('../../Opt_Definition/ConvOpt_1x1')
import mymodel
from utils.view_pt import select_weight_file
from quant_dorefa import activation_quantize_fn
from anypacking.quant_module import HWGQ, QuantConv2d, ImageInputQ
from Accelerator_Template import write_hls_config, write_hls_weights, write_hls_accel
from Opt_Templates import Gen_Opt_Templates
from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_LUT import FP_LUT_Opt_Templates
from ConvOpt_FP_DW import FP_DW_Opt_Templates
from ConvOpt_FP_DW_LUT import FP_DW_LUT_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_KP_LUT import KP_LUT_Opt_Templates
from ConvOpt_1x1 import Conv1x1_Opt_Templates

from dsp_eff_search import DSP_Config_Search

class ConvParam: ...

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
            conv_cur.groups = sub_module.groups if hasattr(sub_module, 'groups') else 1
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
            print(', ich {ich}, och {och}, irow {irow}, icol {icol}, ksp {k}{s}{p}, wbit {wbit}, wstep {wstep}, g {groups}'.format(**vars(conv_cur)))

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
    if hasattr(conv_last, 'convbias'):
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

def adjust_weight(model_param):
    # special_wa_bit = ((4,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(7,2),(7,3)) 
    special_wa_bit = []
    # These packing can't quantize to -2**(wbit-1)
    for conv in model_param:
        if (conv.wbit, conv.abit) in special_wa_bit:
            print(f'Adjust conv_{conv.n} wbit={conv.wbit}')
            conv.w = np.maximum(conv.w, -2**(conv.wbit-1)+1)

def gen_opts(model_param, array_config):
    DSP_Explorer = DSP_Config_Search(27, 18, 8)
    model_opt = []
    for idx, (conv, extra_para) in enumerate(zip(model_param, array_config)):
        conv.simd = extra_para[0]
        conv.pe = extra_para[1]
        conv.actp = extra_para[2]
        conv.kpf = extra_para[3]
        LUT = bool(extra_para[4])
        conv.max_pool = conv.max_pool

        if conv.w.shape[1] == 1 and conv.ich != 1:   # depth-width
            acc_num = conv.k
            DW = True
        else:
            acc_num = conv.k * conv.ich
            DW = False

        if idx == (len(model_param) - 1):
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

        packing_type = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Packing_Type']

        if idx == (len(model_param) - 1):
            conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']
            model_opt.append(Conv1x1_Opt_Templates(conv))
        elif packing_type == 'Filter_Packing':
            if DW:
                opt = FP_DW_LUT_Opt_Templates(conv) if LUT else FP_DW_Opt_Templates(conv)
                model_opt.append(opt)
            else:
                opt = FP_LUT_Opt_Templates(conv) if LUT else FP_Opt_Templates(conv)
                model_opt.append(opt)
        elif packing_type == 'Kernel_Packing':
            conv.pack_flag = DSP_Config_Lookup[f'w{conv.wbit}a{conv.abit}']['Pack_Flag']

            if DW:
                raise TypeError(f"Kernel_Packing operator cannot be used for Depth-wise Convolution!")
            else:
                opt = KP_LUT_Opt_Templates(conv) if LUT else KP_Opt_Templates(conv)
                model_opt.append(opt)
        else:
            raise TypeError(f"Operator {str(opt_type)} is not defined!")

    # for conv, extra_para in zip(model_param, array_config[:, :10]):
    #     conv.simd = extra_para[0]
    #     conv.pe = extra_para[1]
    #     conv.actp = extra_para[2]
    #     conv.kp = extra_para[3] 
    #     conv.np = extra_para[4]
    #     conv.gb = extra_para[5]  
    #     conv.kpf = extra_para[6]
    #     conv.max_pool = extra_para[7]
    #     conv.w_sep = extra_para[8]
    #     conv.a_sep = extra_para[9]

    # model_opt = []
    # for conv, opt_type in zip(model_param, array_config[:, 10]):
    #     pack_flag = False        # to be modified


    #     if opt_type == 0:
    #         conv.pack_flag = pack_flag        # to be modified
    #         model_opt.append(KP_Opt_Templates(conv))
    #     elif opt_type == 1:
    #         conv.pack_flag = pack_flag        # to be modified
    #         model_opt.append(KP_LUT_Opt_Templates(conv))
    #     elif opt_type == 2:
    #         model_opt.append(FP_Opt_Templates(conv))
    #     elif opt_type == 3:
    #         model_opt.append(FP_LUT_Opt_Templates(conv))
    #     elif opt_type == 4:
    #         conv.pack_flag = pack_flag        # to be modified
    #         model_opt.append(Conv1x1_Opt_Templates(conv))
    #     elif opt_type == 5:
    #         model_opt.append(FP_DW_Opt_Templates(conv))
    #     else:
    #         raise ValueError(f"Operator {str(opt_type)} is not defined!")

    return model_opt


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name for the NN accelerator')
    parser.add_argument('-w', '--weight', default='fixed', help='.pt file name in ./weights/')
    parser.add_argument('-m', '--model', default='UltraNet_FixQ', help = 'model class name in mymodel.py')  # UltraNet_FixQ  UltraNet_ismart  SkyNet_FixQ
    parser.add_argument('-c', '--config-simd-pe', default='config_simd_pe', help = '.txt file in ./hls/')
    parser.add_argument('-dp', '--debug-path', default='./debug_path/', help = 'path for debug outpt')
    parser.add_argument('-ip', '--input-path', default='./test/0.bin', help = '.bin file for testing')
    parser.add_argument('--GenTB', default=True)
    opt = parser.parse_args()
    model_name = opt.model
    weight = opt.weight
    config_simd_pe = opt.config_simd_pe
    name = str(opt.name)

    array_config = np.loadtxt('hls/'+config_simd_pe+'.txt', dtype=int, skiprows=1)
    dir_output = 'hls/' + weight + '_' + config_simd_pe + '/'
    if not os.path.exists(dir_output): os.makedirs(dir_output)

    # load model and state_dict
    ptfile:Dict = torch.load('weights/' + weight + '.pt', map_location='cpu')
    model = getattr(mymodel, model_name)(**ptfile.setdefault('model_params', {}))
    model.load_state_dict(ptfile['model'])

    # processs
    model_param = extract_model([1, 160, 320])
    adjust_weight(model_param)
    process_batchnorm(model_param) # get bn param before write hls config
    model_opt = gen_opts(model_param, array_config)
    torch.save(model_param, dir_output + 'model_param.pkl')
    
    write_hls_config(model_opt, dir_output)
    write_hls_weights(model_opt, dir_output)
    write_hls_accel(model_opt, dir_output, net_name=name, GenTB=opt.GenTB, debug_path=opt.debug_path, input_path=opt.input_path)
