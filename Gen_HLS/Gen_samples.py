import argparse
import time
from typing import Dict, List
import torch
import numpy as np
# import sys
# import os
import math
import json
from string import Template
import pathlib

import sys
sys.path.append('..')



def find_CASCADE(gb, kp, np, simd):
    cascade = 1
    upper = min((2**gb) // min(kp, np), simd)
    for factor in range(1, int(upper) + 1):
        if upper%factor == 0:
            cascade = factor

    return cascade


para_temp = Template('''//--------------------Conv ${No}: Parameters--------------------
const unsigned CONV_${No}_M_BIT = CONV_${No}_IN_BIT + CONV_${No}_W_BIT + ${EX_M_BIT};
const unsigned CONV_${No}_SIMD_BIT = ${SIMD_BIT};
const unsigned CONV_${No}_CASCADE = ${CASCADE};
const unsigned CONV_${No}_ROW_LEN = (CONV_${No}_IN_W + CONV_${No}_K - 1 - 1) / CONV_${No}_Np + 1;
const unsigned CONV_${No}_adW_BIT = 1;
    ''')

rebuffer_temp_1 = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_SIMD_INPE<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH, CONV_${No}_OUT_CH / CONV_${No}_PE,
                         CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No_pre}_layer_out, conv_${No}_padding_out, reps);
    ''')

rebuffer_temp_2 = Template('''//--------------------Conv ${No}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_SIMD * CONV_${No}_IN_BIT> > conv_${No}_padding_out("conv_${No}_padding_out");
reshape_buffer_INPE_SIMD<CONV_${No}_K, CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_IN_CH,CONV_${No}_OUT_CH / CONV_${No}_PE,
                         CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(conv_${No_pre}_layer_out, conv_${No}_padding_out, reps);
    ''')

conv_array_temp = Template('''//--------------------Conv ${No}: Computing Array--------------------
stream<ap_uint<CONV_${No}_Np * CONV_${No}_PE * CONV_${No}_M_BIT> > conv_${No}_array_out("conv_${No}_array_out");
Conv_Array_Cascade<CONV_${No}_K, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_IN_CH, CONV_${No}_OUT_CH,
                   CONV_${No}_IN_BIT, CONV_${No}_W_BIT, CONV_${No}_SIMD, CONV_${No}_PE, CONV_${No}_Kp,
                   CONV_${No}_Np, CONV_${No}_PA_BIT, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
                   CONV_${No}_SIMD_BIT, CONV_${No}_adW_BIT>(conv_${No}_padding_out, conv_${No}_w, conv_${No}_array_out, reps);
    ''')

red_bw_temp = Template('''//--------------------Conv ${No}: Decrease Bit-width--------------------
const unsigned CONV_${No}_DEC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_PE) * CONV_${No}_ROW_LEN;
stream<ap_uint<CONV_${No}_ACTP * CONV_${No}_M_BIT> > conv_${No}_dec_bw_out("conv_${No}_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_${No}_Np * CONV_${No}_PE * CONV_${No}_M_BIT, CONV_${No}_ACTP * CONV_${No}_M_BIT, CONV_${No}_DEC_BW_NUM>(conv_${No}_array_out, conv_${No}_dec_bw_out, reps);
    ''')


act_trim_temp = Template('''//--------------------Conv ${No}: Activate and Trim--------------------
stream<ap_uint<CONV_${No}_ACTP * CONV_${No}_OUT_BIT> > conv_${No}_act_out("conv_${No}_act_out");
Activation_Trim<CONV_${No}_K, CONV_${No}_IN_W, CONV_${No}_ROW_LEN, CONV_${No}_IN_H, CONV_${No}_OUT_CH,
CONV_${No}_IN_BIT, CONV_${No}_OUT_BIT, CONV_${No}_W_BIT, CONV_${No}_INC_BIT, CONV_${No}_BIAS_BIT,
CONV_${No}_L_SHIFT, CONV_${No}_PE, CONV_${No}_ACTP, CONV_${No}_Np, CONV_${No}_M_BIT>(conv_${No}_dec_bw_out, conv_${No}_inc, conv_${No}_bias, conv_${No}_act_out, reps);
    ''')


inc_bw_temp_1 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_PE) * CONV_${No}_IN_W * (CONV_${No}_PE / CONV_${No}_ACTP);
stream<ap_uint<2 * CONV_${No}_PE * CONV_${No}_OUT_BIT> > conv_${No}_conv_out("conv_${No}_conv_out");
#pragma HLS STREAM variable = conv_${No}_conv_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, 2 * CONV_${No}_PE * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_conv_out, reps);

#ifdef DEBUG
cout << "conv_${No}_conv_out size " << conv_${No}_conv_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_PE,
                                 CONV_${No_lat}_IN_BIT>(conv_${No}_conv_out, output_path+"conv_${No}_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
hls::stream<ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> > conv_${No}_layer_out("pool_${No}_layer_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = ${POOL_DEPTH}
max_pool2x2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_BIT,
            CONV_${No}_PE>(conv_${No}_conv_out, conv_${No}_layer_out, reps);
#ifdef DEBUG
cout << "conv_${No}_pool_out size " << conv_${No}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H / 2, CONV_${No}_IN_W / 2,
                           CONV_${No}_IN_CH, CONV_${No}_PE, CONV_${No}_OUT_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_pool_out.txt", reps);
#endif
''')

inc_bw_temp_2 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_PE) * CONV_${No}_IN_W * (CONV_${No}_PE / CONV_${No}_ACTP);
stream<ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> > conv_${No}_layer_out("conv_${No}_conv_out");
#pragma HLS STREAM variable = conv_${No}_layer_out depth = ${CONV_DEPTH}
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, CONV_${No}_PE * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, conv_${No}_layer_out, reps);

#ifdef DEBUG
cout << "conv_${No}_out size " << conv_${No}_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_PE,
                                 CONV_${No_lat}_IN_BIT>(conv_${No}_layer_out, output_path+"conv_${No}_conv_out.txt", reps);
#endif
''')


def gen_conv_para(conv_n, conv_k, conv_ich, conv_kp, conv_np, conv_gb, conv_simd):
    return para_temp.substitute(No=str(conv_n), EX_M_BIT=str(math.ceil(math.log2(conv_k * conv_k * conv_ich))),
                                SIMD_BIT=str(math.ceil(math.log2(conv_simd * min(conv_kp, conv_np)))),
                                CASCADE=str(find_CASCADE(conv_gb, conv_kp, conv_np, conv_simd)))


def gen_reshape_buffer(conv_n, conv_simd, conv_in_pe):
    if conv_simd >= conv_in_pe:
        return rebuffer_temp_1.substitute(No=str(conv_n), No_pre=str(conv_n-1))
    else:
        return rebuffer_temp_2.substitute(No=str(conv_n), No_pre=str(conv_n-1))

def gen_conv_array(conv_n):
    return conv_array_temp.substitute(No=str(conv_n))

def gen_reduce_bw(conv_n):
    return red_bw_temp.substitute(No=str(conv_n))

def gen_act_trim(conv_n):
    return act_trim_temp.substitute(No=str(conv_n))

def gen_increase_bw(conv_n, max_pool, conv_icol, conv_och, conv_pe):
    if max_pool:
        return inc_bw_temp_1.substitute(No=str(conv_n), No_lat=str(conv_n+1), CONV_DEPTH=str(math.ceil(conv_icol * conv_och / conv_pe)), POOL_DEPTH=str(math.ceil(conv_icol * conv_och / (conv_pe * 2))))
    else:
        return inc_bw_temp_2.substitute(No=str(conv_n), No_lat=str(conv_n+1), CONV_DEPTH=str(math.ceil(conv_icol * conv_och / conv_pe)))



if __name__ == '__main__':
    path = pathlib.Path('E:/Projects/DeepBurning_MixQ/Sampling/1_samp_gen/gen/')

    with open('E:/Projects/DeepBurning_MixQ/Sampling/1_samp_gen/opt_samples.json', 'r') as fdict:
        opt_dicts = json.load(fdict)

    for idx, opt_key in enumerate(opt_dicts):
        sample_dir = path / str(opt_key)
        if not sample_dir.is_dir():
            sample_dir.mkdir()

        hls_dir = sample_dir / 'hls'
        if not hls_dir.is_dir():
            hls_dir.mkdir()

        opt = opt_dicts[opt_key]

        conv_n = 1
        conv_k = opt['K']
        conv_ich = opt['IN_CH']
        conv_kp = opt['Kp']
        conv_np = opt['Np']
        conv_gb = opt['GUARD_BIT']
        conv_simd = opt['SIMD']
        conv_in_pe = opt['IN_PE']
        max_pool = False
        conv_icol = opt['W']
        conv_och = opt['OUT_CH']
        conv_pe = opt['PE']

        content = "//------------------------------------------"
        content += gen_conv_para(conv_n, conv_k, conv_ich, conv_kp, conv_np, conv_gb, conv_simd)
        content += f'\n'
        content += gen_reshape_buffer(conv_n, conv_simd, conv_in_pe)
        content += f'\n'
        content += gen_conv_array(conv_n)
        content += f'\n'
        content += gen_reduce_bw(conv_n)
        content += f'\n'
        content += gen_act_trim(conv_n)
        content += f'\n'
        content += gen_increase_bw(conv_n, max_pool, conv_icol, conv_och, conv_pe)
        content += f'\n'

        with open(hls_dir / 'conv_layer.cpp', 'w') as f:
            print(content, file=f)


    
    # print(collected_datas)
