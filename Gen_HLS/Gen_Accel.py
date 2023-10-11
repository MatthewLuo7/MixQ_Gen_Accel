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



def find_CASCADE(gb, kp, np, simd):
    cascade = 1
    upper = min((2**gb) // min(kp, np), simd)
    for factor in range(1, int(upper) + 1):
        if upper%factor == 0:
            cascade = factor

    return cascade


def gen_conv_para(conv):
    content = f'''//--------------------Conv {conv.n}: Parameters--------------------
const unsigned CONV_{conv.n}_M_BIT = CONV_{conv.n}_IN_BIT + CONV_{conv.n}_W_BIT + {math.ceil(math.log2(conv.k * conv.k * conv.ich))};
const unsigned CONV_{conv.n}_SIMD_BIT = {math.ceil(math.log2(conv.simd))};
const unsigned CONV_{conv.n}_PA_BIT = {math.ceil(math.log2(conv.simd * min(conv.kp, conv.np))) - math.ceil(math.log2(conv.simd))};
const unsigned CONV_{conv.n}_CASCADE = {find_CASCADE(conv.gb, conv.kp, conv.np, conv.simd)};
const unsigned CONV_{conv.n}_ROW_LEN = (CONV_{conv.n}_IN_W + CONV_{conv.n}_K - 1 - 1) / CONV_{conv.n}_Np + 1;
const unsigned CONV_{conv.n}_adW_BIT = 1;
'''

    return content


def gen_reshape_buffer(conv):
#     content = f'''//--------------------Conv {conv.n}: Reshape and Padding Buffer--------------------
# stream<ap_uint<CONV_{conv.n}_Np * CONV_{conv.n}_SIMD * CONV_{conv.n}_IN_BIT> > conv_{conv.n}_padding_out("conv_{conv.n}_padding_out");
# reshape_buffer_SIMD_INPE<CONV_{conv.n}_K, CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_IN_CH, CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE,
#                          CONV_{conv.n}_Np, CONV_{conv.n}_IN_BIT, CONV_{conv.n}_IN_PE, CONV_{conv.n}_SIMD>(conv_{conv.n-1}_out, conv_{conv.n}_padding_out, reps);
#     '''
    if conv.simd >= conv.in_pe:
        content = f'''//--------------------Conv {conv.n}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_{conv.n}_Np * CONV_{conv.n}_SIMD * CONV_{conv.n}_IN_BIT> > conv_{conv.n}_padding_out("conv_{conv.n}_padding_out");
reshape_buffer_SIMD_INPE<CONV_{conv.n}_K, CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_IN_CH, CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE,
                         CONV_{conv.n}_Np, CONV_{conv.n}_IN_BIT, CONV_{conv.n}_IN_PE, CONV_{conv.n}_SIMD>(conv_{conv.n-1}_layer_out, conv_{conv.n}_padding_out, reps);'''
    else:
        content = f'''//--------------------Conv {conv.n}: Reshape and Padding Buffer--------------------
stream<ap_uint<CONV_{conv.n}_Np * CONV_{conv.n}_SIMD * CONV_{conv.n}_IN_BIT> > conv_{conv.n}_padding_out("conv_{conv.n}_padding_out");
reshape_buffer_INPE_SIMD<CONV_{conv.n}_K, CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_IN_CH,CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE,
                         CONV_{conv.n}_Np, CONV_{conv.n}_IN_BIT, CONV_{conv.n}_IN_PE, CONV_{conv.n}_SIMD>(conv_{conv.n-1}_layer_out, conv_{conv.n}_padding_out, reps);'''

    return content

def gen_conv_array(conv):
    content = f'''//--------------------Conv {conv.n}: Computing Array--------------------
stream<ap_uint<CONV_{conv.n}_Np * CONV_{conv.n}_PE * CONV_{conv.n}_M_BIT> > conv_{conv.n}_array_out("conv_{conv.n}_array_out");
Conv_Array_Cascade<CONV_{conv.n}_K, CONV_{conv.n}_ROW_LEN, CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_CH, CONV_{conv.n}_OUT_CH,
                   CONV_{conv.n}_IN_BIT, CONV_{conv.n}_W_BIT, CONV_{conv.n}_SIMD, CONV_{conv.n}_PE, CONV_{conv.n}_Kp,
                   CONV_{conv.n}_Np, CONV_{conv.n}_PA_BIT, CONV_{conv.n}_CASCADE, CONV_{conv.n}_GUARD_BIT, CONV_{conv.n}_M_BIT, 
                   CONV_{conv.n}_SIMD_BIT, CONV_{conv.n}_adW_BIT>(conv_{conv.n}_padding_out, conv_{conv.n}_w, conv_{conv.n}_array_out, reps);
'''

    return content

def gen_reduce_bw(conv):
    content = f'''//--------------------Conv {conv.n}: Decrease Bit-width--------------------
const unsigned CONV_{conv.n}_DEC_BW_NUM = CONV_{conv.n}_IN_H * (CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE) * CONV_{conv.n}_ROW_LEN;
stream<ap_uint<CONV_{conv.n}_ACTP * CONV_{conv.n}_M_BIT> > conv_{conv.n}_dec_bw_out("conv_{conv.n}_dec_bw_out");
StreamingDataWidthConverter_Batch<CONV_{conv.n}_Np * CONV_{conv.n}_PE * CONV_{conv.n}_M_BIT, CONV_{conv.n}_ACTP * CONV_{conv.n}_M_BIT, CONV_{conv.n}_DEC_BW_NUM>(conv_{conv.n}_array_out, conv_{conv.n}_dec_bw_out, reps);
'''

    return content

def gen_act_trim(conv):
    content = f'''//--------------------Conv {conv.n}: Activate and Trim--------------------
stream<ap_uint<CONV_{conv.n}_ACTP * CONV_{conv.n}_OUT_BIT> > conv_{conv.n}_act_out("conv_{conv.n}_act_out");
Activation_Trim<CONV_{conv.n}_K, CONV_{conv.n}_IN_W, CONV_{conv.n}_ROW_LEN, CONV_{conv.n}_IN_H, CONV_{conv.n}_OUT_CH,
CONV_{conv.n}_IN_BIT, CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_W_BIT, CONV_{conv.n}_INC_BIT, CONV_{conv.n}_BIAS_BIT,
CONV_{conv.n}_L_SHIFT, CONV_{conv.n}_PE, CONV_{conv.n}_ACTP, CONV_{conv.n}_Np, CONV_{conv.n}_M_BIT>(conv_{conv.n}_dec_bw_out, conv_{conv.n}_inc, conv_{conv.n}_bias, conv_{conv.n}_act_out, reps);
'''

    return content

def gen_increase_bw(conv):
#     content = f'''//--------------------Conv {conv.n}: Increase Bit-width--------------------
# const unsigned CONV_{conv.n}_INC_BW_NUM = CONV_{conv.n}_IN_H * (CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE) * CONV_{conv.n}_IN_W * (CONV_{conv.n}_PE / CONV_{conv.n}_ACTP);
# stream<ap_uint<CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT> > conv_{conv.n}_out("conv_{conv.n}_out");
# StreamingDataWidthConverter_Batch<CONV_{conv.n}_ACTP * CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_INC_BW_NUM>(conv_{conv.n}_act_out, conv_{conv.n}_out, reps);

# #ifdef DEBUG
# cout << "conv_{conv.n}_out size " << conv_{conv.n}_out.size() << endl;
# print_mavu_DSPopt_stream_through<CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_OUT_CH, CONV_{conv.n}_PE,
#                                  CONV_{conv.n+1}_IN_BIT>(conv_{conv.n}_out, output_path+"conv_{conv.n}_out.txt", reps);
# #endif
# '''
    if conv.max_pool:
        content = f'''//--------------------Conv {conv.n}: Increase Bit-width--------------------
const unsigned CONV_{conv.n}_INC_BW_NUM = CONV_{conv.n}_IN_H * (CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE) * CONV_{conv.n}_IN_W * (CONV_{conv.n}_PE / CONV_{conv.n}_ACTP);
stream<ap_uint<2 * CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT> > conv_{conv.n}_conv_out("conv_{conv.n}_conv_out");
#pragma HLS STREAM variable = conv_{conv.n}_conv_out depth = {math.ceil(conv.icol * conv.och / conv.pe)}
StreamingDataWidthConverter_Batch<CONV_{conv.n}_ACTP * CONV_{conv.n}_OUT_BIT, 2 * CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_INC_BW_NUM>(conv_{conv.n}_act_out, conv_{conv.n}_conv_out, reps);

#ifdef DEBUG
cout << "conv_{conv.n}_conv_out size " << conv_{conv.n}_conv_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_OUT_CH, CONV_{conv.n}_PE,
                                 CONV_{conv.n+1}_IN_BIT>(conv_{conv.n}_conv_out, output_path+"conv_{conv.n}_conv_out.txt", reps);
#endif

//--------------------Pooling--------------------
hls::stream<ap_uint<CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT> > conv_{conv.n}_layer_out("pool_{conv.n}_layer_out");
#pragma HLS STREAM variable = conv_{conv.n}_layer_out depth = {math.ceil(conv.icol * conv.och / (conv.pe * 2))}
max_pool2x2<CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_OUT_CH, CONV_{conv.n}_OUT_BIT,
            CONV_{conv.n}_PE>(conv_{conv.n}_conv_out, conv_{conv.n}_layer_out, reps);
#ifdef DEBUG
cout << "conv_{conv.n}_pool_out size " << conv_{conv.n}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_{conv.n}_IN_H / 2, CONV_{conv.n}_IN_W / 2,
                           CONV_{conv.n}_IN_CH, CONV_{conv.n}_PE, CONV_{conv.n}_OUT_BIT>(conv_{conv.n}_layer_out, output_path+"conv_{conv.n}_pool_out.txt", reps);
#endif
'''
    else:
        content = f'''//--------------------Conv {conv.n}: Increase Bit-width--------------------
const unsigned CONV_{conv.n}_INC_BW_NUM = CONV_{conv.n}_IN_H * (CONV_{conv.n}_OUT_CH / CONV_{conv.n}_PE) * CONV_{conv.n}_IN_W * (CONV_{conv.n}_PE / CONV_{conv.n}_ACTP);
stream<ap_uint<CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT> > conv_{conv.n}_layer_out("conv_{conv.n}_conv_out");
#pragma HLS STREAM variable = conv_{conv.n}_layer_out depth = {math.ceil(conv.icol * conv.och / conv.pe)}
StreamingDataWidthConverter_Batch<CONV_{conv.n}_ACTP * CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_PE * CONV_{conv.n}_OUT_BIT, CONV_{conv.n}_INC_BW_NUM>(conv_{conv.n}_act_out, conv_{conv.n}_layer_out, reps);

#ifdef DEBUG
cout << "conv_{conv.n}_out size " << conv_{conv.n}_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_{conv.n}_IN_H, CONV_{conv.n}_IN_W, CONV_{conv.n}_OUT_CH, CONV_{conv.n}_PE,
                                 CONV_{conv.n+1}_IN_BIT>(conv_{conv.n}_layer_out, output_path+"conv_{conv.n}_conv_out.txt", reps);
#endif
'''