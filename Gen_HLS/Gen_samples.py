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


front_temp = Template('''/********************************************************************************
* Operator Name: ${opt_name}
* Filename: conv_layer.cpp
* Date: ${computer_time}
* Description: this file is generated for sampling operator configurations
********************************************************************************/
//#define DEBUG
//#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include "config.h"
#include "weights.hpp"
#include <ap_int.h>
#include <stdint.h>
#include "${template_dir}/S2P_buffer.hpp"
#include "${template_dir}/Conv_Opt.hpp"
#include "${template_dir}/function.h"
#include "${template_dir}/pool_reord.hpp"
#include "${template_dir}/stream_tools.h"

string output_path = "./${debug_path}/";

struct in_ap_axis {
    ap_uint<IP_BW> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

struct out_ap_axis {
    ap_uint<OP_BW> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};


void operator_wrapper(stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &in, stream<ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> > &out,
                      const unsigned reps=1){
#pragma HLS DATAFLOW

    ''')


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
                         CONV_${No}_Np, CONV_${No}_IN_BIT, CONV_${No}_IN_PE, CONV_${No}_SIMD>(in, conv_${No}_padding_out, reps);
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
                   CONV_${No}_Np, CONV_${No}_CASCADE, CONV_${No}_GUARD_BIT, CONV_${No}_M_BIT, 
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
max_pool2x2<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_OUT_BIT,
            CONV_${No}_PE>(conv_${No}_conv_out, out, reps);
#ifdef DEBUG
cout << "conv_${No}_pool_out size " << conv_${No}_layer_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H / 2, CONV_${No}_IN_W / 2,
                           CONV_${No}_IN_CH, CONV_${No}_PE, CONV_${No}_OUT_BIT>(out, output_path+"conv_${No}_pool_out.txt", reps);
#endif
''')

inc_bw_temp_2 = Template('''//--------------------Conv ${No}: Increase Bit-width--------------------
const unsigned CONV_${No}_INC_BW_NUM = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_PE) * CONV_${No}_IN_W * (CONV_${No}_PE / CONV_${No}_ACTP);
StreamingDataWidthConverter_Batch<CONV_${No}_ACTP * CONV_${No}_OUT_BIT, CONV_${No}_PE * CONV_${No}_OUT_BIT, CONV_${No}_INC_BW_NUM>(conv_${No}_act_out, out, reps);

#ifdef DEBUG
cout << "conv_${No}_out size " << conv_${No}_out.size() << endl;
print_mavu_DSPopt_stream_through<CONV_${No}_IN_H, CONV_${No}_IN_W, CONV_${No}_OUT_CH, CONV_${No}_PE,
                                 CONV_${No_lat}_IN_BIT>(out, output_path+"conv_${No}_conv_out.txt", reps);
#endif
''')

back_temp = Template('''}

template <unsigned in_NumLines>
void axi2in(stream<in_ap_axis> &in, stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &operator_in, const unsigned reps=1){
    in_ap_axis in_temp;
    for (unsigned rep = 0; rep < reps * in_NumLines; rep++) {
#pragma HLS PIPELINE II = 1
        in_temp = in.read();
        operator_in.write(in_temp.data(CONV_${No}_IN_PE * CONV_${No}_IN_BIT - 1, 0));
    }
}

template <unsigned out_NumLines>
void out2axi(stream<ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> > &add_last_in, stream<out_ap_axis> &out, unsigned reps=1){
    out_ap_axis out_temp;
    out_temp.keep = 0xff;

    for (unsigned i = 0; i < reps * out_NumLines - 1; i++) {
#pragma HLS pipeline
        ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> low_seg = add_last_in.read();
        out_temp.data = ((ap_uint<OP_BW - CONV_${No}_PE*CONV_${No}_OUT_BIT>) 0, low_seg);
        out_temp.last = 0;
        out.write(out_temp);
    }
    ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> low_seg = add_last_in.read();
    out_temp.data = ((ap_uint<OP_BW - CONV_${No}_PE*CONV_${No}_OUT_BIT>) 0, low_seg);
    out_temp.last = 1;
    out.write(out_temp);
}


void conv_operator(stream<in_ap_axis> &in, stream<out_ap_axis> &out,
                   const unsigned reps) {
#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_${No}_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_bias complete dim = 1

#pragma HLS DATAFLOW

    stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > operator_in;
#pragma HLS STREAM variable = operator_in depth = 64
    const unsigned in_NumLines = CONV_${No}_IN_H * (CONV_${No}_IN_CH / CONV_${No}_IN_PE) * CONV_${No}_IN_W;
    axi2in<in_NumLines>(in, operator_in, reps); 

    stream<ap_uint<CONV_${No}_PE * CONV_${No}_OUT_BIT> > add_last_in;
#pragma HLS STREAM variable = add_last_in depth = 64
    operator_wrapper(operator_in, add_last_in, reps);

    const unsigned out_NumLines = CONV_${No}_IN_H * (CONV_${No}_OUT_CH / CONV_${No}_PE) * CONV_${No}_IN_W;
    out2axi<out_NumLines>(add_last_in, out, reps);
}
    ''')

def gen_front(opt_name, conv_n, debug_dir, template_dir):
    return front_temp.substitute(opt_name=opt_name, No=str(conv_n), debug_path=debug_dir, computer_time=str(time.ctime()), template_dir=template_dir)


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

def gen_back(conv_n):
    return back_temp.substitute(No=str(conv_n))


tcl_template = Template("""#********************************************************************************
#* Operator Name: ${opt_name}
#* Filename: hls_proj.tcl
#* Date: ${computer_time}
#* Description: script for generating HLS project
#********************************************************************************/
open_project ${proj_name} 
set_top conv_operator
add_files ${template_dir}/function.h
add_files ${template_dir}/debug.hpp
add_files ${template_dir}/pool_reord.hpp
add_files ${template_dir}/stream_tools.h
add_files ${template_dir}/S2P_buffer.hpp
add_files ${template_dir}/Conv_Opt.hpp
add_files ./conv_layer.cpp
add_files ./config.h
add_files ./weights.hpp
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period ${clock_cycle} -name default
exit
""")


def gen_hls_tcl(opt_name, template_dir, clock_cycle):
    return tcl_template.substitute(proj_name=str(opt_name)+"_hls",
        template_dir=str(template_dir).replace("\\", "/"),
        clock_cycle=str(clock_cycle),
        opt_name=str(opt_name),
        computer_time=time.ctime())

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

def write_pseudo_weights(opt_name, path, och, ich, k, simd, pe, wbit, abit, actp, n):
    f = open(path / 'weights.hpp', 'w')
    print(f'''/********************************************************************************
* Operator Name: {str(opt_name)}
* Filename: weights.hpp
* Date: {time.ctime()}
* Description: weights file for the operator
********************************************************************************/

#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_
#include <ap_int.h>
''', file=f)

    w = (2**(wbit-1)) * (np.random.rand(och, ich, k, k) - 0.5)    # [och, ich, kr, kc]
    w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]
    w = w.reshape(och//pe, pe, k, k*ich//simd, simd)  # [och/pe, pe, kc, kr*ich/simd, simd]
    w = w.transpose(1, 0, 3, 2, 4) # [pe, och/pe, kr*ich/simd, kc, simd]
    # w = w[:, :, :, ::-1, :]  # omit for simplification, since weights are fake
    w = w.reshape(pe, -1, k*simd)  # [pe, och/pe*kr*ich/simd, kc*simd]

    print(f"// layer: {n}, PE: {pe}, SIMD: {simd}, wbit: {wbit}", file=f)

    # print conv weight,  merge [SIMD] value into one ap_uint
    print(f"const ap_uint<{k * wbit * simd}> conv_{n}_w[{pe}][{w.shape[1]}]=", file=f)
        
    hex_str = lambda x: '"' + hex(x) + '"'
    def pack1d_str(arr): # x: 1d-array
        x = 0
        for v in arr[::-1]: # [!] reverse simd pack, it is related to hls implemention
            v = int(v) # use python bignumber, not np.int
            assert -1<<wbit-1 <= v < 1<<wbit-1, f'got v={v} while wbit={wbit}'
            x=(x<<wbit) + (v&(2**wbit-1))
        return hex_str(x)
    print_ndarray_recursion(w, pack1d_str, f, stop=1)
    print(';', file=f)

    inc = (2**(wbit+abit+6-1)) * (np.random.rand(actp, och//actp) - 0.5)
    bias = (2**(wbit+abit+15-1)) * (np.random.rand(actp, och//actp) - 0.5)

    inc = inc.astype(np.int64)
    bias = bias.astype(np.int64)

    hex_str = lambda x: '"' + hex(x) + '"'
    print(f"const ap_int<{wbit+abit+6}> conv_{n}_inc[{actp}][{och//actp}]=", file=f)
    print_ndarray_recursion(inc, hex_str, f)
    print(';', file=f)
    print(f"const ap_int<{wbit+abit+15}> conv_{n}_bias[{actp}][{och//actp}]=", file=f)
    print_ndarray_recursion(bias, hex_str, f)
    print(';', file=f)
    print('#endif', file=f)

def write_hls_config(opt_name, opt, n, path):
    attr_list = ['K', 'IN_W', 'IN_H', 'IN_CH', 'OUT_CH', 'IN_BIT', 'W_BIT', 'OUT_BIT', 'IN_PE', 'SIMD', 'PE', 'ACTP', 'Kp', 'Np', 'GUARD_BIT', 'T_mul', 'INC_BIT', 'BIAS_BIT', 'L_SHIFT']
    content = f'''/********************************************************************************
* Operator Name: {str(opt_name)}
* Filename: config.h
* Date: {time.ctime()}
* Description: configuration file for the operator
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

'''
    for key in attr_list:
        if opt.get(key):
            content += f'#define CONV_{n}_{str(key)} {opt[key]}\n'

    content += f"#define IP_BW {2 ** math.ceil(math.log2(opt['IN_PE'] * opt['IN_BIT']))}\n"
    content += f"#define OP_BW {2 ** math.ceil(math.log2(opt['PE'] * opt['OUT_BIT']))}\n"
    content += f"#endif"
    with open(path / 'config.h', 'w') as f:
        print(content, file=f)


if __name__ == '__main__':
    path = pathlib.Path('E:/Projects/DeepBurning_MixQ/Sampling/1_samp_gen/gen/')
    template_dir = pathlib.Path('E:/Projects/DeepBurning_MixQ/Sampling/1_samp_gen/src')

    with open('./opt_samples.json', 'r') as fdict:
        opt_dicts = json.load(fdict)

    for idx, opt_key in enumerate(opt_dicts):
        sample_dir = path / str(opt_key)
        if not sample_dir.is_dir():
            sample_dir.mkdir()

        hls_dir = sample_dir / 'hls'
        if not hls_dir.is_dir():
            hls_dir.mkdir()

        hls_debug_dir = hls_dir / 'debug'
        if not hls_debug_dir.is_dir():
            hls_debug_dir.mkdir()

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
        conv_icol = opt['IN_W']
        conv_och = opt['OUT_CH']
        conv_pe = opt['PE']
        conv_wbit = opt['W_BIT']
        conv_abit = opt['IN_BIT']
        conv_actp = opt['ACTP']

        content = gen_front(str(opt_key), conv_n, str(hls_debug_dir).replace("\\", "/"), str(template_dir).replace("\\", "/"))
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
        content += gen_back(conv_n)


        with open(hls_dir / 'conv_layer.cpp', 'w') as f:
            print(content, file=f)

        with open(hls_dir / 'hls_proj.tcl', 'w') as f:
            print(gen_hls_tcl(opt_key, template_dir, 4), file=f)

        write_pseudo_weights(opt_key, hls_dir, conv_och, conv_ich, conv_k, conv_simd, conv_pe, conv_wbit, conv_abit, conv_actp, 1)
        write_hls_config(opt_key, opt, 1, hls_dir)


    
    # print(collected_datas)
