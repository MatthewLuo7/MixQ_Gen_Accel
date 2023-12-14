import time
import numpy as np
import math
import json
import pathlib
from string import Template
from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates

name_mapping = {
    'n': 'No',
    'k': 'K',
    'icol': 'IN_W',
    'irow': 'IN_H',
    'ich': 'IN_CH',
    'och': 'OUT_CH',
    'abit': 'IN_BIT',
    'wbit': 'W_BIT',
    'obit': 'OUT_BIT',
    'in_pe': 'IN_PE',
    'simd': 'SIMD',
    'pe': 'PE',
    'actp': 'ACTP',
    'kp': 'Kp',
    'np': 'Np',
    'gb': 'GUARD_BIT',
    'incbit': 'INC_BIT',
    'biasbit': 'BIAS_BIT',
    'lshift': 'L_SHIFT',
    'kpf': 'KPF',
    'max_pool': 'MAX_POOL'
    }

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
#include "${template_dir}/Opt_FP.hpp"
#include "${template_dir}/Opt_KP.hpp"
#include "${template_dir}/function.h"
#include "${template_dir}/pool_reord.hpp"
#include "${template_dir}/stream_tools.h"

string output_path = "./${debug_path}/";

void conv_operator(stream<ap_uint<CONV_${No}_IN_PE * CONV_${No}_IN_BIT> > &conv0_in, stream<ap_uint<${Out_BW}> > &out,
                      const unsigned reps=1){
#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = conv0_in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_${No}_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_bias complete dim = 1
#pragma HLS DATAFLOW

''')

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
add_files ${template_dir}/Opt_FP.hpp
add_files ${template_dir}/Opt_KP.hpp
add_files ../src/conv_layer.cpp
add_files ../src/config.h
add_files ../src/weights.hpp
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period ${clock_cycle} -name default
csynth_design
export_design -flow syn -rtl verilog -format ip_catalog
exit
""")

para_template = Template('''/********************************************************************************
* Operator Name: ${opt_name}
* Filename: weights.hpp
* Date: ${computer_time}
* Description: weights file for the operator
********************************************************************************/

#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_
#include <ap_int.h>

''')

config_template = Template('''/********************************************************************************
* Operator Name: ${opt_name}
* Filename: config.h
* Date: ${computer_time}
* Description: configuration file for the operator
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

''')

class ConvParam: ...

def dict_to_conv(sample_dict):
    conv = ConvParam()
    opt_type = sample_dict['Opt_Type']
    for k, v in name_mapping.items():
        if v in sample_dict:
            setattr(conv, k, sample_dict[v])
        else:
            if str(v) == 'KPF':
                setattr(conv, k, 1)
            else:
                raise ValueError(f'There is no key named as {str(v)} in sample dictionary!')

    # generate pseudo parameters
    conv.w = np.round((2**(conv.wbit-1) - 1) * (np.random.rand(conv.och, conv.ich, conv.k, conv.k)*2 - 1)).astype(np.int32)    # [och, ich, kr, kc]
    conv.inc = np.round((2**(conv.incbit-1) - 1) * (np.random.rand(conv.och)*2 - 1)).astype(np.int64)
    conv.bias = np.round((2**(conv.biasbit-1) - 1) * (np.random.rand(conv.och)*2 - 1)).astype(np.int64)

    return conv, opt_type

def opt_tcl(opt_name, hls_dir, template_dir, clock_cycle=4):
    content = tcl_template.substitute(proj_name=str(opt_name)+"_hls", template_dir=str(template_dir).replace("\\", "/"),
                                      clock_cycle=str(clock_cycle), opt_name=str(opt_name), computer_time=time.ctime())
    with open(hls_dir / 'hls_proj.tcl', 'w') as f:
        print(content, file=f)

def opt_hls(opt, opt_name, opt_type, src_dir, template_dir):
    hls_debug_dir = src_dir / 'debug'
    if not hls_debug_dir.is_dir():
        hls_debug_dir.mkdir()
    if opt_type == 'FP_Opt':
        Out_BW = f'CONV_{opt.conv.n}_PE * CONV_{opt.conv.n}_OUT_BIT'
    elif opt_type == 'KP_Opt':
        Out_BW = f'CONV_{opt.conv.n}_PE * CONV_{opt.conv.n}_Kp * CONV_{opt.conv.n}_OUT_BIT'
    content = front_temp.substitute(opt_name=opt_name, No=str(opt.conv.n), debug_path=str(hls_debug_dir).replace("\\", "/"),
                                    computer_time=str(time.ctime()), template_dir=str(template_dir).replace("\\", "/"),
                                    Out_BW=Out_BW)
    content += opt.gen_operator_for_sampling()
    content += '}'

    with open(src_dir / 'conv_layer.cpp', 'w') as f:
        print(content, file=f)

def opt_para(opt, opt_name, src_dir):
    content = para_template.substitute(opt_name=opt_name, computer_time=str(time.ctime()))
    content += opt.write_weights()
    content += '#endif'

    with open(src_dir / 'weights.hpp', 'w') as f:
        print(content, file=f)

def opt_config(opt, opt_name, src_dir):
    content = config_template.substitute(opt_name=opt_name, computer_time=str(time.ctime()))
    for k, v in name_mapping.items():
        if k == 'max_pool':
            continue
        if hasattr(opt.conv, k): # e.g. conv_last has no incbit
            content += f'#define CONV_{opt.conv.n}_{v} {getattr(opt.conv, k)}\n'
    content += '#endif'

    with open(src_dir / 'config.h', 'w') as f:
        print(content, file=f)

def conv_to_opt(conv, opt_type):
    if opt_type == 'FP_Opt':
        return FP_Opt_Templates(conv)
    elif opt_type == 'KP_Opt':
        return KP_Opt_Templates(conv)
    else:
        raise ValueError(f'{opt_type} is not a defined operator!')

def gen_pseudo_opt(path, template_dir, opt_name, sample_dict):
    conv, opt_type = dict_to_conv(sample_dict)
    opt = conv_to_opt(conv, opt_type)

    # Check and create paths
    if not template_dir.is_dir():
        print(f"Error: {template_dir} doesn't exist!")
        exit(0)

    sample_dir = path / str(opt_name)
    if not sample_dir.is_dir():
        sample_dir.mkdir()

    src_dir = sample_dir / 'src'
    if not src_dir.is_dir():
        src_dir.mkdir()

    hls_dir =sample_dir / 'hls'
    if not hls_dir.is_dir():
        hls_dir.mkdir()

    # Generate files
    opt_config(opt, opt_name, src_dir)
    opt_para(opt, opt_name, src_dir)
    opt_hls(opt, opt_name, opt_type, src_dir, template_dir)
    opt_tcl(opt_name, hls_dir, template_dir)




if __name__ == '__main__':
    with open('./opt_samples_5.json', 'r') as fdict:
        opt_dicts = json.load(fdict)

    path = pathlib.Path('/media/lab_admin/Data/Erjing/Sampling/3_sampFP1000/samples')
    template_dir = pathlib.Path('/media/lab_admin/Data/Erjing/Sampling/3_sampFP1000/templates')

    for k, v in opt_dicts.items():
        gen_pseudo_opt(path, template_dir, k, v)
