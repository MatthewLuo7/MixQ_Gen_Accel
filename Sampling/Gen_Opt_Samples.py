import argparse
import time
import numpy as np
import math
import json
import pathlib
from string import Template
import sys
sys.path.append('../DSP_Explorer/')
sys.path.append('../Opt_Definition/ConvOpt_FP/')
sys.path.append('../Opt_Definition/ConvOpt_FP_DW/')
sys.path.append('../Opt_Definition/ConvOpt_FP_DW_LUT/')
sys.path.append('../Opt_Definition/ConvOpt_FP_LUT/')
sys.path.append('../Opt_Definition/ConvOpt_KP/')
sys.path.append('../Opt_Definition/ConvOpt_KP_LUT/')

from ConvOpt_FP import FP_Opt_Templates
from ConvOpt_FP_LUT import FP_LUT_Opt_Templates
from ConvOpt_FP_DW import FP_DW_Opt_Templates
from ConvOpt_FP_DW_LUT import FP_DW_LUT_Opt_Templates
from ConvOpt_KP import KP_Opt_Templates
from ConvOpt_KP_LUT import KP_LUT_Opt_Templates

from Sampling import attr_names

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
        'kpf': 'KPF'
    }


def get_hls_template_list(path):
    hls_list = []
    for file in path.iterdir():
        file_name = str(file).replace("\\", "/")
        if file_name.split('/')[-1] == 'debug.hpp':
            continue
        hls_list.append(file_name)

    return hls_list

accel_temp = Template('''/********************************************************************************
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

${hls_templates}

#define INP_BW ${INP_BW}

string output_path = "./${debug_path}/";

void conv_operator(stream<ap_uint<INP_BW> > &conv0_in, stream<ap_uint<${Out_BW}> > &out,
                      const unsigned reps=1){
#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = conv0_in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS ARRAY_PARTITION variable = conv_${No}_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_${No}_bias complete dim = 1
#pragma HLS DATAFLOW

${sampled_opt}
}
''')

tcl_template = Template("""#********************************************************************************
#* Operator Name: ${opt_name}
#* Filename: hls_proj.tcl
#* Date: ${computer_time}
#* Description: script for generating HLS project
#********************************************************************************/
open_project ${proj_name} 
set_top conv_operator

${add_templates}

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
    setattr(conv, 'n', 0)
    opt_type = sample_dict['Type']
    opt_config = sample_dict['Config']
    for attr in attr_names:
        if attr in opt_config:
            setattr(conv, attr, opt_config[attr])

    # generate pseudo parameters
    conv.w = np.round((2**(conv.wbit-1) - 1) * (np.random.rand(conv.och, conv.ich, conv.k, conv.k)*2 - 1)).astype(np.int32)    # [och, ich, kr, kc]
    conv.inc = np.round((2**(conv.incbit-1) - 1) * (np.random.rand(conv.och)*2 - 1)).astype(np.int64)
    conv.bias = np.round((2**(conv.biasbit-1) - 1) * (np.random.rand(conv.och)*2 - 1)).astype(np.int64)

    return conv, opt_type

def opt_tcl(opt_name, hls_dir, template_path, clock_cycle=4):
    hls_templates_list = get_hls_template_list(template_path)
    add_templates = ''
    for file in hls_templates_list:
        add_templates += f'add_files {file}\n'

    content = tcl_template.substitute(proj_name=str(opt_name)+"_hls", template_path=str(template_path).replace("\\", "/"),
                                      clock_cycle=str(clock_cycle), opt_name=str(opt_name), computer_time=time.ctime(), add_templates=add_templates)
    with open(hls_dir / 'hls_proj.tcl', 'w') as f:
        print(content, file=f)

def opt_hls(opt, opt_name, Packing, src_dir, template_path):
    hls_debug_dir = src_dir / 'debug'
    if not hls_debug_dir.is_dir():
        hls_debug_dir.mkdir()
    if Packing == 'FP':
        Out_BW = f'CONV_{opt.conv.n}_PE * CONV_{opt.conv.n}_OUT_BIT'
    elif Packing == 'KP':
        Out_BW = f'CONV_{opt.conv.n}_PE * CONV_{opt.conv.n}_Kp * CONV_{opt.conv.n}_OUT_BIT'

    hls_templates_list = get_hls_template_list(template_path)
    hls_templates = ''
    for file in hls_templates_list:
        hls_templates += f'#include "{file}"\n'

    content = accel_temp.substitute(opt_name=opt_name, No=str(opt.conv.n), debug_path=str(hls_debug_dir).replace("\\", "/"),
                                    computer_time=str(time.ctime()), Out_BW=Out_BW, hls_templates=hls_templates,
                                    sampled_opt=opt.gen_operator_for_sampling(), INP_BW=str(opt.conv.in_pe * opt.conv.abit))

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
        if hasattr(opt.conv, k): # e.g. conv_last has no incbit
            content += f'#define CONV_{opt.conv.n}_{v} {getattr(opt.conv, k)}\n'
    content += '#endif'

    with open(src_dir / 'config.h', 'w') as f:
        print(content, file=f)

def conv_to_opt(conv, opt_type):
    Packing = opt_type['Packing']
    DW = opt_type['DW']
    LUT = opt_type['LUT']
    if Packing == 'FP':
        if DW:
            opt = FP_DW_LUT_Opt_Templates(conv) if LUT else FP_DW_Opt_Templates(conv)
        else:
            opt = FP_LUT_Opt_Templates(conv) if LUT else FP_Opt_Templates(conv)
    elif Packing == 'KP':
        opt = KP_LUT_Opt_Templates(conv) if LUT else KP_Opt_Templates(conv)
    else:
        raise NotImplementedError(f'{Packing} is not defined!')

    return opt

def gen_pseudo_opt(path, template_path, opt_name, sample_dict):
    conv, opt_type = dict_to_conv(sample_dict)
    Packing = opt_type['Packing']
    opt = conv_to_opt(conv, opt_type)

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
    opt_hls(opt, opt_name, Packing, src_dir, template_path)
    opt_tcl(opt_name, hls_dir, template_path, clock_cycle=opt_type['Latency'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample-file', help='file name of operator samples')
    parser.add_argument('-g', '--gen-path', help='path for generating samples')
    parser.add_argument('-t', '--template-path', help='path of hls templates', default='../HLS_Templates/')
    parser.add_argument('--start-idx', help='index for the first sample', default=0, type=int)
    parser.add_argument('--num', help='number of generated samples', default=-1, type=int)
    opt = parser.parse_args()

    sample_file = opt.sample_file
    gen_path = pathlib.Path(opt.gen_path).absolute()
    template_path = pathlib.Path(opt.template_path).absolute()
    
    # Check and create paths
    if not gen_path.is_dir():
        raise ValueError(f"Directory {gen_path} doesn't exist!")
    if not template_path.is_dir():
        raise ValueError(f"Directory {template_path} doesn't exist!")

    with open(sample_file, 'r') as fdict:
        opt_dicts = json.load(fdict)
    total_num = len(opt_dicts)

    start_idx = opt.start_idx
    if (start_idx < 0) or (start_idx >= total_num):
        raise ValueError(f"Start idx {start_idx} is illegal or exceeds the maximum length {total_num}.")

    num = opt.num
    end_idx = total_num if (num < 0) else start_idx + num
    if end_idx > total_num:
        raise ValueError(f'End idx {end_idx} exceeds with start idx {start_idx} and sample number {num}')

    for idx, (k, v) in enumerate(opt_dicts.items()):
        if (idx >= start_idx) and (idx < end_idx):
            gen_pseudo_opt(gen_path, template_path, k, v)
