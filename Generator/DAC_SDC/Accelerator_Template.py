from string import Template
import time
import pathlib
from TB_Template import gen_tb

hls_templates_path = pathlib.Path('../../HLS_Templates/')

def get_headfiles(path=hls_templates_path):
    content = f'''/********************************************************************************
* Filename: weights.hpp
* Date: ${str(time.ctime())}
* Description: accelerator main function
********************************************************************************/
'''
    content += '''//#define DEBUG
//#include "debug.hpp"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include <ap_int.h>
#include <iostream>
#include <stdint.h>

#include "config.h"
#include "weights.hpp"
'''
    for f in path.iterdir():
        file_name = str(f).split('\\')[-1]
        if file_name == 'debug.hpp':
            continue
        content += f'#include "{file_name}"\n'

    content += '\n'

    return content

def get_parameter_partition(model_opt):
    content = ''
    for opt in model_opt:
        if hasattr(opt.conv, 'w') and opt.conv.w is not None:
            content += f'#pragma HLS ARRAY_PARTITION variable = conv_{opt.conv.n}_w complete dim = 1\n'
        if hasattr(opt.conv, 'inc') and opt.conv.inc is not None:
            content += f'#pragma HLS ARRAY_PARTITION variable = conv_{opt.conv.n}_inc complete dim = 1\n'
        if hasattr(opt.conv, 'bias') and opt.conv.bias is not None:
            content += f'#pragma HLS ARRAY_PARTITION variable = conv_{opt.conv.n}_bias complete dim = 1\n'

        content += '\n'

    return content

Accelerator = Template('''${head_files}

string output_path = "${debug_path}";

template <unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT, unsigned IN_NUM>
void input_quant(hls::stream<ap_uint<IN_BIT * IN_CH> > &in,
                 hls::stream<ap_uint<OUT_BIT * IN_CH> > &out,
                 const unsigned int reps){
  for(unsigned int i = 0; i < (IN_NUM * reps); i++){
#pragma HLS PIPELINE II = 1
    ap_uint<IN_BIT * IN_CH> indata = in.read();
    ap_uint<OUT_BIT * IN_CH> outdata;
    for(unsigned int j = 0; j < IN_CH; j++){
#pragma HLS unroll
      outdata(j * OUT_BIT + OUT_BIT - 1, j * OUT_BIT) = indata(j * IN_BIT + IN_BIT - 1, j * IN_BIT + IN_BIT - OUT_BIT);
    }
    out.write(outdata);
  }
}

void compute_pipeline(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
                 const unsigned int reps = 1) {
#pragma HLS DATAFLOW

  const unsigned int num_per_rep = ${input_height} * ${input_width} * 3 * 8 / 64;

  hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable = in_stream_extract depth = 256
  ExtractPixels<64, num_per_rep>(in, in_stream_extract, reps);

  hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable = in_stream0 depth = 256
  StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract,
                                                             in_stream0, reps);

  hls::stream<ap_uint<8 * CONV_0_IN_CH> > in_stream1("in_stream1");
#pragma HLS STREAM variable = in_stream1 depth = 512

  StreamingDataWidthConverter_Batch<64 * 3, 8 * CONV_0_IN_CH,
                                    num_per_rep / 3>(in_stream0, in_stream1,
                                                     reps);
#ifdef DEBUG
  cout << "in_stream1 size " << in_stream1.size() << endl;

#endif

  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_IN_CH> > conv0_in("conv0_in");
#pragma HLS STREAM variable = conv0_in depth = 512
  const unsigned input_quant_num = ${input_height} * ${input_width} * 3 / CONV_0_IN_CH;
  input_quant<8, CONV_0_IN_CH, CONV_0_IN_BIT, input_quant_num>(in_stream1, conv0_in, reps);

${layers}
}

void ${net_name}(stream<my_ap_axis> &in, stream<my_ap_axis> &out,
               const unsigned reps) {

#pragma HLS INTERFACE axis register both port = out
#pragma HLS INTERFACE axis register both port = in
#pragma HLS INTERFACE s_axilite port = reps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

${parameter_partition}

  compute_pipeline(in, out, reps);
}
''')

def get_accelerator(net_name, input_width, input_height, layers, model_opt, debug_path='./debug_path/'):
    head_files = get_headfiles()
    parameter_partition = get_parameter_partition(model_opt)
    accelerator = Accelerator.substitute(head_files=head_files, debug_path=str(debug_path).replace('\\', '/'), input_height=str(input_height),
                                         input_width=str(input_width), layers=layers, net_name=str(net_name), parameter_partition=parameter_partition)

    return accelerator


def write_hls_accel(model_opt, path, net_name, GenTB, debug_path='./debug_path/', input_path='./input/test1.bin', repeat_num=1):
    # set in_pe
    max_pool_scale = 1
    for idx, opt in enumerate(model_opt):
        if idx == 0:
            setattr(opt.conv, 'in_pe', 3)
        else:
            setattr(opt.conv, 'in_pe', model_opt[idx - 1].get_opf())
        if hasattr(opt.conv, 'max_pool') and opt.conv.max_pool == True:
            max_pool_scale = max_pool_scale * 2

    input_width = model_opt[0].conv.icol
    input_height = model_opt[0].conv.irow

    layers = ''
    for opt in model_opt:
        layers += opt.gen_operator()

    content = get_accelerator(net_name, input_width, input_height, layers, model_opt, debug_path=debug_path)

    with open(path + 'accelerator.cpp', 'w') as f:
        print(content, file=f)

    if GenTB:
        grid_row = model_opt[-1].conv.irow
        grid_col = model_opt[-1].conv.icol
        org_row = 360
        org_col = 640
        inp_row = model_opt[0].conv.irow
        inp_col = model_opt[0].conv.icol
        div = model_opt[-1].conv.div
    
        post_processing = f'''#define grid_row {grid_row}
#define grid_col {grid_col}
#define org_row {org_row}
#define org_col {org_col}
#define inp_row {inp_row}
#define inp_col {inp_col}
#define div {div}\n'''
        with open(path + 'tb.cpp', 'w') as f:
            content = gen_tb(str(debug_path).replace('\\', '/'), net_name, str(input_path).replace('\\', '/'), max_pool_scale, post_processing, repeat_num)
            print(content, file=f)

def write_hls_config(model_opt, path):
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
    content = f'''/********************************************************************************
* Filename: config.h
* Date: ${str(time.ctime())}
* Description: configuration file for the generated accelerator
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

'''
    for n, opt in enumerate(model_opt):
        content += f'// conv_{n}\n'
        for k, v in name_mapping.items():
            if hasattr(opt.conv, k): # e.g. conv_last has no incbit
                content += f'#define CONV_{n}_{v} {getattr(opt.conv, k)}\n'
        content += '\n'

    content += '#endif'

    with open(path + 'config.h', 'w') as f:
        print(content, file=f)


def write_hls_weights(model_opt, path):
    '''write_hls_weights(model_param, path)
    Write hls weights+inc+bias array code according to numpy shape.
    '''
    f = open(path + 'weights.hpp', 'w')

    print(f'''/********************************************************************************
* Filename: weights.hpp
* Date: ${str(time.ctime())}
* Description: weights and other parameters for the generated accelerator
********************************************************************************/

#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_
#include <ap_int.h>
''', file=f)

    for opt in model_opt:
        print(f"Write conv_{opt.conv.n} weight, simd {opt.conv.simd}, pe {opt.conv.pe}, kpf {opt.conv.kpf}, actp {opt.conv.actp}, wbit {opt.conv.wbit}")
        content = opt.write_weights()
        print(content, file=f, end='')
    
    print('#endif', file=f)
    f.close()



