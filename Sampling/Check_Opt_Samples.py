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

from Sampling import attr_names, attr_to_tuple, Random_Opt
from Gen_Opt_Samples import dict_to_conv, conv_to_opt

def check_opt(sample_dict):
    conv, opt_type = dict_to_conv(sample_dict)
    Packing = opt_type['Packing']
    opt = conv_to_opt(conv, opt_type)
    # debug
    if max(opt.weight_shape()) > 8192:
        flag = True
    else:
        flag = False

    return flag, conv, float(opt_type['Latency'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample-file', help='file name of operator samples')
    # parser.add_argument('-g', '--gen-path', help='path for generating samples')
    # parser.add_argument('-t', '--template-path', help='path of hls templates', default='../HLS_Templates/')
    # parser.add_argument('--start-idx', help='index for the first sample', default=0, type=int)
    # parser.add_argument('--num', help='number of generated samples', default=-1, type=int)
    opt = parser.parse_args()

    sample_file = opt.sample_file
    # gen_path = pathlib.Path(opt.gen_path).absolute()
    # template_path = pathlib.Path(opt.template_path).absolute()
    
    # # Check and create paths
    # if not gen_path.is_dir():
    #     raise ValueError(f"Directory {gen_path} doesn't exist!")
    # if not template_path.is_dir():
    #     raise ValueError(f"Directory {template_path} doesn't exist!")

    with open(sample_file, 'r') as fdict:
        opt_dicts = json.load(fdict)
    total_num = len(opt_dicts)

    sample_tuple = ()
    problem_list = []
    for idx, (k, v) in enumerate(opt_dicts.items()):
        flag, conv, latency = check_opt(v)

        attr_vec = attr_to_tuple(conv, latency)
        temp_list = list(sample_tuple)
        temp_list.append(attr_vec)
        sample_tuple = tuple(temp_list)

        if flag:
            problem_list.append(k)

    print(problem_list)

    for key in problem_list:
        for i in range(100):
            constraint_flag, conv, Packing, DW, LUT, attr_vec, latency = Random_Opt()

            C1 = constraint_flag
            C2 = not attr_vec in sample_tuple

            if C1 and C2:
                dict_elem = {'Type': {'Packing': Packing, 'LUT': LUT, 'DW': DW, 'Latency': latency}}
                Config = {}
                for attr in attr_names:
                    if hasattr(conv, attr):
                        val = getattr(conv, attr)
                        if isinstance(val, np.integer):
                            val = int(val)
                        Config[attr] = val
                        
                dict_elem['Config'] = Config

                opt_dicts[key] = dict_elem

                break

            if i == 99:
                print(f"Failed to re-sample a new opt within 100 attempts. Exit!")
                exit(0)

    print(f"Finished fixing {len(problem_list)} samples!")

    with open('./Samples/opt_10000_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(opt_dicts, f, indent=4)

    with open('problem_list.json', 'w', encoding='utf-8') as f:
        json.dump(problem_list, f)


