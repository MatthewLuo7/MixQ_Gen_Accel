import argparse
import time
import json
from string import Template
import pathlib
import subprocess
from multiprocessing import Process
import sys
import os
import math
import time

import os
import xml.etree.ElementTree as ET


def search_II(cur_II, report_xml):
    tree = ET.parse(report_xml)
    II = cur_II
    for elem in tree.iter(tag='PipelineII'):
        II = max(II, int(elem.text))

    return II

def get_II(xml_path):
    II = 0
    for f in xml_path.iterdir():
        if f.suffix == ".xml":
            II = search_II(II, f)

    return II


def collect_sample(opt_name, proj_dir):
    report_dir = proj_dir / 'solution1' / 'impl' / 'report' / 'verilog'
    if not report_dir.is_dir():
        print(f"Error! Case {opt_name} does not have any Vivado Synthesis report!")
        return None
    csyn_dir = proj_dir / 'solution1' / 'syn' / 'report'
    if not csyn_dir.is_dir():
        print(f"Error! Case {opt_name} does not have any HLS C-Synthesis report!")
        return None
    syn_res_xml = os.path.normpath(str(report_dir / 'conv_operator_export.xml'))    

    # vivado syn res
    syn_res = {"resources":{},"timing":{}}
    tree = ET.parse(syn_res_xml)
    resources = tree.getroot().iterfind("AreaReport/Resources")
    for child in resources:
        for res_node in child:
            syn_res["resources"][res_node.tag] = int(res_node.text)
    timing = tree.getroot().iterfind("TimingReport")
    for child in timing:
        for tim_node in child:
            syn_res["timing"][tim_node.tag] = float(tim_node.text)

    # read II from csyn res
    II = int(get_II(csyn_dir))
    syn_res["timing"]["PipelineII"] = II
    if II == 0:
        print(f"Warning! Case {opt_name} does not have a valid II!")

    return syn_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample-file', help='file name of operator samples')
    parser.add_argument('-g', '--gen-path', help='path for generating samples')
    parser.add_argument('--start-idx', help='index for the first sample', default=0, type=int)
    parser.add_argument('--num', help='number of generated samples', default=-1, type=int)
    parser.add_argument('-d', '--dataset', help='save path for the dataset')
    opt = parser.parse_args()
    sample_file = opt.sample_file
    gen_path = pathlib.Path(opt.gen_path).absolute()
    save_dataset = opt.dataset

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

    opt_syn_dicts = {}
    Find_Num = 10
    for idx, (opt_key, sample_dict) in enumerate(opt_dicts.items()):
        if not ((idx >= start_idx) and (idx < end_idx)):
            continue
        sample_dir = gen_path / str(opt_key)
        if not sample_dir.is_dir():
            print(f"Cannot find the directory: {sample_dir}. Exit this case!")
            continue

        hls_dir = sample_dir / 'hls'
        if not hls_dir.is_dir():
            print(f"Cannot find the directory: {hls_dir}. Exit this case!")
            continue

        proj_dir = hls_dir / (str(opt_key)+"_hls")
        if not proj_dir.is_dir():
            print(f"Cannot find the directory: {proj_dir}. Exit this case!")
            continue

        opt_syn_dicts[opt_key] = {'Type': sample_dict['Type'], 'Config': sample_dict['Config'], 'Syn_Res': collect_sample(opt_key, proj_dir)}

    with open(save_dataset, 'w', encoding='utf-8') as f:
        json.dump(opt_syn_dicts, f, indent=4)
