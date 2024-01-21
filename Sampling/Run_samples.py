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

def run_hls(opt_name, hls_path):
    print("Job Case "+ str(opt_name)+" submitted.")
    process = subprocess.Popen("cd "+str(hls_path).replace('\\', '/')+" && vivado_hls -f "+"./hls_proj.tcl", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        print("[ERROR Case "+str(opt_name)+"]: "+error.decode())

def run_hls_batch(opt_list, gen_path, batch_begin_idx, batch_end_idx):
    for idx in range(batch_begin_idx, batch_end_idx):
        opt_key, _ = opt_list[idx]
        sample_dir = gen_path / str(opt_key)
        if not sample_dir.is_dir():
            print(f"Cannot find the directory: {sample_dir}. Exit this case!")
            continue
        hls_dir = sample_dir / 'hls'
        if not hls_dir.is_dir():
            print(f"Cannot find the directory: {hls_dir}. Exit this case!")
            continue
       
        process = Process(target=run_hls, args=(opt_key, hls_dir))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample-file', help='file name of operator samples')
    parser.add_argument('-g', '--gen-path', help='path for generating samples')
    parser.add_argument('--start-idx', help='index for the first sample', default=0, type=int)
    parser.add_argument('--num', help='number of generated samples', default=-1, type=int)
    parser.add_argument('-b', '--batch-size', help='batch size for generation', default=8, type=int)
    opt = parser.parse_args()
    sample_file = opt.sample_file
    gen_path = pathlib.Path(opt.gen_path).absolute()

    with open(sample_file, 'r') as fdict:
        opt_dicts = json.load(fdict)
    total_num = len(opt_dicts)

    Batch_Size = opt.batch_size
    Begin_Idx = opt.start_idx
    Sample_Num = opt.num
    End_Idx = total_num if (Sample_Num < 0) else Begin_Idx + Sample_Num

    opt_list = list(opt_dicts.items())
    Batch_Num = math.ceil(Sample_Num / Batch_Size)
    period = 0
    for batch in range(0,Batch_Num):
        processes = []
        print(f"Batch {batch} begins. Current time: {str(time.ctime())}")
        batch_begin_idx = Begin_Idx + batch*Batch_Size
        batch_end_idx = min(batch_begin_idx + Batch_Size, End_Idx)

        t1 = time.time()  
        run_hls_batch(opt_list, gen_path, batch_begin_idx, batch_end_idx)
        t2 = time.time()
        print(f"Batch {batch} consisting of {len(processes)} tasks was completed within {(t2 - t1) / 60} minutes. Current time: {str(time.ctime())}")
        period += (t2 - t1)

    print(f"All {Sample_Num} tasks were completed within {period / 60} minutes")
    
