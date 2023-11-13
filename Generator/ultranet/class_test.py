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
import mymodel
from utils.view_pt import select_weight_file
from quant_dorefa import activation_quantize_fn
from quant_module import HWGQ, QuantConv2d, ImageInputQ
from Front_Back import get_front, get_back
from Opt_Templates import Gen_Opt_Templates

Opt_list = [1, 2, 3, 4, 5, 6, 7]
Opt_list_KRowP = []
Opt_list_KP = []
Opt_list_KP_KRowP = [0]


class ConvParam:
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__=='__main__':
    conv_cur = ConvParam(1, 2)
    conv_cur.c = 9

    print(conv_cur.c)
    # conv_cur.a = "eee"
    # conv_cur.dic = {'a': 2, 'b': 78}
    #
    # print(conv_cur.dic)
    # print(conv_cur.a)