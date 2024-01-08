import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import sys

from Generate_HLS import ConvParam
from mymodel import YOLOLayer
from test import get_prebox, hyp, bbox_iou, select_weight_file
from torch.utils.data import DataLoader

from datasets import LoadImagesAndLabels

def print_ndarray_recursion_outputstream(arr, str_func=str, file=sys.stdout, stop=0):
    if not hasattr(arr, '__iter__') or len(arr.shape) == stop:
        print(str_func(arr), file=file, end='')
        return
    ends = '' if (len(arr.shape)==stop+1) else '\n'
    print('{', file=file, end='')
    for i, item in enumerate(arr):
        print_ndarray_recursion_outputstream(item, str_func, file, stop)
        if i!=len(arr)-1: print(',', file=file, end=ends)
    print(ends+'}', file=file, end='')


class QConvLayer:
    def __init__(self, conv_param):
        self.conv = conv_param
        self.w = torch.tensor(self.conv.w, dtype = torch.int64)
    
    def __call__(self, x):
        if self.conv.icol < x.shape[-1]: # maxpool
            assert self.conv.irow*2, self.conv.icol*2 == x.shape[2:]
            x = F.max_pool2d(x.float(), kernel_size = 2, stride = 2).to(dtype=torch.int64)

        groups = self.conv.groups if hasattr(self.conv, 'groups') else 1
        x = F.conv2d(x, self.w, bias=None, stride=self.conv.s, padding=self.conv.p, groups=groups) # [N, OCH, OROW, OCOL]

        och = x.shape[1]
        if True:
            if self.conv.inc is not None:
                inc_ch = self.conv.inc.reshape((1, och, 1, 1))
                x *= inc_ch
            if hasattr(self.conv, 'bias'):
                bias_ch = self.conv.bias.reshape((1, och, 1, 1))
                x += bias_ch
            if hasattr(self.conv, 'lshift'):
                x += 1 << self.conv.lshift_T-1
                x >>= self.conv.lshift_T

        else: ## no inc/bias quantization
            if self.conv.inc is not None:
                inc_ch = self.conv.inc_raw.reshape((1, och, 1, 1))
                x *= inc_ch
            if hasattr(self.conv, 'bias'):
                bias_ch = self.conv.bias_raw.reshape((1, och, 1, 1))
                x += bias_ch
            x = torch.round(x).to(dtype = torch.int64)
        
        if hasattr(self.conv, 'obit'):
            x.clip_(0, 2**(self.conv.obit)-1)

        with open(output_path + '/CONV_%d_act.txt'%(self.conv.n), 'w') as f:   #activation of each layer
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    print('[%4d,%4d]'%(i,j), end='', file=f)
                    xtemp = x[0,:,i,j]
                    xtemp = xtemp.numpy()
                    xtemp = xtemp.astype(np.int32)
                    xtemp = list(xtemp)
                    # print(','.join(map(hex, xtemp)), file=f)
                    print(','.join(map(str, xtemp)), file=f)

        return x

def reorg(x):
    stride = 2
    B = x.data.size(0)
    C = x.data.size(1)
    H = x.data.size(2)
    W = x.data.size(3)
    ws = stride
    hs = stride
    x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
    x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
    x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
    x = x.view([B, hs*ws*C, H//hs, W//ws])
    return x

class HWModel:
    def __init__(self, model_param):
        self.layers = [QConvLayer(conv_param) for conv_param in model_param]
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yololayer.eval()

    def __call__(self, x):
        assert len(x.shape) == 4 and x.dtype == torch.int64
        img_size = x.shape[-2:]

        if self.layers[0].conv.abit<8: # ImageInputQ
            x=x>>(8-self.layers[0].conv.abit) 

        if not opt.bypass:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        else:
            for i in [0,1,2,3]:
                x = self.layers[i](x)
            p4_in = torch.round(reorg(x) * 
                        self.layers[4].conv.astep / self.layers[7].conv.astep).to(dtype=torch.int64)
            for i in [4,5,6]:
                x = self.layers[i](x)
            x = torch.cat([p4_in, x], 1)
            for i in [7,8]:
                x= self.layers[i](x)
        
        x = x.float() / self.layers[-1].conv.div

        io, p = self.yololayer(x, img_size)
        return io

def testdataset(hwmodel):
    img_size = 320
    dataset = LoadImagesAndLabels(opt.datapath, img_size, opt.batch_size, rect=False, cache_labels=True, hyp=hyp, augment=False)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            #num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    
    iou_sum = 0.0
    test_n = 0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(dataloader):
        if batch_i == opt.num_batch: break
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        imgs = imgs.to(dtype = torch.int64)
        inf_out = hwmodel(imgs)
        pre_box = get_prebox(inf_out)

        tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height])
        ious = bbox_iou(pre_box, tbox)
        iou_sum += ious.sum()

        np.set_printoptions(precision = 2)
        for p in range(len(imgs)):
            print('pbox_xywh', pre_box[p].numpy(), 'tbox_xywh', tbox[p].numpy(), 'iou %.4f'%ious[p].item())
    
        meaniou = iou_sum / test_n

    print('iou', meaniou)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--accelerator', help='accelerator folder name in ./hls/, which contians model_param.pkl')
    parser.add_argument('-bp', '--bypass', action='store_true', help='use bypass model')
    parser.add_argument('-ip', '--input-path', help='path of simulated picture (.bin)')
    parser.add_argument('-op', '--output-path', help='path of output')
    opt = parser.parse_args()
    output_path = opt.output_path
    
    if opt.accelerator is None: opt.accelerator = select_weight_file()
    
    x = torch.zeros([1,3,320,160], dtype=torch.int64)
    hwmodel = HWModel(torch.load('./hls/'+opt.accelerator+'/model_param.pkl'))

    img = np.fromfile(opt.input_path, dtype=np.uint8)
    img = img.reshape(1, 160, 320, -1)
    img = img.transpose(0, 3, 1, 2)
    img_sw = img.copy()
    w0, h0 = 640, 360

    img_t = torch.tensor(img_sw, dtype=torch.int64)

    inf_out = hwmodel(img_t)
    box1 = get_prebox(inf_out)

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b1_x1 = b1_x1 * float(w0) / 320.0
    b1_x2 = b1_x2 * float(w0) / 320.0
    b1_y1 = b1_y1 * float(h0) / 160.0
    b1_y2 = b1_y2 * float(h0) / 160.0

    print("xmin:", b1_x1, " xmax:", b1_x2, " ymin:", b1_y1, " ymax:", b1_y2)
    
    # testdataset(hwmodel)
