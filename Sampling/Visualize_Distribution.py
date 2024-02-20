from random import choice, sample
import json
import pathlib
import math
import numpy as np
import argparse
from Sampling import attr_names, Opt_Sampling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_name', help='file name of operator samples')
	parser.add_argument('-a', '--attr', help='attributes', choices=attr_names)
	parser.add_argument('-n', '--num', help='number of samples', default=-1)
	opt = parser.parse_args()

	attr_key = str(opt.attr)

	with open(opt.file_name, 'r') as fdict:
	    opt_dicts = json.load(fdict)

	num = int(opt.num)
	
	Packing_stat = []
	DW_stat = []
	LUT_stat = []
	lat_stat = []
	attr_stat = []
	para_factor = []
	for idx, (k, opt) in enumerate(opt_dicts.items()):
		if num >= 0 and idx >= num:
			break
		packing_bool = opt['Type']['Packing'] == 'FP'
		Packing_stat.append(packing_bool)
		DW_stat.append(opt['Type']['DW'])
		LUT_stat.append(opt['Type']['LUT'])
		lat_stat.append(opt['Type']['Latency'])
		para_factor.append(opt['Config']['w_sep'] * opt['Config']['a_sep'] * opt['Config']['simd'] * opt['Config']['pe'] * opt['Config']['kpf'])
		if attr_key in opt['Config'].keys():
			attr_stat.append(opt['Config'][attr_key])

	print(f'Collect {len(attr_stat)} samples!')

	sns.set()                                   #设置seaborn默认格式
	from matplotlib.font_manager import FontProperties   #显示中文，并指定字体
	myfont=FontProperties(fname=r'C:/Windows/Fonts/simhei.ttf',size=14)
	sns.set(font=myfont.get_name())
	plt.rcParams['axes.unicode_minus']=False      #显示负号
	
	plt.rcParams['figure.figsize'] = (16, 12)    #设定图片大小
	f = plt.figure()                            #确定画布

	f.add_subplot(2,3,1)
	sns.distplot(Packing_stat, kde=False)                 #绘制频数直方图
	plt.ylabel("Number of Samples", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'Packing stat, {len(Packing_stat)} samples', fontsize=20)             #设置子图标题

	f.add_subplot(2,3,2)
	sns.distplot(DW_stat, kde=False)                 #绘制频数直方图
	plt.ylabel("Number of Samples", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'DW stat, {len(DW_stat)} samples', fontsize=20)             #设置子图标题

	f.add_subplot(2,3,3)
	sns.distplot(LUT_stat, kde=False)                 #绘制频数直方图
	plt.ylabel("Number of Samples", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'LUT stat, {len(LUT_stat)} samples', fontsize=20)             #设置子图标题

	f.add_subplot(2,3,4)
	sns.distplot(lat_stat, kde=False)                 #绘制频数直方图
	plt.ylabel("Number of Samples", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'Latency stat, {len(lat_stat)} samples', fontsize=20)             #设置子图标题

	f.add_subplot(2,3,5)
	sns.distplot(para_factor, kde=False)                 #绘制频数直方图
	plt.ylabel("para_factor", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'PF stat, {len(para_factor)} samples', fontsize=20)             #设置子图标题

	f.add_subplot(2,3,6)
	sns.distplot(attr_stat, kde=False)                 #绘制频数直方图
	plt.ylabel("Number of Samples", fontsize=16)
	plt.xticks(fontsize=16)                    #设置x轴刻度值的字体大小
	plt.yticks(fontsize=16)                   #设置y轴刻度值的字体大小
	plt.title(f'{attr_key} stat, {len(attr_stat)} samples', fontsize=20)             #设置子图标题

	plt.show()