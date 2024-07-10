#使用命令行运行脚本，如下所示：
# python calculate_scores_LRS.py --data_root <DATA_ROOT> [OTHER_OPTIONS]
#将 <DATA_ROOT> 替换为包含视频文件的目录的路径。其他可选参数（如 --initial_model、--batch_size 等）可以根据需要添加或省略

#用于计算和评估 Wav2Lip 模型在 Lip Sync Error (LSE) 上的性能，使用 SyncNetInstance 计算视频的口型同步偏移、置信度和最小距离
#参数说明
#--initial_model: SyncNet 模型的初始权重文件的路径。
#--batch_size: 处理视频时的批处理大小。
#--vshift: 考虑的最大音频和视频之间的偏移。
#--data_root: 包含要评估的视频的目录。
#--tmp_dir: 临时目录的路径，用于存储中间结果。
#--reference: 评估的参考模式，默认为 "demo"。
#主要功能
#脚本首先解析命令行参数并加载 SyncNet 模型。
#使用 glob 模块遍历 data_root 目录中的所有视频文件。
#对于每个视频文件，使用 SyncNetInstance (SyncNetInstance_calc_scores.py 中定义) 计算偏移、置信度和最小距离。
#使用 tqdm 来显示进度条，并计算所有视频的平均置信度和平均最小距离。

#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm

from SyncNetInstance_calc_scores import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_root', type=str, required=True, help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);
path = os.path.join(opt.data_root, "*.mp4")

all_videos = glob.glob(path)

prog_bar = tqdm(range(len(all_videos)))
avg_confidence = 0.
avg_min_distance = 0.


for videofile_idx in prog_bar:
	videofile = all_videos[videofile_idx]
	offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
	avg_confidence += confidence
	avg_min_distance += min_distance
	prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / (videofile_idx + 1), 3), round(avg_min_distance / (videofile_idx + 1), 3)))
	prog_bar.refresh()

print ('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
print ('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))



