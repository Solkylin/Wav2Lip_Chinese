#使用命令行运行脚本，如下所示：
# python calculate_scores_real_videos.py --data_dir <DATA_DIR> [OTHER_OPTIONS]
#将 <DATA_DIR> 替换为包含视频文件的目录的路径。其他可选参数（如 --initial_model、--batch_size 等）可以根据需要添加或省略

#用于计算和评估 Wav2Lip 模型在实际视频数据集上的 Lip Sync Error (LSE)，主要使用 SyncNetInstance来计算视频的口型同步偏移、置信度和距离分数
#参数说明
#--initial_model: SyncNet 模型的初始权重文件的路径。
#--batch_size: 处理视频时的批处理大小。
#--vshift: 考虑的最大音频和视频之间的偏移。
#--data_dir: 包含视频文件和临时文件的目录。
#--videofile: 要评估的单个视频文件的路径（如果指定）。
#--reference: 参考模式或标识，用于选择特定的视频文件。
#主要功能
#脚本首先解析命令行参数。
#加载 SyncNet 模型参数。
#获取 crop_dir 目录下特定参考模式的视频文件列表。
#遍历每个视频文件，并使用 SyncNetInstance 计算偏移、置信度和距离。
#输出每个视频文件的距离分数和置信度到控制台。

#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from SyncNetInstance_calc_scores import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);

flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
flist.sort()

# ==================== GET OFFSETS ====================

dists = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt,videofile=fname)
    print (str(dist)+" "+str(conf))
      
# ==================== PRINT RESULTS TO FILE ====================

#with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
#    pickle.dump(dists, fil)
