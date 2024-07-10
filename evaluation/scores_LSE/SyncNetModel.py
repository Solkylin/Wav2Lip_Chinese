#非原项目有的文件，因为SyncNetInstance_calc_scores.py中有from SyncNetModel import *的语句而原项目不含SyncNetModel.py
#所以在github上找到了某个开源项目中的该同名文件，恰好该文件有另一个报错中关于S的定义

#辅助模块

#定义了一个用于语音与嘴唇运动同步检测的 SyncNet 模型
#SyncNet 模型（S 类）
#S 类 一个 PyTorch 模型类，用于定义 SyncNet 网络结构。包含用于处理音频和视频输入的两个主要部分：

#音频网络 (netcnnaud 和 netfcaud)
#netcnnaud: 音频输入的卷积神经网络层，用于提取音频特征。
#netfcaud: 全连接层，用于进一步处理提取的音频特征。
#嘴唇运动网络 (netcnnlip 和 netfclip)
#netcnnlip: 用于处理视频输入的 3D 卷积神经网络层，专注于提取嘴唇运动特征。
#netfclip: 全连接层，用于进一步处理提取的嘴唇运动特征。
#SyncNet 类的方法
#forward_aud: 该方法接收音频输入并通过音频网络部分进行处理，最终输出音频特征。
#forward_lip: 接收视频输入并通过嘴唇运动网络部分进行处理，最终输出嘴唇运动特征。
#forward_lipfeat: 专门用于提取视频帧中的嘴唇特征，用于某些应用场景。
#辅助函数
#save: 保存模型到指定文件。
#load: 从指定文件加载模型。

#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

def save(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f);
        print("%s saved."%filename);

def load(filename):
    net = torch.load(filename)
    return net;
    
class S(nn.Module):
    def __init__(self, num_layers_in_fc_layers = 1024):
        super(S, self).__init__();

        self.__nFeatures__ = 24;
        self.__nChs__ = 32;
        self.__midChs__ = 32;

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        );

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        );

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        );

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        );

    def forward_aud(self, x):

        mid = self.netcnnaud(x); # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfcaud(mid);

        return out;

    def forward_lip(self, x):

        mid = self.netcnnlip(x); 
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfclip(mid);

        return out;

    def forward_lipfeat(self, x):

        mid = self.netcnnlip(x);
        out = mid.view((mid.size()[0], -1)); # N x (ch x 24)

        return out;