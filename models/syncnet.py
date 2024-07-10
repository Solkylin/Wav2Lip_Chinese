#SyncNet_color 是一个模型类，它本身不直接运行。要使用这个模型，你需要创建一个实例并将数据传递给它。例如：
# 创建模型实例
#syncnet = SyncNet_color()
# 假设 audio_data 和 face_data 是已经预处理好的音频和面部数据
#audio_embedding, face_embedding = syncnet(audio_data, face_data)
#通常作为更大系统（如 Wav2Lip）的一部分来使用，用于音视频同步任务。需要有适当的数据预处理和后处理步骤，以及整合到整体的训练和推理流程中。

#脚本定义了 SyncNet_color 类，这是一个神经网络模型，用于同步音频和视频流。它包含两个主要部分：面部编码器和音频编码器。

#面部编码器是一个卷积神经网络，用于从输入的面部序列中提取特征。它包括多层 Conv2d（自定义的卷积层），其中每层都有不同的配置（如核大小、步长、填充和是否使用残差连接）。

#音频编码器也是一个卷积神经网络，用于从输入的音频序列中提取特征。它的结构类似于面部编码器，但是针对音频数据进行了优化。

#前向传播 (forward) 方法
#forward 方法接收两个参数：audio_sequences（音频序列）和 face_sequences（面部序列）。
#使用面部编码器和音频编码器对输入数据进行特征提取。
#提取的特征被展平（view 方法）并进行归一化（使用 F.normalize）。
#返回音频嵌入和面部嵌入，这些嵌入可用于进一步的处理或训练。

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2d

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
