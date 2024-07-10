#FaceAlignment 类通常作为更大的人脸处理系统的一部分来使用

#FaceAlignment 类，用于检测图像中的人脸并定位其特征点（landmarks）。
#FaceAlignment 类是人脸检测和特征点定位的主要接口。它使用一个预训练的神经网络来检测人脸并定位特征点。类的主要功能和属性包括：
#初始化 (__init__): 在初始化时，类接收几个参数来设置模型：
#landmarks_type：定义要检测的特征点类型（2D, 2.5D, 3D）。
#network_size：定义网络模型的大小。
#device：指定运行模型的设备（例如 'cuda' 或 'cpu'）。
#flip_input：是否在处理图像时翻转输入。
#face_detector：使用的人脸检测器类型。
#verbose：是否打印额外的信息。
#人脸检测 (get_detections_for_batch): 此方法接受一批图像，并返回检测到的人脸的位置。对于每张图像，返回一个矩形框（x1, y1, x2, y2），表示人脸的位置。

from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results