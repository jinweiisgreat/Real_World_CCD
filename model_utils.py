import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from loguru import logger
from torch.nn import functional as F
from config import *

class WeightedGamma(nn.Module):
    """用于融合图像和文本特征的加权模块"""

    def __init__(self, args):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.5))  # 可学习参数

    def forward(self, img_feats, txt_feats):
        # 融合图像和文本特征
        fusion_feat = self.gamma * img_feats + (1 - self.gamma) * txt_feats
        return F.normalize(fusion_feat, dim=-1)

class WeightedGamma_clip(nn.Module):
    def __init__(self, args):
        super(WeightedGamma, self).__init__()

        self.weight1 = nn.Parameter(torch.ones(1, 1).cuda())
        self.weight2 = nn.Parameter(torch.ones(1, 1).cuda())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = x.cuda()
        y = y.cuda()

        '''
        加权，获得融合特征
        '''
        w1 = self.sigmoid(self.weight1)
        w2 = self.sigmoid(self.weight2)

        gamma = w1 / (w1 + w2)
        print('gamma:', gamma)

        return (1 - gamma) * x + gamma * y
        # return x


class AverageFusion(nn.Module):
    def __init__(self):
        super(AverageFusion, self).__init__()
        # 不需要额外的参数或激活函数
        self.gamma = None  # AverageFusion不需要gamma参数，这里占一个位，方便后面的代码统一处理

    def forward(self, x, y):
        # 确保输入在CUDA设备上
        x = x.cuda()
        y = y.cuda()

        # 进行平均融合
        fused = (x + y) / 2
        return fused


class ConcatFusion(nn.Module):
    def __init__(self, input_dim):
        super(ConcatFusion, self).__init__()
        self.input_dim = input_dim
        self.gamma = None
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.input_dim),
            nn.ReLU()  # 可以根据需要更换为其他激活函数
        ).cuda()

    def forward(self, x, y):
        # 确保输入在CUDA设备上（根据用户的设备需求）
        x = x.float().cuda()  # (128,768) batch_size, input_dim
        y = y.float().cuda()
        # 拼接特征
        concatenated = torch.cat((x, y), dim=1)  # 维度为(batch_size, 2 * input_dim)
        # 通过全连接层降维
        fused = self.fusion(concatenated)

        return fused