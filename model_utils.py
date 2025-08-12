import argparse
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import vision_transformer as vits
from loguru import logger
from torch.nn import functional as F
from config import *


class WeightedGamma(nn.Module):
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


'''
确认model里面的参数情况
'''


def check_parms(model):
    for name, parms in model.named_parameters():
        print(f'{name} : {parms}')


'''
确认model里面的的梯度回传情况
'''


def check_is_grad(model, optim_params=None):
    if optim_params != None:
        for name, parms in model.named_parameters():
            # print(f'-->name:{name} -->grad_requirs:{parms.requires_grad}')
            # print(f'-->weight{torch.mean(parms.data)} -->grad_value:{torch.mean(parms.grad)}')
            if name in optim_params:
                # print(f'-->weight{parms.data} -->grad_value:{parms.grad}')
                print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')
    else:
        for name, parms in model.named_parameters():
            print(f'-->name:{name} -->grad_requirs:{parms.requires_grad} -->grad_value:{parms.grad}')


'''
用于根据参数名，修改CLIP的image encoder中的参数“是否要回传梯度”
'''


def change_need_grad(model, optim_params=None):
    for name, parms in model.named_parameters():
        if name in optim_params:
            parms.requires_grad = True
        else:
            parms.requires_grad = False  # 名字不在array中的全部参数冻结


'''
用于确认CLIP的image encoder哪些参数要回传梯度
'''


def get_optim_params(model_name: str):
    return ['visual.transformer.resblocks.23.attn.in_proj_weight',
            'visual.transformer.resblocks.23.attn.in_proj_bias',
            'visual.transformer.resblocks.23.attn.out_proj.weight',
            'visual.transformer.resblocks.23.attn.out_proj.bias',
            'visual.transformer.resblocks.23.ln_1.weight',
            'visual.transformer.resblocks.23.ln_1.bias',
            'visual.transformer.resblocks.23.mlp.c_fc.weight',
            'visual.transformer.resblocks.23.mlp.c_fc.bias',
            'visual.transformer.resblocks.23.mlp.c_proj.weight',
            'visual.transformer.resblocks.23.mlp.c_proj.bias',
            'visual.transformer.resblocks.23.ln_2.weight',
            'visual.transformer.resblocks.23.ln_2.bias']


'''
用于获得第二阶段训练完的模型
'''
# def get_pretrained_model(state_path,weighted_path,projector_path,args):
#     state_dict = torch.load(state_path, map_location='cpu')
#     weighted = torch.load(weighted_path)
#     model = torch.load(pretrained_model_path)
#     # state_dict = torch.load(state_dict['model'])
#     model.load_state_dict(state_dict['model'])

#     projector = torch.load(projector_path)

#     # optimizer = SGD(clip_model.model.parameters(), lr=args.lr, momentum=args.momentum)

#     #optimizer这里的第一个参数，需要和当时那次训练的参数一致，否则下面load_state_dict会报错
#     optimizer = SGD(list(clip_model.model.parameters()) + list(projector.parameters()), lr=args.lr, momentum=args.momentum)

#     optimizer.load_state_dict(state_dict['optimizer'])

#     epoch = state_dict['epoch']

#     return model,weighted,projector,optimizer,epoch


'''
与数字和名字对应，这些具体内容存在config.py中
'''
get_dataset_num = {
    'cifar10': pretrained_cifar10_num,
    'cifar100': pretrained_cifar100_num,
    'imagenet_100': pretrained_imagenet_100_num,
    'cub': pretrained_cub_num,
    'aircraft': pretrained_cub_num
}
get_dataset_name = {
    'cifar10': pretrained_cifar10_name,
    'cifar100': pretrained_cifar100_name,
    'imagenet_100': pretrained_imagenet_100_name,
    'cub': pretrained_cub_name,
    'aircraft': pretrained_cub_name
}

'''
用于在train.py里面获取第一阶段预训练的路径
'''


def get_pretrained_model(args):
    pretrained_model_num = get_dataset_num[args.dataset_name]
    pretrained_model_name = get_dataset_name[args.dataset_name]
    return pretrained_save_path + pretrained_model_num + pretrained_model_name