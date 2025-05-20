import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
from torch.utils.data import DataLoader

def compute_energy_scores(model, data_loader, temperature=1.0, device='cuda'):
    """计算所有样本的能量分数"""
    model.eval()
    energy_scores = []
    all_features = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing energy scores"):
            images, _, _, _ = batch  # 解包batch数据
            images = images[0].to(device) if isinstance(images, list) else images.to(device)  # 处理多视图情况

            # 提取特征和logits
            features = model.backbone(images)
            features = F.normalize(features, dim=1)
            logits = model.projector.last_layer(features)

            # 计算能量分数：-logsumexp(logits/temperature)
            batch_energy = -torch.logsumexp(logits / temperature, dim=1)

            energy_scores.append(batch_energy.cpu())
            all_features.append(features.cpu())
            all_logits.append(logits.cpu())

    energy_scores = torch.cat(energy_scores)
    all_features = torch.cat(all_features)
    all_logits = torch.cat(all_logits)

    return energy_scores, all_features, all_logits


def determine_energy_threshold(energy_scores, percentile=90):
    """使用百分位数确定能量阈值"""
    threshold = torch.quantile(energy_scores, percentile / 100.0)
    return threshold

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.iter = 0

    @torch.no_grad()
    def forward(self, logits):
        # Q = torch.exp(logits / self.epsilon).t()
        Q = torch.exp(logits).t()
        B = Q.shape[0]
        K = Q.shape[1]  # how many prototypes
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()