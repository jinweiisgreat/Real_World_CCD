import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model_utils import *


'''
2024-05-03
prototype augmentation utils, including:
* prototype augmentation loss functions
* save prototypes for the offline session (according to ground-truth labels)
* save prototypes for each online continual session (according to pseudo-labels)

update 1: [2024-05-04] hardness-aware prototype sampling
'''


class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None   # NOTE!!! mean similarity of each prototype, for hardness-aware sampling
        self.hardness_temp = hardness_temp   # NOTE!!! temperature to compute mean similarity to softmax prob for hardness-aware sampling
        self.radius = 0
        self.radius_scale = radius_scale
        self.logger = logger


    def save_proto_aug_dict(self, save_path):
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
        }

        torch.save(proto_aug_dict, save_path)

    # load continual   # NOTE!!!
    def load_proto_aug_dict(self, load_path):
        proto_aug_dict = torch.load(load_path)

        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']


    def compute_proto_aug_loss(self, model):
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)
        prototypes_labels = torch.randint(0, len(prototypes), (self.batch_size,)).to(self.device)   # dtype=torch.long
        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        #prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss

    # 在每个在线阶段(Online Session)中，使用上一阶段保存的原型和难度分布进行采样

    # 为ProtoAug Manager添加CLIP支持的方法
    def compute_proto_aug_hardness_aware_loss_clip(self, projector):
        """为CLIP特征定制的ProtoAug损失计算"""
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # 难度感知采样
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True,
                                             p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                device=self.device) * self.radius * self.radius_scale

        # 通过投影头获取logits
        _, prototypes_output = projector(prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss

    def update_prototypes_offline_clip(self, clip_model, weighted_gamma, train_loader, num_labeled_classes):
        """专门为CLIP设计的原型更新方法"""
        clip_model.eval()

        all_feats_list = []
        all_labels_list = []

        for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                # 直接调用CLIP特征提取
                feats = get_clip_features_and_fusion(clip_model, weighted_gamma, images)
                all_feats_list.append(feats)
                all_labels_list.append(label)

        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # 计算原型和半径
        prototypes_list = []
        radius_list = []
        for c in range(num_labeled_classes):
            feats_c = all_feats[all_labels == c]
            feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)
            feats_c_center = feats_c - feats_c_mean
            cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)
            radius = torch.trace(cov) / self.feature_dim
            radius_list.append(radius)

        avg_radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
        prototypes_all = torch.stack(prototypes_list, dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # 更新平均相似度
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

    def update_prototypes_online_clip(self, clip_model, projector, weighted_gamma, train_loader, num_seen_classes,
                                      num_all_classes):
        """专门为CLIP设计的在线原型更新方法"""
        clip_model.eval()
        projector.eval()
        weighted_gamma.eval()

        all_preds_list = []
        all_feats_list = []

        # 前向传播获取特征和预测
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Computing prototypes")):
            # 处理不同的数据加载器格式
            if len(batch) == 4:
                images, label, _, _ = batch
            else:
                images, label, _ = batch[:3]

            images = images.cuda(non_blocking=True)

            with torch.no_grad():
                # 获取CLIP融合特征
                feats = get_clip_features_and_fusion(clip_model, weighted_gamma, images)
                # 获取分类预测
                _, logits = projector(feats)

                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 计算新类别的原型
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info(f'No predictions for class {c}, using classifier weights...')
                # 使用分类器的权重作为原型
                feats_c_mean = projector.last_layer.weight_v.data[c]
            else:
                self.logger.info(f'Computing mean for class {c} with {len(feats_c)} samples...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        # 合并旧原型和新原型
        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新原型
        self.prototypes = prototypes_all

        # 更新原型间的平均相似度（用于难度感知采样）
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]  # 移除对角线元素
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity



def get_clip_features_and_fusion(clip_model, weighted_gamma, images):
    """获取CLIP特征并进行融合"""
    # 移除torch.no_grad()，允许梯度回传
    all_img_feats, all_txt_feats = clip_model(images)
    all_img_feats = all_img_feats.float()
    all_txt_feats = all_txt_feats.float()
    fusion_feat = weighted_gamma(all_img_feats, all_txt_feats)
    return fusion_feat