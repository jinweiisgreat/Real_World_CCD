import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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

    def compute_proto_aug_hardness_aware_loss(self, model):
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device) # shape: (num_seen_classes, feature_dim) （50，768）

        # hardness-aware sampling
        # 通过hardness-aware sampling机制，使用 sampling_prob 来控制采样概率
        # mean_similarity越高，表示原型之间越相似，采样概率越高，越需要关注这部分。
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        # prototypes_labels = [5,5,6,7,9,11...] (范围是Session t-1的原型个数,即 Seen 的类别数)
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=sampling_prob) # shape: (batch_size*2,) (256,)

        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels] # shape: (batch_size, feature_dim) (256, 768)
        """
        参考ProtoAug论文
        prototypes_sampled 是一个原型矩阵(batch_size,feature_dim)
        prototypes_augmented 是prototypes_sampled经过加强的原型矩阵
        然后用t-1时刻的prototypes_augmented喂给t时刻分类器，以强化seen类的分类边界
        """
        # 从分布中采样
        # prototypes_augmented 包含了 batch_size 个样本
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        # prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        # 这种机制确保了即使在特征空间中偏离中心点，模型仍能保持类别决策边界的稳定性
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss

    def update_prototypes_offline_clip(self, model_func, train_loader, num_labeled_classes):
        """为CLIP特征定制的离线原型更新"""
        all_feats_list = []
        all_labels_list = []

        for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                feats, _ = model_func(images)
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

    def update_prototypes_online_clip(self, model_func, train_loader, num_seen_classes, num_all_classes):
        """为CLIP特征定制的在线原型更新"""
        all_preds_list = []
        all_feats_list = []

        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                feats, logits = model_func(images)
                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 计算新类别原型
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info('No pred of this class, using random initialization...')
                feats_c_mean = torch.randn(self.feature_dim, device=self.device)
            else:
                self.logger.info('computing (predicted) class-wise mean...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新
        self.prototypes = prototypes_all

        # 更新平均相似度
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

    def update_prototypes_online_clip(self, model_func, train_loader, num_seen_classes, num_all_classes):
        """为CLIP特征定制的在线原型更新"""
        all_preds_list = []
        all_feats_list = []

        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                feats, logits = model_func(images)
                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 计算新类别原型
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info('No pred of this class, using random initialization...')
                feats_c_mean = torch.randn(self.feature_dim, device=self.device)
            else:
                self.logger.info('computing (predicted) class-wise mean...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新
        self.prototypes = prototypes_all

        # 更新平均相似度
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity
