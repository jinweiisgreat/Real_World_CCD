import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class ProtoAugManager:
    """
    支持增强backbone的原型增强管理器
    """

    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None  # 每个原型的平均相似度，用于hardness-aware sampling
        self.hardness_temp = hardness_temp  # 温度参数用于计算hardness-aware sampling的softmax概率
        self.radius = 0
        self.radius_scale = radius_scale
        self.logger = logger

    def save_proto_aug_dict(self, save_path):
        """保存原型增强字典"""
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
        }
        torch.save(proto_aug_dict, save_path)

    def load_proto_aug_dict(self, load_path):
        """加载原型增强字典"""
        proto_aug_dict = torch.load(load_path)
        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']

    def compute_proto_aug_hardness_aware_loss(self, model):
        """
        计算hardness-aware的原型增强损失

        Args:
            model: 完整模型 [enhanced_backbone, projector]

        Returns:
            proto_aug_loss: 原型增强损失
        """
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)  # shape: (num_seen_classes, feature_dim)

        # hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]  # shape: (batch_size, feature_dim)

        # 从分布中采样增强原型
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                device=self.device) * self.radius * self.radius_scale

        # 通过模型获取logits
        # 对于增强的backbone，我们需要创建适当的输入格式
        if hasattr(model[0], 'base_vit'):
            # 对于增强的backbone，直接使用基础特征（不通过prompts）
            # 因为原型增强的目的是维持已见类的边界，不应该受到prompts影响

            # 方法1: 直接通过projector（推荐）
            _, prototypes_output = model[1](prototypes_augmented)

        else:
            # 普通backbone的处理方式
            _, prototypes_output = model[1](prototypes_augmented)

        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss

    def update_prototypes_offline(self, model, prompts_enhancer, train_loader, num_labeled_classes):
        """
        离线阶段更新原型

        Args:
            model: 模型 [backbone, projector]
            prompts_enhancer: PromptsEnhancer实例（可能为None）
            train_loader: 训练数据加载器
            num_labeled_classes: 标记类别数量
        """
        model.eval()
        if prompts_enhancer is not None:
            prompts_enhancer.eval()

        all_feats_list = []
        all_labels_list = []

        # 前向传播获取数据
        for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                # 获取基础特征（不使用prompts增强）
                if hasattr(model[0], 'base_vit'):
                    # 使用基础ViT获取纯净特征用于原型计算
                    feats = model[0].base_vit(images)  # 基础backbone特征
                else:
                    feats = model[0](images)  # 普通backbone

                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_labels_list.append(label)

        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # 计算原型和半径
        prototypes_list = []
        radius_list = []
        for c in range(num_labeled_classes):
            feats_c = all_feats[all_labels == c]
            feats_c_mean = torch.mean(feats_c, dim=0)  # feats_c_mean.shape: (768,)
            prototypes_list.append(feats_c_mean)
            feats_c_center = feats_c - feats_c_mean  # 特征向量减去原型（中心化）
            cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)  # 计算协方差矩阵
            radius = torch.trace(cov) / self.feature_dim  # 协方差矩阵的平均特征值
            radius_list.append(radius)

        avg_radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
        prototypes_all = torch.stack(prototypes_list, dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)  # shape: (num_labeled_classes, feature_dim)

        # 更新
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # 更新每个原型的平均相似度
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

    def update_prototypes_online(self, model, prompts_enhancer, train_loader, num_seen_classes, num_all_classes):
        """
        在线阶段更新原型

        Args:
            model: 当前模型
            prompts_enhancer: PromptsEnhancer实例（可能为None）
            train_loader: 训练数据加载器
            num_seen_classes: 已见类别数量
            num_all_classes: 所有类别数量
        """
        model.eval()
        if prompts_enhancer is not None:
            prompts_enhancer.eval()

        all_preds_list = []
        all_feats_list = []

        # 前向传播获取数据
        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                # 使用完整模型获取预测（包括prompts增强）
                _, logits = model(images)
                all_preds_list.append(logits.argmax(1))

                # 获取基础特征用于原型计算
                if hasattr(model[0], 'base_vit'):
                    # 使用基础ViT获取纯净特征
                    feats = model[0].base_vit(images)
                else:
                    feats = model[0](images)

                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 计算新类别的原型
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info('No pred of this class, using fc (last_layer) parameters...')
                # 使用分类器权重作为原型
                feats_c_mean = model[1].last_layer.weight_v.data[c]
            else:
                self.logger.info('computing (predicted) class-wise mean...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新
        self.prototypes = prototypes_all

        # 更新每个原型的平均相似度
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity