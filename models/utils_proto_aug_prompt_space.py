import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.prompt_enhanced_model import PromptEnhancedModel


'''
参考Spacing Loss
借助ProtoAugManager原型系统，分离原型之间的距离，维护Seen Novel决策边界
'''


class ProtoAugSpacingManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger,
                 spacing_alpha=1.2, spacing_weight=1.0):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None
        self.hardness_temp = hardness_temp
        self.radius = 0
        self.radius_scale = radius_scale
        self.logger = logger

        # Spacing Loss 相关参数
        self.spacing_alpha = spacing_alpha  # 距离放大因子
        self.spacing_weight = spacing_weight  # spacing loss权重
        self.equidistant_points = None  # 等距点缓存

    def save_proto_aug_dict(self, save_path):
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
            'equidistant_points': self.equidistant_points,
            'spacing_alpha': self.spacing_alpha,
        }
        torch.save(proto_aug_dict, save_path)

    def load_proto_aug_dict(self, load_path):
        proto_aug_dict = torch.load(load_path)
        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']
        self.equidistant_points = proto_aug_dict.get('equidistant_points', None)
        self.spacing_alpha = proto_aug_dict.get('spacing_alpha', 1.2)

    def compute_equidistant_points(self):
        """计算等距点，只在prototypes更新时调用"""
        if self.prototypes is None or len(self.prototypes) < 2:
            return None

        prototypes = F.normalize(self.prototypes, dim=-1, p=2)
        num_classes = len(prototypes)

        # 计算最大距离
        distances = torch.cdist(prototypes, prototypes)
        max_distance = distances.max().item()
        target_distance = self.spacing_alpha * max_distance

        # 构建目标距离矩阵
        target_distances = torch.full((num_classes, num_classes), target_distance, device=self.device)
        target_distances.fill_diagonal_(0)

        # 简化的等距点计算：使用MDS的近似方法
        equidistant_points = self._approximate_mds(target_distances, prototypes)

        return equidistant_points

    def _approximate_mds(self, target_distances, initial_points):
        """简化的多维标定，迭代优化"""
        points = initial_points.clone()
        num_classes = len(points)

        # 简单的迭代优化
        for _ in range(10):  # 限制迭代次数
            current_distances = torch.cdist(points, points)

            # 计算梯度方向（简化版）
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    current_dist = current_distances[i, j]
                    target_dist = target_distances[i, j]

                    if current_dist > 0:
                        direction = (points[i] - points[j]) / current_dist
                        adjustment = 0.1 * (target_dist - current_dist) * direction
                        points[i] += adjustment / 2
                        points[j] -= adjustment / 2

            # 重新归一化
            points = F.normalize(points, dim=-1, p=2)

        return points

    def compute_spacing_loss(self):
        """计算spacing loss，直接使用缓存的等距点"""
        if self.equidistant_points is None:
            return torch.tensor(0.0, device=self.device)

        prototypes_norm = F.normalize(self.prototypes, dim=-1, p=2)
        spacing_loss = F.mse_loss(prototypes_norm, self.equidistant_points)

        return spacing_loss

    def compute_proto_aug_hardness_aware_loss(self, model):
        """计算综合的prototype augmentation损失，包含spacing loss"""
        if self.prototypes is None:
            return torch.tensor(0.0, device=self.device)

        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # 原始的hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                device=self.device) * self.radius * self.radius_scale

        # 根据模型类型选择合适的forward方式
        enhanced_prototypes, _ = model.prompt_pool.forward(prototypes_augmented)
        _, prototypes_output = model.projector(enhanced_prototypes)

        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        # 添加spacing loss
        spacing_loss = self.compute_spacing_loss()

        # 组合损失
        total_loss = proto_aug_loss + self.spacing_weight * spacing_loss

        return total_loss

    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        """在离线阶段更新prototypes，并计算等距点"""
        model.eval()

        all_feats_list = []
        all_labels_list = []

        model.enable_prompt_learning()

        # forward data
        for batch_idx, (images, label, _) in enumerate(
                tqdm(train_loader, desc="Extracting features for prototypes")):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                feats = model.backbone(images)  # 直接使用backbone
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_labels_list.append(label)

        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # 计算prototypes和radius
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

        # 更新状态
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # 更新mean similarity
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

        # 计算等距点
        self.equidistant_points = self.compute_equidistant_points()

        self.logger.info(f"Updated prototypes: {len(prototypes_list)} classes, radius: {avg_radius:.4f}")
        if self.equidistant_points is not None:
            self.logger.info(f"Computed equidistant points with spacing_alpha: {self.spacing_alpha}")

    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        """在线阶段更新prototypes，并重新计算等距点"""
        model.eval()
        model.enable_prompt_learning()

        all_preds_list = []
        all_feats_list = []

        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader, desc="Updating online prototypes")):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                _, logits = model(images)
                feats = model.backbone(images)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 计算新类别的prototypes
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info(f'No pred of class {c}, using fc parameters...')
                feats_c_mean = model.projector.last_layer.weight_v.data[c]
            else:
                self.logger.info(f'Computing class-wise mean for class {c}...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新状态
        self.prototypes = prototypes_all

        # 更新mean similarity
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

        # 重新计算等距点
        self.equidistant_points = self.compute_equidistant_points()

        self.logger.info(f"Updated prototypes: {len(prototypes_all)} classes total")
        if self.equidistant_points is not None:
            self.logger.info(f"Recomputed equidistant points for {len(prototypes_all)} classes")
