import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from models.prompt_enhanced_model import PromptEnhancedModel


'''
参考Spacing Loss
借助ProtoAugManager原型系统，分离原型之间的距离，维护Seen Novel决策边界
'''


class ProtoAugSpacingManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger,
                 spacing_alpha = None, spacing_weight=1.0, spacing_momentum=1.0):
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
        self.spacing_momentum = spacing_momentum
        self.equidistant_points = None  # 等距点缓存

        # 用于跟踪每个类别的访问频率（spacing loss论文中的v）
        self.class_visit_counts = None

    def save_proto_aug_dict(self, save_path):
        """保存所有状态"""
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
            'equidistant_points': self.equidistant_points,
            'spacing_alpha': self.spacing_alpha,
            'class_visit_counts': self.class_visit_counts,
        }
        torch.save(proto_aug_dict, save_path)

    def load_proto_aug_dict(self, load_path):
        """加载所有状态"""
        proto_aug_dict = torch.load(load_path)
        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']
        self.equidistant_points = proto_aug_dict.get('equidistant_points', None)
        self.spacing_alpha = proto_aug_dict.get('spacing_alpha', 1.2)
        self.class_visit_counts = proto_aug_dict.get('class_visit_counts', None)

    def compute_equidistant_points(self):
        """计算等距点，使用论文中的方法"""
        if self.prototypes is None or len(self.prototypes) < 2:
            return None

        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)
        num_classes = len(prototypes)

        # 计算最大距离 (论文中的pdist)
        distances = torch.cdist(prototypes, prototypes)
        max_distance = distances.max().item()
        target_distance = self.spacing_alpha * max_distance

        # 构建目标距离矩阵Δ
        target_distances = torch.full((num_classes, num_classes), target_distance, device=self.device)
        target_distances.fill_diagonal_(0)

        # 使用简化的MDS计算等距点
        equidistant_points = self._approximate_mds(target_distances, prototypes)

        return equidistant_points

    def _approximate_mds(self, target_distances, initial_points):
        """简化的MDS实现"""
        points = initial_points.clone()
        num_classes = len(points)

        # 简单的迭代优化
        for _ in range(10):  # 限制迭代次数
            current_distances = torch.cdist(points, points)

            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    current_dist = current_distances[i, j]
                    target_dist = target_distances[i, j]

                    if current_dist > 0:
                        direction = (points[i] - points[j]) / current_dist
                        adjustment = 0.1 * (target_dist - current_dist) * direction
                        points[i] += adjustment / 2
                        points[j] -= adjustment / 2

            points = F.normalize(points, dim=-1, p=2)

        return points

    def compute_spacing_loss(self, features, model=None):
        """
        按照论文Algorithm 2实现的Spacing Loss

        Args:
            features: 当前batch的特征 [batch_size, feature_dim]
            model: 用于更新特征提取器的模型（可选）

        Returns:
            spacing_loss: spacing损失
            assignment_info: 分配信息（用于调试）
        """
        if self.equidistant_points is None or self.prototypes is None:
            return torch.tensor(0.0, device=self.device), None

        features_norm = F.normalize(features, dim=-1, p=2)
        prototypes_norm = F.normalize(self.prototypes, dim=-1, p=2) # [num_prototypes, feature_dim]
        equidistant_norm = F.normalize(self.equidistant_points, dim=-1, p=2)

        # 一次性计算所有相似度（用于筛选和后续CE Loss）
        similarities = features_norm @ prototypes_norm.T  # [batch_size, num_prototypes]
        max_similarities, _ = similarities.max(dim=1)  # [batch_size]

        # 筛选seen类样本
        similarity_threshold = 0.9
        seen_mask = max_similarities > similarity_threshold

        if seen_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), None

        features_norm = features_norm[seen_mask]
        similarities = similarities[seen_mask]

        # Step 1: 动态分配 - 将每个特征分配给最近的原型 (Algorithm 2, Line 7)
        distances = torch.cdist(features_norm, prototypes_norm)  # [batch_size, num_prototypes]
        assignments = distances.argmin(dim=1)  # [batch_size]
        # assignments = similarities.argmax(dim=1)


        # Step 2: 计算特征到原型的损失 (Algorithm 2, Line 8)
        assigned_prototypes = prototypes_norm[assignments]  # [batch_size, feature_dim]
        feature_to_prototype_loss = F.mse_loss(features_norm, assigned_prototypes) # feature_norm/assigned_prototypes.shape: [256,768]
        # CELoss
        # feature_to_prototype_loss = F.cross_entropy(similarities / 0.1, assignments)

        # Step 3: 更新原型，向"特征+等距点"组合移动 (Algorithm 2, Line 15)
        updated_prototypes = prototypes_norm.clone()

        # 初始化访问计数（如果还没有）
        if self.class_visit_counts is None:
            self.class_visit_counts = torch.zeros(len(self.prototypes), device=self.device)

        unique_assignments = torch.unique(assignments) # 如果是soft_assignment，这里写 hard_assignments
        assignment_info = {}

        # hard_assignments
        for class_idx in unique_assignments:
            mask = (assignments == class_idx)
            class_features = features_norm[mask]  # 属于该类的特征
            num_samples = mask.sum().item()

            if num_samples > 0:
                # 更新访问计数 (Algorithm 2, Line 13)
                self.class_visit_counts[class_idx] += num_samples

                # 计算动量参数η (Algorithm 2, Line 14)
                eta = self.spacing_momentum / (1 + self.class_visit_counts[class_idx] * 0.01)
                # eta = 1.0 / self.class_visit_counts[class_idx]
                eta = torch.clamp(eta, 0.001, 0.5)  #

                # 计算目标：当前特征均值 + 对应等距点 (Algorithm 2, Line 15)
                class_features_mean = class_features.mean(dim=0)
                equidistant_point = self.equidistant_points[class_idx]
                target_combination = class_features_mean + equidistant_point

                # 原型更新：向目标组合移动
                current_prototype = prototypes_norm[class_idx]
                target_prototype = (1 - eta) * current_prototype + eta * target_combination
                target_prototype = F.normalize(target_prototype, dim=-1, p=2)

                updated_prototypes[class_idx] = target_prototype

                # Algorithm 2, Line 15: pczi ← (1-η)pczi + η(zi + pe_czi)
                # for sample_idx in torch.where(mask)[0]:
                #     zi = features_norm[sample_idx]  # 单个样本特征
                #     current_prototype = updated_prototypes[class_idx]
                #     equidistant_point = self.equidistant_points[class_idx]
                #
                #     # 按照论文公式：zi + pe_czi
                #     target_combination = zi + equidistant_point
                #     target_combination = F.normalize(target_combination, dim=-1, p=2)
                #
                #     # 更新原型
                #     updated_prototype = (1 - eta) * current_prototype + eta * target_combination
                #     updated_prototypes[class_idx] = F.normalize(updated_prototype, dim=-1, p=2)

                # 记录分配信息
                assignment_info[class_idx.item()] = {
                    'num_samples': num_samples,
                    'eta': eta.item(),
                    'visit_count': self.class_visit_counts[class_idx].item(),
                }

        alignment_loss = F.mse_loss(prototypes_norm, equidistant_norm)


        # 更新存储的原型
        with torch.no_grad():
            self.prototypes = F.normalize(updated_prototypes, dim=-1, p=2)

        # return feature_to_prototype_loss, assignment_info
        return alignment_loss


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

    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        """离线阶段更新prototypes并初始化等距点"""
        model.eval()
        all_feats_list = []
        all_labels_list = []

        # 提取特征
        with torch.no_grad():
            for batch_idx, (images, label, _) in enumerate(
                    tqdm(train_loader, desc="Extracting features for prototypes")):
                images = images.cuda(non_blocking=True)
                # feats = model.backbone(images)
                feats = model[0](images)
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

        # 计算等距点
        self.equidistant_points = self.compute_equidistant_points()

        # 更新mean similarity
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

        # 初始化访问计数
        self.class_visit_counts = torch.zeros(num_labeled_classes, device=self.device)

        self.logger.info(f"Offline: Updated prototypes for {len(prototypes_list)} classes")
        if self.equidistant_points is not None:
            self.logger.info(f"Computed equidistant points with spacing_alpha: {self.spacing_alpha}")

    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        """在线阶段更新prototypes"""
        model.eval()
        all_preds_list = []
        all_feats_list = []

        with torch.no_grad():
            for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader, desc="Updating online prototypes")):
                images = images.cuda(non_blocking=True)
                _, logits = model(images)
                # feats = model.backbone(images)
                feats = model[0](images)
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
                # feats_c_mean = model.projector.last_layer.weight_v.data[c]
                feats_c_mean = model[1].last_layer.weight_v.data[c]
            else:
                self.logger.info(f'Computing class-wise mean for class {c}...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        prototypes_cur = torch.stack(prototypes_list, dim=0)
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # 更新状态
        self.prototypes = prototypes_all

        # 重新计算等距点
        self.equidistant_points = self.compute_equidistant_points()

        # 更新mean similarity
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

        # 扩展访问计数
        new_counts = torch.zeros(len(prototypes_cur), device=self.device)
        self.class_visit_counts = torch.cat([self.class_visit_counts, new_counts])

        self.logger.info(f"Online: Updated prototypes to {len(prototypes_all)} classes total")
        if self.equidistant_points is not None:
            self.logger.info(f"Recomputed equidistant points for {len(prototypes_all)} classes")
