"""
Modify: 更改ProtoAug以适用Prompt Pool + PromptEnhancedModel
Date: 2025/5/14
Author: Wei Jin

Update: 兼容可学习的Prompt Pool
Date: 2025/5/16
Author: Wei Jin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.prompt_enhanced_model import PromptEnhancedModel
from models.utils_seen_novel_contrastive import SeenNovelContrastiveLearning
from scipy.optimize import linear_sum_assignment as linear_assignment



class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        # 原有ProtoAugManager的所有属性
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.hardness_temp = hardness_temp
        self.radius = 0
        self.radius_scale = radius_scale

        # 核心原型系统
        self.prototypes = None
        self.prototype_labels = None
        self.mean_similarity = None

        # 匈牙利匹配参数
        # self.alignment_threshold = 0.7
        # self.seen_novel_threshold = 0.6

        self.num_old_classes = 50  # 新增：固定为50个old类
        self.alignment_threshold = 0.5

        # 新增：Seen Novel对比学习模块
        self.seen_novel_cl = SeenNovelContrastiveLearning(
            temperature=0.07,
            confidence_threshold=0.65,
            device=device,
            logger=logger
        )

        # 当前session的seen novel原型
        self.current_seen_novel_prototypes = None
        self.current_seen_novel_labels = None

    def save_proto_aug_dict(self, save_path):
        """保存原型系统"""
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'prototype_labels': self.prototype_labels,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
            'current_seen_novel_prototypes': self.current_seen_novel_prototypes,
            'current_seen_novel_labels': self.current_seen_novel_labels,
        }
        torch.save(proto_aug_dict, save_path)

    def load_proto_aug_dict(self, load_path):
        """加载原型系统"""
        proto_aug_dict = torch.load(load_path)
        self.prototypes = proto_aug_dict['prototypes']
        self.prototype_labels = proto_aug_dict.get('prototype_labels', None)
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']
        self.current_seen_novel_prototypes = proto_aug_dict.get('current_seen_novel_prototypes', None)
        self.current_seen_novel_labels = proto_aug_dict.get('current_seen_novel_labels', None)

    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        """
        离线阶段：创建初始原型池
        """
        self.logger.info(f"Creating initial prototype pool for {num_labeled_classes} labeled classes...")

        model.eval()

        # 禁用prompt增强
        original_prompt_training = getattr(model, 'enable_prompt_training', False)
        if hasattr(model, 'disable_prompt_learning'):
            model.disable_prompt_learning()

        try:
            all_feats_list = []
            all_labels_list = []

            for batch_idx, (images, label, _) in enumerate(tqdm(train_loader, desc="Extracting features")):
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    feats = model.backbone(images)
                    feats = F.normalize(feats, dim=-1)
                    all_feats_list.append(feats)
                    all_labels_list.append(label)

            all_feats = torch.cat(all_feats_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0)

            # 计算原型和半径
            prototypes_list = []
            labels_list = []
            radius_list = []

            for c in range(num_labeled_classes):
                feats_c = all_feats[all_labels == c]
                feats_c_mean = torch.mean(feats_c, dim=0)
                feats_c_mean = F.normalize(feats_c_mean, dim=0)

                prototypes_list.append(feats_c_mean)
                labels_list.append(c)

                # 计算半径
                feats_c_center = feats_c - feats_c_mean
                cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)
                radius = torch.trace(cov) / self.feature_dim
                radius_list.append(radius)

            self.prototypes = torch.stack(prototypes_list, dim=0)
            self.prototype_labels = torch.tensor(labels_list, device=self.device)
            self.radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
            self._update_similarity_stats()

            self.logger.info(f"Created {len(prototypes_list)} prototypes with radius: {self.radius:.4f}")

        finally:
            if hasattr(model, 'enable_prompt_learning') and original_prompt_training:
                model.enable_prompt_learning()

    '''
    def prototype_distillation_with_hungarian(self, model_cur, num_old_classes, data_loader):
        """
        步骤1: 基于匈牙利匹配的原型析出

        Returns:
            seen_novel_prototypes: 析出的seen novel原型
            distillation_info: 析出信息
        """
        self.logger.info("=== 开始原型析出：匈牙利匹配算法 ===")

        if self.prototypes is None:
            self.logger.warning("No existing prototypes for distillation")
            return None, {}

        # 计算当前session的原型
        cur_prototypes, cur_labels = self._compute_current_prototypes(model_cur, data_loader)

        if len(cur_prototypes) == 0:
            self.logger.warning("No prototypes computed from current data")
            return None, {}

        # 匈牙利匹配对齐old classes
        old_alignment_map, alignment_quality = self._hungarian_align_old_classes(
            cur_prototypes, cur_labels, num_old_classes
        )

        # 析出seen novel原型
        seen_novel_prototypes, seen_novel_labels = self._extract_seen_novel_after_hungarian(
            cur_prototypes, cur_labels, old_alignment_map
        )

        # 存储当前session的seen novel信息
        self.current_seen_novel_prototypes = seen_novel_prototypes
        self.current_seen_novel_labels = seen_novel_labels

        distillation_info = {
            'hungarian_alignment_map': old_alignment_map,
            'alignment_quality': alignment_quality,
            'seen_novel_count': len(seen_novel_prototypes) if seen_novel_prototypes is not None else 0,
            'total_cur_prototypes': len(cur_prototypes),
        }

        if seen_novel_prototypes is not None:
            self.logger.info(f"✓ 析出 {len(seen_novel_prototypes)} 个seen novel原型")
        else:
            self.logger.info("✗ 未发现seen novel原型")

        return seen_novel_prototypes, distillation_info
    '''

    def prototype_distillation_with_hungarian(self, model_cur, data_loader):
        """
        简化的原型析出：匈牙利匹配识别50个old类，剩余全部为Seen Novel

        Returns:
            seen_novel_prototypes: 析出的seen novel原型
            distillation_info: 析出信息
        """
        self.logger.info("=== 开始简化的原型析出：识别50个old类，剩余全部为Seen Novel ===")

        if self.prototypes is None:
            self.logger.warning("No existing prototypes for distillation")
            return None, {}

        # 计算当前session的原型
        cur_prototypes, cur_labels = self._compute_current_prototypes(model_cur, data_loader)

        if len(cur_prototypes) == 0:
            self.logger.warning("No prototypes computed from current data")
            return None, {}

        # 简化的匈牙利匹配：只对前50个old classes进行对齐
        old_alignment_map, alignment_quality = self._hungarian_align_old_classes(
            cur_prototypes, cur_labels
        )

        # 简化的seen novel析出：剩余的全部作为seen novel
        seen_novel_prototypes, seen_novel_labels = self._extract_all_remaining_as_seen_novel(
            cur_prototypes, cur_labels, old_alignment_map
        )

        # 存储当前session的seen novel信息
        self.current_seen_novel_prototypes = seen_novel_prototypes
        self.current_seen_novel_labels = seen_novel_labels

        distillation_info = {
            'simplified_hungarian_alignment_map': old_alignment_map,
            'alignment_quality': alignment_quality,
            'seen_novel_count': len(seen_novel_prototypes) if seen_novel_prototypes is not None else 0,
            'total_cur_prototypes': len(cur_prototypes),
            'old_classes_aligned': len([info for info in old_alignment_map.values() if info['is_valid']]),
        }

        if seen_novel_prototypes is not None:
            self.logger.info(f"✓ 识别了 {len([info for info in old_alignment_map.values() if info['is_valid']])} 个old类")
            self.logger.info(f"✓ 析出 {len(seen_novel_prototypes)} 个seen novel原型")
        else:
            self.logger.info("✗ 未发现seen novel原型")

        return seen_novel_prototypes, distillation_info




    def _compute_current_prototypes(self, model, data_loader):
        """计算当前session的原型"""
        model.eval()

        original_prompt_training = getattr(model, 'enable_prompt_training', False)
        if hasattr(model, 'disable_prompt_learning'):
            model.disable_prompt_learning()

        try:
            all_features = []
            all_predictions = []
            all_confidences = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing current prototypes")):

                    if len(batch) >= 3:
                        if isinstance(batch[0], list):
                            images = batch[0][0].to(self.device)
                        else:
                            images = batch[0].to(self.device)
                    else:
                        continue


                    features = model.backbone(images)
                    features = F.normalize(features, dim=1)
                    _, logits = model.projector(features)

                    predictions = logits.argmax(1)
                    confidences = F.softmax(logits, dim=1).max(1)[0]

                    all_features.append(features.cpu())
                    all_predictions.append(predictions.cpu())
                    all_confidences.append(confidences.cpu())

            all_features = torch.cat(all_features, dim=0)
            all_predictions = torch.cat(all_predictions, dim=0)
            all_confidences = torch.cat(all_confidences, dim=0)

            # 计算每个类别的原型
            unique_classes = torch.unique(all_predictions)
            prototypes = []
            labels = []

            for cls in unique_classes:
                cls_mask = (all_predictions == cls)
                cls_features = all_features[cls_mask]
                cls_confidences = all_confidences[cls_mask]

                # 对可能的old classes使用高置信度样本
                high_conf_mask = cls_confidences > 0.8
                if high_conf_mask.sum() > max(5, len(cls_features) * 0.3):
                    cls_features = cls_features[high_conf_mask]

                if len(cls_features) > 0:
                    cls_prototype = cls_features.mean(dim=0)
                    cls_prototype = F.normalize(cls_prototype, dim=0)
                    prototypes.append(cls_prototype)
                    labels.append(cls.item())

            return torch.stack(prototypes) if prototypes else torch.empty(0, self.feature_dim), labels

        finally:
            if hasattr(model, 'enable_prompt_learning') and original_prompt_training:
                model.enable_prompt_learning()

    '''
    def _hungarian_align_old_classes(self, cur_prototypes, cur_labels, num_old_classes):
        """匈牙利算法对齐old classes"""
        if len(self.prototypes) < num_old_classes:
            self.logger.warning(f"Existing prototypes ({len(self.prototypes)}) < num_old_classes ({num_old_classes})")
            return {}, 0.0

        pre_old_prototypes = self.prototypes[:num_old_classes]

        # 计算相似度矩阵
        pre_norm = F.normalize(pre_old_prototypes, dim=1).to(self.device)
        cur_norm = F.normalize(cur_prototypes, dim=1).to(self.device)
        similarity_matrix = torch.mm(pre_norm, cur_norm.T)

        # 匈牙利匹配
        cost_matrix = (1.0 - similarity_matrix).cpu().numpy()
        pre_indices, cur_indices = linear_sum_assignment(cost_matrix)

        # 构建对齐映射
        alignment_map = {}
        valid_matches = 0
        total_similarity = 0.0

        for pre_idx, cur_idx in zip(pre_indices, cur_indices):
            similarity = similarity_matrix[pre_idx, cur_idx].item()

            match_info = {
                'cur_idx': cur_idx,
                'cur_label': cur_labels[cur_idx],
                'similarity': similarity,
                'is_valid': similarity > self.alignment_threshold
            }

            alignment_map[pre_idx] = match_info
            total_similarity += similarity

            if similarity > self.alignment_threshold:
                valid_matches += 1
                self.logger.debug(f"✓ Old class {pre_idx} -> cur[{cur_idx}] (sim: {similarity:.3f})")
            else:
                self.logger.debug(f"✗ Low quality: old class {pre_idx} -> cur[{cur_idx}] (sim: {similarity:.3f})")

        # 计算对齐质量
        if len(pre_indices) > 0:
            avg_similarity = total_similarity / len(pre_indices)
            valid_ratio = valid_matches / len(pre_indices)
            coverage_ratio = len(pre_indices) / num_old_classes
            alignment_quality = avg_similarity * valid_ratio * coverage_ratio
        else:
            alignment_quality = 0.0

        self.logger.info(
            f"匈牙利对齐结果: {valid_matches}/{len(pre_indices)} 有效匹配, 质量分数: {alignment_quality:.3f}")

        return alignment_map, alignment_quality
        '''

    def _hungarian_align_old_classes(self, cur_prototypes, cur_labels):
        """
        简化的匈牙利算法：只对前50个old classes进行对齐
        """
        if len(self.prototypes) < self.num_old_classes:
            self.logger.warning(
                f"Existing prototypes ({len(self.prototypes)}) < num_old_classes ({self.num_old_classes})")
            return {}, 0.0

        # 只使用前50个old prototypes
        pre_old_prototypes = self.prototypes[:self.num_old_classes]

        # 计算相似度矩阵
        pre_norm = F.normalize(pre_old_prototypes, dim=1).to(self.device)
        cur_norm = F.normalize(cur_prototypes, dim=1).to(self.device)
        similarity_matrix = torch.mm(pre_norm, cur_norm.T)

        # 匈牙利匹配
        cost_matrix = (1.0 - similarity_matrix).cpu().numpy()

        # 确保cost_matrix的维度正确
        num_pre_old = pre_old_prototypes.shape[0]
        num_cur = cur_prototypes.shape[0]

        if num_cur < num_pre_old:
            # 如果当前原型数量少于old类数量，扩展cost_matrix
            extended_cost_matrix = np.ones((num_pre_old, num_pre_old)) * 2.0  # 大于最大可能的cost
            extended_cost_matrix[:num_pre_old, :num_cur] = cost_matrix
            cost_matrix = extended_cost_matrix
            pre_indices, cur_indices = linear_assignment(cost_matrix)
            # 只保留有效的匹配
            valid_mask = cur_indices < num_cur
            pre_indices = pre_indices[valid_mask]
            cur_indices = cur_indices[valid_mask]
        else:
            pre_indices, cur_indices = linear_assignment(cost_matrix)

        # 构建对齐映射
        alignment_map = {}
        valid_matches = 0
        total_similarity = 0.0

        for pre_idx, cur_idx in zip(pre_indices, cur_indices):
            similarity = similarity_matrix[pre_idx, cur_idx].item()

            match_info = {
                'cur_idx': cur_idx,
                'cur_label': cur_labels[cur_idx],
                'similarity': similarity,
                'is_valid': similarity > self.alignment_threshold  # 使用降低的阈值
            }

            alignment_map[pre_idx] = match_info
            total_similarity += similarity

            if similarity > self.alignment_threshold:
                valid_matches += 1
                self.logger.debug(f"✓ Old class {pre_idx} -> cur[{cur_idx}] (sim: {similarity:.3f})")
            else:
                self.logger.debug(f"✗ Low quality: old class {pre_idx} -> cur[{cur_idx}] (sim: {similarity:.3f})")

        # 计算对齐质量
        if len(pre_indices) > 0:
            avg_similarity = total_similarity / len(pre_indices)
            valid_ratio = valid_matches / len(pre_indices)
            alignment_quality = avg_similarity * valid_ratio
        else:
            alignment_quality = 0.0

        self.logger.info(
            f"简化匈牙利对齐结果: {valid_matches}/{len(pre_indices)} 有效匹配, 质量分数: {alignment_quality:.3f}")

        return alignment_map, alignment_quality

    def _extract_seen_novel_after_hungarian(self, cur_prototypes, cur_labels, hungarian_alignment_map):
        """匈牙利匹配后析出seen novel原型"""
        # 找到被有效对齐的cur原型索引
        valid_aligned_cur_indices = {
            info['cur_idx'] for info in hungarian_alignment_map.values()
            if info['is_valid']
        }

        # 剩余的就是候选seen novel
        all_cur_indices = set(range(len(cur_prototypes)))
        candidate_indices = all_cur_indices - valid_aligned_cur_indices

        self.logger.info(f"候选seen novel: {len(candidate_indices)} 个")

        if len(candidate_indices) == 0:
            return None, []

        # 进一步筛选
        seen_novel_prototypes = []
        seen_novel_labels = []

        num_prev_old = len([info for info in hungarian_alignment_map.values() if info['is_valid']])

        if len(self.prototypes) > num_prev_old:
            # 存在previous novel prototypes
            prev_novel_prototypes = self.prototypes[num_prev_old:]
            prev_novel_norm = F.normalize(prev_novel_prototypes, dim=1).to(self.device)

            for cur_idx in candidate_indices:
                cur_proto_norm = F.normalize(cur_prototypes[cur_idx:cur_idx + 1], dim=1).to(self.device)
                similarities = torch.mm(cur_proto_norm, prev_novel_norm.T).squeeze(0)
                max_similarity = similarities.max().item() if len(similarities) > 0 else 0.0

                if max_similarity > self.seen_novel_threshold:
                    seen_novel_prototypes.append(cur_prototypes[cur_idx])
                    seen_novel_labels.append(cur_labels[cur_idx])

                    best_prev_novel_idx = similarities.argmax().item()
                    self.logger.debug(
                        f"✓ Seen novel: cur[{cur_idx}] -> prev_novel[{best_prev_novel_idx}] (sim: {max_similarity:.3f})")
        else:
            # 第一个在线session
            self.logger.info("第一个在线session - 将候选标记为potential seen novel")
            for cur_idx in candidate_indices:
                seen_novel_prototypes.append(cur_prototypes[cur_idx])
                seen_novel_labels.append(cur_labels[cur_idx])

        if seen_novel_prototypes:
            return torch.stack(seen_novel_prototypes), seen_novel_labels
        else:
            return None, []

    def _extract_all_remaining_as_seen_novel(self, cur_prototypes, cur_labels, hungarian_alignment_map):
        """
        简化的seen novel析出：所有未被匹配为old的原型都作为seen novel
        """
        # 找到被有效对齐的cur原型索引
        valid_aligned_cur_indices = {
            info['cur_idx'] for info in hungarian_alignment_map.values()
            if info['is_valid']
        }

        # 剩余的全部作为seen novel
        all_cur_indices = set(range(len(cur_prototypes)))
        seen_novel_indices = all_cur_indices - valid_aligned_cur_indices

        self.logger.info(f"识别为old类: {len(valid_aligned_cur_indices)} 个")
        self.logger.info(f"识别为seen novel: {len(seen_novel_indices)} 个")

        if len(seen_novel_indices) == 0:
            self.logger.info("✗ 未发现seen novel类")
            return None, []

        # 收集所有seen novel原型
        seen_novel_prototypes = []
        seen_novel_labels = []

        for cur_idx in seen_novel_indices:
            seen_novel_prototypes.append(cur_prototypes[cur_idx])
            seen_novel_labels.append(cur_labels[cur_idx])
            self.logger.debug(f"✓ Seen novel: cur[{cur_idx}] -> label[{cur_labels[cur_idx]}]")

        if seen_novel_prototypes:
            self.logger.info(f"✓ 成功析出 {len(seen_novel_prototypes)} 个seen novel原型")
            return torch.stack(seen_novel_prototypes), seen_novel_labels
        else:
            return None, []


    def compute_enhanced_proto_aug_loss(self, model, current_features=None):
        """
        增强的ProtoAug损失：原有ProtoAug + Seen Novel对比学习

        Args:
            model: 当前模型
            current_features: 当前batch特征（用于seen novel对比学习）

        Returns:
            total_loss: 总损失
            loss_breakdown: 损失分解
        """
        # 原有的ProtoAug损失
        original_proto_aug_loss = self.compute_proto_aug_hardness_aware_loss(model)

        # Seen Novel对比学习损失
        seen_novel_loss = torch.tensor(0.0, device=self.device)
        seen_novel_details = {}

        if (current_features is not None and
                self.current_seen_novel_prototypes is not None and
                len(self.current_seen_novel_prototypes) > 0):
            seen_novel_loss, seen_novel_details = self.seen_novel_cl.compute_comprehensive_seen_novel_loss(
                current_features, self.current_seen_novel_prototypes, model
            )

        # 组合损失
        seen_novel_weight = 0.5  # 可调参数
        total_loss = original_proto_aug_loss + seen_novel_weight * seen_novel_loss

        loss_breakdown = {
            'original_proto_aug': original_proto_aug_loss.item(),
            'seen_novel_contrastive': seen_novel_loss.item(),
            'seen_novel_weight': seen_novel_weight,
            'total': total_loss.item(),
            'seen_novel_details': seen_novel_details
        }

        return total_loss, loss_breakdown

    def compute_proto_aug_hardness_aware_loss(self, model):
        """原有的ProtoAug损失计算"""
        if self.prototypes is None:
            return torch.tensor(0.0, device=self.device)

        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,),
                                             replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                device=self.device) * self.radius * self.radius_scale

        # 模型前向传播
        if hasattr(model, 'projector'):
            original_prompt_training = getattr(model, 'enable_prompt_training', False)
            if hasattr(model, 'disable_prompt_learning'):
                model.disable_prompt_learning()
            try:
                _, prototypes_output = model.projector(prototypes_augmented)
            finally:
                if hasattr(model, 'enable_prompt_learning') and original_prompt_training:
                    model.enable_prompt_learning()
        else:
            _, prototypes_output = model(prototypes_augmented)

        proto_aug_loss = F.cross_entropy(prototypes_output / 0.1, prototypes_labels)
        return proto_aug_loss

    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        """在线更新：为unseen novel classes创建原型"""
        self.logger.info(f"Updating prototypes for unseen novel classes ({num_seen_classes} -> {num_all_classes})")

        model.eval()

        all_preds_list = []
        all_feats_list = []

        with torch.no_grad():
            for batch_idx, (images, label, _, _) in enumerate(
                    tqdm(train_loader, desc="Computing unseen novel prototypes")):
                images = images.cuda(non_blocking=True)
                _, logits = model(images)

                # 提取backbone特征
                if hasattr(model, 'disable_prompt_learning'):
                    model.disable_prompt_learning()
                feats = model.backbone(images)
                feats = F.normalize(feats, dim=-1)
                if hasattr(model, 'enable_prompt_learning'):
                    model.enable_prompt_learning()

                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # 只为真正的unseen novel classes计算原型
        current_prototype_count = len(self.prototypes)
        prototypes_list = []

        for c in range(current_prototype_count, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info(f'No predictions for class {c}, using classifier weights...')
                feats_c_mean = model.projector.last_layer.weight_v.data[c]
            else:
                self.logger.info(f'Computing prototype for unseen novel class {c} ({len(feats_c)} samples)')
                feats_c_mean = torch.mean(feats_c, dim=0)

            feats_c_mean = F.normalize(feats_c_mean, dim=0)
            prototypes_list.append(feats_c_mean)

        if prototypes_list:
            prototypes_cur = torch.stack(prototypes_list, dim=0)
            self.prototypes = torch.cat([self.prototypes, prototypes_cur], dim=0)

            # 更新标签
            new_labels = torch.arange(current_prototype_count, num_all_classes, device=self.device)
            self.prototype_labels = torch.cat([self.prototype_labels, new_labels], dim=0)

            # 更新相似度统计
            self._update_similarity_stats()

            self.logger.info(f"Added {len(prototypes_list)} unseen novel prototypes. Total: {len(self.prototypes)}")

    def update_seen_novel_learning_strategy(self, epoch):
        """更新seen novel学习策略"""
        if self.current_seen_novel_prototypes is not None:
            # 自适应调整阈值
            if epoch % 5 == 0 and epoch > 0:
                self.seen_novel_cl.adaptive_threshold_adjustment()

            # 定期报告统计信息
            if epoch % 10 == 0:
                stats = self.seen_novel_cl.assignment_stats
                self.logger.info(f"Seen Novel Learning Stats - Epoch {epoch}: "
                                 f"Assignment Rate: {stats['assignment_rate']:.3f}, "
                                 f"Avg Confidence: {stats['avg_confidence']:.3f}")

    def _update_similarity_stats(self):
        """更新原型间相似度统计"""
        if self.prototypes is None or len(self.prototypes) <= 1:
            return

        similarity_matrix = torch.mm(self.prototypes, self.prototypes.T)
        for i in range(len(similarity_matrix)):
            similarity_matrix[i, i] = 0
        self.mean_similarity = torch.sum(similarity_matrix, dim=-1) / (len(similarity_matrix) - 1)

    def get_comprehensive_statistics(self):
        """获取综合统计信息"""
        stats = {
            'prototype_system': {
                'num_prototypes': len(self.prototypes) if self.prototypes is not None else 0,
                'radius': self.radius.item() if isinstance(self.radius, torch.Tensor) else self.radius,
            },
            'seen_novel_system': {
                'num_seen_novel_prototypes': len(
                    self.current_seen_novel_prototypes) if self.current_seen_novel_prototypes is not None else 0,
                'assignment_stats': self.seen_novel_cl.assignment_stats.copy(),
                'confidence_threshold': self.seen_novel_cl.confidence_threshold,
            }
        }
        return stats