class SeenNovelContrastiveLearning:
    """
    Seen Novel对比学习核心模块
    """

    def __init__(self, temperature=0.07, confidence_threshold=0.65,
                 negative_sampling_ratio=0.5, device='cuda', logger=None):
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.negative_sampling_ratio = negative_sampling_ratio
        self.device = device
        self.logger = logger

        # 统计信息
        self.assignment_stats = {
            'total_samples': 0,
            'assigned_samples': 0,
            'assignment_rate': 0.0,
            'avg_confidence': 0.0
        }

    def soft_assign_seen_novel_samples(self, features, seen_novel_prototypes):
        """
        步骤2: 基于相似度软分配seen novel样本

        Args:
            features: 当前batch特征 [B, D]
            seen_novel_prototypes: 析出的seen novel原型 [K, D]

        Returns:
            assignments: 分配结果字典 或 None
        """
        if seen_novel_prototypes is None or len(seen_novel_prototypes) == 0:
            return None

        batch_size = features.shape[0]

        # 归一化特征和原型
        features_norm = F.normalize(features, dim=1)  # [B, D]
        prototypes_norm = F.normalize(seen_novel_prototypes, dim=1)  # [K, D]

        # 计算相似度矩阵 [B, K]
        similarities = torch.mm(features_norm, prototypes_norm.T)

        # 为每个样本找到最相似的seen novel原型
        max_similarities, best_prototype_indices = similarities.max(dim=1)  # [B], [B]

        # 基于置信度阈值进行软分配
        confident_mask = max_similarities >= self.confidence_threshold
        assigned_count = confident_mask.sum().item()

        if assigned_count == 0:
            if self.logger:
                self.logger.debug(
                    f"No samples assigned (max sim: {max_similarities.max():.3f}, threshold: {self.confidence_threshold:.3f})")
            return None

        # 构建分配结果
        assignments = {
            'features': features_norm[confident_mask],  # [A, D]
            'prototype_indices': best_prototype_indices[confident_mask],  # [A]
            'similarities': max_similarities[confident_mask],  # [A]
            'sample_indices': torch.arange(batch_size, device=features.device)[confident_mask],  # [A]
            'prototypes': prototypes_norm,  # [K, D]
            'all_similarities': similarities[confident_mask],  # [A, K] 用于对比学习
        }

        # 更新统计信息
        self.assignment_stats['total_samples'] += batch_size
        self.assignment_stats['assigned_samples'] += assigned_count
        self.assignment_stats['assignment_rate'] = (
                self.assignment_stats['assigned_samples'] / self.assignment_stats['total_samples']
        )
        avg_conf = max_similarities[confident_mask].mean().item()
        self.assignment_stats['avg_confidence'] = avg_conf

        if self.logger:
            self.logger.debug(f"Assigned {assigned_count}/{batch_size} samples to seen novel "
                              f"(avg confidence: {avg_conf:.3f})")

        return assignments

    def compute_seen_novel_contrastive_loss(self, assignments):
        """
        步骤3: 计算seen novel的对比学习损失

        Args:
            assignments: 软分配结果

        Returns:
            contrastive_loss: 对比学习损失
            loss_info: 详细损失信息
        """
        if assignments is None:
            return torch.tensor(0.0, device=self.device), {}

        assigned_features = assignments['features']  # [A, D]
        prototype_indices = assignments['prototype_indices']  # [A]
        similarities = assignments['similarities']  # [A]
        all_similarities = assignments['all_similarities']  # [A, K]

        num_assigned = len(assigned_features)

        if num_assigned == 0:
            return torch.tensor(0.0, device=self.device), {}

        # 计算正样本相似度：每个assigned sample与其对应prototype
        positive_similarities = similarities  # 已经是max similarities

        # 对比学习损失计算
        # log(exp(pos_sim/τ) / Σ_k exp(sim_k/τ))
        numerator = torch.exp(positive_similarities / self.temperature)  # [A]
        denominator = torch.exp(all_similarities / self.temperature).sum(dim=1)  # [A]

        # 基础对比损失
        base_contrastive_loss = -torch.log(numerator / (denominator + 1e-8)).mean()

        # 加权对比损失：根据置信度加权
        confidence_weights = similarities / (similarities.sum() + 1e-8)  # 归一化权重
        weighted_contrastive_loss = -(confidence_weights * torch.log(numerator / (denominator + 1e-8))).sum()

        # 使用加权损失
        contrastive_loss = weighted_contrastive_loss

        # 详细信息
        loss_info = {
            'num_assigned_samples': num_assigned,
            'avg_positive_similarity': positive_similarities.mean().item(),
            'avg_assignment_confidence': similarities.mean().item(),
            'base_contrastive_loss': base_contrastive_loss.item(),
            'weighted_contrastive_loss': weighted_contrastive_loss.item(),
        }

        return contrastive_loss, loss_info

    def compute_consistency_regularization(self, assignments, model):
        """
        一致性正则化：确保模型预测与软分配一致

        Args:
            assignments: 软分配结果
            model: 当前模型

        Returns:
            consistency_loss: 一致性损失
        """
        if assignments is None:
            return torch.tensor(0.0, device=self.device)

        assigned_features = assignments['features']  # [A, D]
        prototype_indices = assignments['prototype_indices']  # [A]

        if len(assigned_features) == 0:
            return torch.tensor(0.0, device=self.device)

        # 通过模型获得预测
        _, logits = model.projector(assigned_features)  # [A, num_classes]

        # 计算软标签：基于与所有seen novel prototypes的相似度
        all_similarities = assignments['all_similarities']  # [A, K]
        soft_targets = F.softmax(all_similarities / 0.1, dim=1)  # [A, K]

        # 假设seen novel classes在分类器的后面部分
        num_seen_novel = len(assignments['prototypes'])
        seen_novel_logits = logits[:, -num_seen_novel:]  # [A, K]

        # 软标签交叉熵
        log_probs = F.log_softmax(seen_novel_logits / 0.1, dim=1)
        consistency_loss = -torch.sum(soft_targets * log_probs, dim=1).mean()

        return consistency_loss

    def compute_comprehensive_seen_novel_loss(self, features, seen_novel_prototypes, model):
        """
        综合seen novel损失计算

        Args:
            features: 当前batch特征
            seen_novel_prototypes: 析出的seen novel原型
            model: 当前模型

        Returns:
            total_loss: 总损失
            loss_details: 详细损失信息
        """
        # 步骤1: 软分配
        assignments = self.soft_assign_seen_novel_samples(features, seen_novel_prototypes)

        # 步骤2: 对比学习损失
        contrastive_loss, contrastive_info = self.compute_seen_novel_contrastive_loss(assignments)

        # 步骤3: 一致性正则化
        consistency_loss = self.compute_consistency_regularization(assignments, model)

        # 组合损失
        total_loss = contrastive_loss + 0.2 * consistency_loss

        loss_details = {
            'contrastive_loss': contrastive_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item(),
            'contrastive_info': contrastive_info,
            'assignment_stats': self.assignment_stats.copy() if assignments else {}
        }

        return total_loss, loss_details

    def adaptive_threshold_adjustment(self, target_rate=0.15):
        """
        自适应调整置信度阈值
        """
        current_rate = self.assignment_stats['assignment_rate']

        if self.assignment_stats['total_samples'] < 100:
            return  # 样本太少，不调整

        if current_rate < target_rate * 0.5:
            # 分配率太低
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
            if self.logger:
                self.logger.info(f"Lowered seen novel confidence threshold to {self.confidence_threshold:.3f}")
        elif current_rate > target_rate * 2.5:
            # 分配率太高
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.02)
            if self.logger:
                self.logger.info(f"Raised seen novel confidence threshold to {self.confidence_threshold:.3f}")

    def reset_statistics(self):
        """重置统计信息"""
        self.assignment_stats = {
            'total_samples': 0,
            'assigned_samples': 0,
            'assignment_rate': 0.0,
            'avg_confidence': 0.0
        }