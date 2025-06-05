'''
Modify: 更改ProtoAug以适用Prompt Pool + PromptEnhancedModel
Date: 2025/5/14
Author: Wei Jin

Update: 兼容可学习的Prompt Pool
Date: 2025/5/16
Author: Wei Jin
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.prompt_enhanced_model import PromptEnhancedModel


class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None
        self.hardness_temp = hardness_temp
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

    def load_proto_aug_dict(self, load_path):
        proto_aug_dict = torch.load(load_path)
        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']

    def compute_proto_aug_hardness_aware_loss(self, model):
        """
        计算基于hardness-aware采样的prototype augmentation损失

        Args:
            model: PromptEnhancedModel 或兼容的模型

        Returns:
            prototype augmentation损失
        """
        if self.prototypes is None:
            return torch.tensor(0.0, device=self.device)

        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                device=self.device) * self.radius * self.radius_scale

        # 根据模型类型选择合适的forward方式
        if isinstance(model, PromptEnhancedModel):
            # 对于PromptEnhancedModel，我们需要处理prompt增强
            # 在计算proto_aug_loss时，我们可能希望使用原始特征（不使用prompt增强）
            # 或者我们可以让prototype也受益于prompt增强

            # 选项1：禁用prompt增强来计算prototype loss
            original_prompt_training = model.enable_prompt_training
            model.disable_prompt_learning()
            try:
                _, prototypes_output = model.projector(prototypes_augmented)
            finally:
                if original_prompt_training:
                    model.enable_prompt_learning()

            # 选项2：使用prompt增强的prototype（注释掉上面的代码，启用下面的代码）
            # enhanced_prototypes = model.prompt_pool.forward(prototypes_augmented) if model.prompt_pool is not None else prototypes_augmented
            # _, prototypes_output = model.projector(enhanced_prototypes)

        else:
            # 兼容原始的Sequential模型结构
            _, prototypes_output = model.projector(prototypes_augmented)

        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss

    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        """
        在离线阶段更新prototypes

        Args:
            model: PromptEnhancedModel 或兼容的模型
            train_loader: 训练数据加载器
            num_labeled_classes: 标记类别数量
        """
        model.eval()

        all_feats_list = []
        all_labels_list = []

        # 在特征提取时禁用prompt增强，确保使用原始backbone特征
        original_prompt_training = False
        if isinstance(model, PromptEnhancedModel):
            original_prompt_training = model.enable_prompt_training
            model.disable_prompt_learning()

        try:
            # forward data
            for batch_idx, (images, label, _) in enumerate(
                    tqdm(train_loader, desc="Extracting features for prototypes")):
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    feats = model.backbone(images)  # 直接使用backbone
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    all_feats_list.append(feats)
                    all_labels_list.append(label)
        finally:
            # 恢复原始prompt训练设置
            if isinstance(model, PromptEnhancedModel) and original_prompt_training:
                model.enable_prompt_learning()

        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # compute prototypes and radius
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

        # update
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # update mean similarity for each prototype
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i, i] -= similarity[i, i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
        self.mean_similarity = mean_similarity

        self.logger.info(f"Updated prototypes: {len(prototypes_list)} classes, radius: {avg_radius:.4f}")

    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        """
        在线阶段更新prototypes

        Args:
            model: PromptEnhancedModel 或兼容的模型
            train_loader: 训练数据加载器
            num_seen_classes: 已见类别数量
            num_all_classes: 总类别数量
        """
        model.eval()

        all_preds_list = []
        all_feats_list = []

        # 在特征提取时使用完整模型进行预测，但提取原始backbone特征用于prototype
        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader, desc="Updating online prototypes")):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                if isinstance(model, PromptEnhancedModel):
                    # 使用完整模型进行预测（包括prompt增强）
                    _, logits = model(images)
                    all_preds_list.append(logits.argmax(1))

                    # 但提取原始backbone特征用于计算prototype
                    original_prompt_training = model.enable_prompt_training
                    model.disable_prompt_learning()
                    try:
                        feats = model.backbone(images)
                    finally:
                        if original_prompt_training:
                            model.enable_prompt_learning()
                else:
                    # 兼容原始Sequential模型
                    feats, logits = model(images)
                    all_preds_list.append(logits.argmax(1))

                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)

        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # compute prototypes for new classes
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds == c]
            if len(feats_c) == 0:
                self.logger.info(f'No pred of class {c}, using fc (last_layer) parameters...')
                if isinstance(model, PromptEnhancedModel):
                    feats_c_mean = model.projector.last_layer.weight_v.data[c]
                else:
                    feats_c_mean = model[1].last_layer.weight_v.data[c]
            else:
                self.logger.info(f'Computing (predicted) class-wise mean for class {c}...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)

        if prototypes_list:
            prototypes_cur = torch.stack(prototypes_list, dim=0)
            prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
            prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

            # update
            self.prototypes = prototypes_all

            # update mean similarity for each prototype
            similarity = prototypes_all @ prototypes_all.T
            for i in range(len(similarity)):
                similarity[i, i] -= similarity[i, i]
            mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
            self.mean_similarity = mean_similarity

            self.logger.info(f"Updated prototypes: {len(prototypes_all)} classes total")
        else:
            self.logger.warning("No new prototypes to add")

    def get_prototype_statistics(self):
        """
        获取prototype的统计信息

        Returns:
            Dict containing prototype statistics
        """
        if self.prototypes is None:
            return {"error": "No prototypes available"}

        stats = {
            'num_prototypes': len(self.prototypes),
            'feature_dim': self.feature_dim,
            'radius': self.radius.item() if isinstance(self.radius, torch.Tensor) else self.radius,
            'mean_similarity_stats': {
                'mean': self.mean_similarity.mean().item() if self.mean_similarity is not None else 0,
                'std': self.mean_similarity.std().item() if self.mean_similarity is not None else 0,
                'min': self.mean_similarity.min().item() if self.mean_similarity is not None else 0,
                'max': self.mean_similarity.max().item() if self.mean_similarity is not None else 0,
            }
        }

        return stats

    def visualize_prototype_similarities(self, save_path):
        """
        可视化prototype之间的相似度矩阵

        Args:
            save_path: 保存路径
        """
        if self.prototypes is None:
            self.logger.warning("No prototypes available for visualization")
            return

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 计算相似度矩阵
            similarity_matrix = self.prototypes @ self.prototypes.T
            similarity_matrix = similarity_matrix.cpu().numpy()

            # 创建热力图
            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_matrix, annot=False, cmap='viridis',
                        xticklabels=False, yticklabels=False)
            plt.title('Prototype Similarity Matrix')
            plt.xlabel('Prototype Index')
            plt.ylabel('Prototype Index')

            # 添加统计信息
            stats_text = f"""
            Prototypes: {len(self.prototypes)}
            Radius: {self.radius:.4f}
            Avg Similarity: {self.mean_similarity.mean():.4f}
            """
            plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Prototype similarity matrix saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to visualize prototype similarities: {str(e)}")

    def analyze_hardness_distribution(self):
        """
        分析hardness分布

        Returns:
            Dict containing hardness analysis
        """
        if self.mean_similarity is None:
            return {"error": "No similarity data available"}

        # 计算hardness分布
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)

        analysis = {
            'hardness_temp': self.hardness_temp,
            'mean_similarity_stats': {
                'mean': self.mean_similarity.mean().item(),
                'std': self.mean_similarity.std().item(),
                'min': self.mean_similarity.min().item(),
                'max': self.mean_similarity.max().item(),
            },
            'sampling_prob_stats': {
                'mean': sampling_prob.mean().item(),
                'std': sampling_prob.std().item(),
                'min': sampling_prob.min().item(),
                'max': sampling_prob.max().item(),
                'entropy': -(sampling_prob * torch.log(sampling_prob + 1e-8)).sum().item(),
            },
            'top_5_hardest_indices': self.mean_similarity.argsort(descending=True)[:5].tolist(),
            'top_5_easiest_indices': self.mean_similarity.argsort(descending=False)[:5].tolist(),
        }

        return analysis