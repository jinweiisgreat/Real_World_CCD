import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class PromptsEnhancer(nn.Module):
    """
    简洁的Prompts增强器，用于加载预训练的prompts pool并进行特征增强
    """

    def __init__(self, feature_dim, max_prompts=200, device='cuda'):
        super(PromptsEnhancer, self).__init__()

        self.feature_dim = feature_dim
        self.max_prompts = max_prompts
        self.device = device
        self.num_prompts = 0

        # 可学习的prompt参数：Key和Value分离设计
        self.prompt_keys = None  # 用于相似度计算和prompt选择
        self.prompt_values = None  # 用于实际的特征增强

        # K-Q-V投影层
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

        # 层归一化
        self.norm = nn.LayerNorm(feature_dim)

        # 使用统计
        self.prompt_usage_stats = None

        # 训练权重
        self.consistency_weight = 0.1  # Key-Value一致性权重
        self.similarity_weight = 0.15  # 相似度损失权重

    def load_prompts_pool(self, prompts_pool_path):
        """
        从预训练的prompts pool加载prompts

        Args:
            prompts_pool_path: prompts pool的保存路径

        Returns:
            bool: 是否成功加载
        """
        print(f"Loading Prepare from {prompts_pool_path}...")

        # 加载prompts pool数据
        with open(prompts_pool_path, 'rb') as f:
            prompts_data = pickle.load(f)

        prompts_pool = prompts_data['prompts_pool']  # [num_prompts, feature_dim]

        print(f"Loaded Prepare with shape: {prompts_pool.shape}")

        # 转换为tensor并初始化可学习参数
        initial_prompts = torch.tensor(prompts_pool, dtype=torch.float32).to(self.device)
        self._initialize_learnable_prompts(initial_prompts)

        print(f"Successfully initialized {self.num_prompts} learnable prompts")
        return True

    def _initialize_learnable_prompts(self, initial_prompts):
        """
        从初始prompts初始化可学习的参数

        Args:
            initial_prompts: 初始prompt张量 [num_prompts, feature_dim]
        """
        if initial_prompts is not None:
            self.num_prompts = min(len(initial_prompts), self.max_prompts)

            # 初始化Key和Value参数
            # Key：专门用于计算相似度，决定prompt选择
            self.prompt_keys = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            # Value：专门用于特征增强
            self.prompt_values = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            # 初始化使用统计
            self.prompt_usage_stats = torch.zeros(
                self.num_prompts,
                device=self.device,
                dtype=torch.float
            )

            print(f"Initialized {self.num_prompts} learnable prompts with separate Keys and Values")
            print(f"Prompts Keys shape: {self.prompt_keys.shape}")
            print(f"Prompts Values shape: {self.prompt_values.shape}")
        else:
            print("No initial prompts provided, skipping initialization")

    def forward(self, features, top_k=5):
        """
        前向传播：使用prompts增强特征

        Args:
            features: 输入特征 [B, D] 或 [2*B, D]
            top_k: 使用top-k个最相似的prompts

        Returns:
            enhanced_features: 增强后的特征 [B, D] 或 [2*B, D]
            attention_info: 注意力信息字典
        """
        if self.prompt_keys is None or self.num_prompts == 0:
            return features, None

        batch_size = features.shape[0]

        # 1. 计算特征与prompt keys的相似度
        features_norm = F.normalize(features, dim=1)
        keys_norm = F.normalize(self.prompt_keys, dim=1)

        # 相似度矩阵 [B, num_prompts]
        similarity = features_norm @ keys_norm.T

        # 2. 选择top-k个最相似的prompts
        top_k = min(top_k, self.num_prompts)
        top_k_values, top_k_indices = torch.topk(similarity, top_k, dim=1)

        # 3. 统计使用情况
        if self.prompt_usage_stats is not None:
            batch_usage = torch.zeros(self.num_prompts, device=self.device, dtype=torch.float)
            for indices in top_k_indices:
                for idx in indices:
                    batch_usage[idx] += 1
            self.prompt_usage_stats += batch_usage

        # 4. 计算注意力权重
        attention_weights = F.softmax(top_k_values / 0.1, dim=1)  # [B, top_k]

        # 5. 获取选中的prompt values
        selected_prompt_values = self.prompt_values[top_k_indices]  # [B, top_k, D]

        # 6. K-Q-V融合
        enhanced_features = self._kqv_fusion(features, selected_prompt_values, attention_weights)

        # 7. 准备注意力信息
        attention_info = {
            'attention_weights': attention_weights,
            'selected_prompt_indices': top_k_indices,
            'top_k_similarities': top_k_values,
        }

        return enhanced_features, attention_info

    def _kqv_fusion(self, features, selected_prompts, attention_weights):
        """
        K-Q-V融合机制

        Args:
            features: 原始特征 [B, D]
            selected_prompts: 选中的prompt values [B, top_k, D]
            attention_weights: 注意力权重 [B, top_k]

        Returns:
            enhanced_features: 融合后的特征 [B, D]
        """
        # Query: 来自原始特征
        Q = self.query_proj(features).unsqueeze(1)  # [B, 1, D]

        # Key和Value: 来自选中的prompts
        K = self.key_proj(selected_prompts)  # [B, top_k, D]
        V = self.value_proj(selected_prompts)  # [B, top_k, D]

        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, 1, top_k]
        attention_scores = attention_scores / (self.feature_dim ** 0.5)

        # 应用softmax
        attn_weights = F.softmax(attention_scores, dim=-1)  # [B, 1, top_k]

        # 应用注意力到Values
        context = torch.bmm(attn_weights, V)  # [B, 1, D]
        context = context.squeeze(1)  # [B, D]

        # 输出投影
        context = self.output_proj(context)

        # 残差连接 + 层归一化
        enhanced_features = self.norm(features + context)

        return enhanced_features

    def compute_prompts_losses(self, features, enhanced_features, attention_info):
        """
        计算prompts相关的训练损失

        Args:
            features: 原始特征
            enhanced_features: 增强后的特征
            attention_info: 注意力信息

        Returns:
            losses: 损失字典
        """
        losses = {}
        device = features.device

        if self.prompt_keys is None or self.num_prompts == 0:
            return {'consistency_loss': torch.tensor(0.0, device=device),
                    'similarity_loss': torch.tensor(0.0, device=device),
                    'total': torch.tensor(0.0, device=device)}

        # 1. Key-Value一致性损失
        keys_norm = F.normalize(self.prompt_keys, dim=1)
        values_norm = F.normalize(self.prompt_values, dim=1)
        key_value_similarity = torch.sum(keys_norm * values_norm, dim=1)
        target_similarity = 0.7  # 期望的相似度
        consistency_loss = F.mse_loss(key_value_similarity,
                                      torch.full_like(key_value_similarity, target_similarity))

        # 2. 相似度损失：鼓励选中的prompts与特征更相似
        similarity_loss = torch.tensor(0.0, device=device)
        if attention_info is not None and 'top_k_similarities' in attention_info:
            top_k_similarities = attention_info['top_k_similarities']
            attention_weights = attention_info['attention_weights']

            # 加权相似度
            weighted_similarities = top_k_similarities * attention_weights
            avg_weighted_similarity = weighted_similarities.sum(dim=1) / (attention_weights.sum(dim=1) + 1e-8)

            # 损失：鼓励高相似度
            similarity_loss = (1.0 - avg_weighted_similarity).mean()

        # 组合损失
        losses['consistency_loss'] = self.consistency_weight * consistency_loss
        losses['similarity_loss'] = self.similarity_weight * similarity_loss
        losses['total'] = losses['consistency_loss'] + losses['similarity_loss']

        return losses

    def get_prompts_parameters(self):
        """获取prompts相关的可学习参数"""
        params = []
        if self.prompt_keys is not None:
            params.append(self.prompt_keys)
        if self.prompt_values is not None:
            params.append(self.prompt_values)
        return params

    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
        }

        if self.prompt_usage_stats is not None:
            usage_stats = self.prompt_usage_stats
            stats.update({
                'most_used_prompt_usage': usage_stats.max().item(),
                'least_used_prompt_usage': usage_stats.min().item(),
                'average_usage': usage_stats.mean().item(),
                'unused_prompts': (usage_stats == 0).sum().item(),
            })

        return stats

    def save_state(self, save_path):
        """保存当前状态"""
        state_dict = {
            'prompt_keys': self.prompt_keys.cpu() if self.prompt_keys is not None else None,
            'prompt_values': self.prompt_values.cpu() if self.prompt_values is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'prompt_usage_stats': self.prompt_usage_stats.cpu() if self.prompt_usage_stats is not None else None,
        }
        torch.save(state_dict, save_path)
        print(f"Prompts enhancer state saved to {save_path}")

    def load_state(self, load_path):
        """加载状态"""
        state_dict = torch.load(load_path, map_location=self.device)

        if state_dict['prompt_keys'] is not None:
            self.prompt_keys = nn.Parameter(state_dict['prompt_keys'].to(self.device))
            self.prompt_values = nn.Parameter(state_dict['prompt_values'].to(self.device))

        self.num_prompts = state_dict['num_prompts']

        if state_dict['prompt_usage_stats'] is not None:
            self.prompt_usage_stats = state_dict['prompt_usage_stats'].to(self.device)

        print(f"Prompts enhancer state loaded from {load_path}")


# 使用示例函数
def create_prompts_enhancer(feature_dim, prompts_pool_path, device='cuda'):
    """
    创建并初始化PromptsEnhancer

    Args:
        feature_dim: 特征维度
        prompts_pool_path: prompts pool文件路径
        device: 计算设备

    Returns:
        PromptsEnhancer实例
    """
    enhancer = PromptsEnhancer(feature_dim=feature_dim, device=device)

    # 加载预训练的prompts pool
    success = enhancer.load_prompts_pool(prompts_pool_path)

    if success:
        print("PromptsEnhancer created successfully!")
        print(f"Statistics: {enhancer.get_statistics()}")
    else:
        print("Failed to create PromptsEnhancer with Prepare")

    return enhancer