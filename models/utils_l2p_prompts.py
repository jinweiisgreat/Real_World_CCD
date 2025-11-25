import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class PromptsEnhancer(nn.Module):
    """
    端到端训练的Prompts增强器，采用L2P风格的prepend机制
    """

    def __init__(self, feature_dim, max_prompts=200, top_k=5, device='cuda'):
        super(PromptsEnhancer, self).__init__()

        self.feature_dim = feature_dim
        self.max_prompts = max_prompts
        self.top_k = top_k
        self.device = device
        self.num_prompts = 0

        # 可学习的prompts embeddings - 直接作为token embeddings
        self.prompts_embeddings = None  # [num_prompts, feature_dim]

        # Query projection for prompt selection
        self.query_proj = nn.Linear(feature_dim, feature_dim)

        # 可选的prompts projection
        self.prompts_proj = nn.Linear(feature_dim, feature_dim)

        # 使用统计
        self.prompt_usage_stats = None

        # 训练相关参数
        self.diversity_weight = 0.01  # 多样性损失权重

    def load_prompts_pool(self, prompts_pool_path):
        """
        从预训练的prompts pool加载prompts作为初始embeddings

        Args:
            prompts_pool_path: prompts pool的保存路径

        Returns:
            bool: 是否成功加载
        """
        print(f"Loading prompts pool from {prompts_pool_path}...")

        # 加载prompts pool数据
        with open(prompts_pool_path, 'rb') as f:
            prompts_data = pickle.load(f)

        prompts_pool = prompts_data['prompts_pool']  # [num_prompts, feature_dim]

        print(f"Loaded prompts pool with shape: {prompts_pool.shape}")

        # 转换为tensor并初始化可学习参数
        initial_prompts = torch.tensor(prompts_pool, dtype=torch.float32).to(self.device)
        self._initialize_learnable_prompts(initial_prompts)

        print(f"Successfully initialized {self.num_prompts} learnable prompts")
        return True

    def _initialize_learnable_prompts(self, initial_prompts):
        """
        从初始prompts初始化可学习的embeddings

        Args:
            initial_prompts: 初始prompt张量 [num_prompts, feature_dim]
        """
        if initial_prompts is not None:
            self.num_prompts = min(len(initial_prompts), self.max_prompts)

            # 初始化可学习的prompts embeddings
            self.prompts_embeddings = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            # 初始化使用统计
            self.register_buffer('prompt_usage_stats',
                                 torch.zeros(self.num_prompts, dtype=torch.float))

            print(f"Initialized {self.num_prompts} learnable prompts embeddings")
            print(f"Prompts embeddings shape: {self.prompts_embeddings.shape}")
        else:
            print("No initial prompts provided, skipping initialization")

    def forward(self, x, return_attention=False):
        """
        前向传播：选择top_k prompts并prepend到输入序列

        Args:
            x: [B, seq_len, D] 输入token序列
            return_attention: 是否返回注意力信息

        Returns:
            enhanced_x: [B, top_k + seq_len, D] prepend后的序列
            attention_info: 注意力信息
        """
        if self.prompts_embeddings is None or self.num_prompts == 0:
            return x, None

        batch_size, seq_len, _ = x.shape

        # 1. 计算query - 使用CLS token (假设第一个token是CLS)
        cls_token = x[:, 0]  # [B, D]
        query = self.query_proj(cls_token)  # [B, D]

        # 2. 计算与prompts的相似度
        query_norm = F.normalize(query, dim=1)  # [B, D]
        prompts_norm = F.normalize(self.prompts_embeddings, dim=1)  # [num_prompts, D]

        similarity = torch.matmul(query_norm, prompts_norm.T)  # [B, num_prompts]

        # 3. 选择top_k个最相关的prompts
        top_k_actual = min(self.top_k, self.num_prompts)
        top_k_values, top_k_indices = torch.topk(similarity, top_k_actual, dim=1)  # [B, top_k]

        # 4. 获取选中的prompts并投影
        selected_prompts = self.prompts_embeddings[top_k_indices]  # [B, top_k, D]
        selected_prompts = self.prompts_proj(selected_prompts)  # [B, top_k, D]

        # 5. Prepend prompts到输入序列前面
        enhanced_x = torch.cat([selected_prompts, x], dim=1)  # [B, top_k + seq_len, D]

        # 6. 更新使用统计
        self.update_usage_stats(top_k_indices)

        attention_info = {
            'selected_indices': top_k_indices,
            'similarity_scores': top_k_values,
            'num_prompts_used': top_k_actual
        } if return_attention else None

        return enhanced_x, attention_info

    def update_usage_stats(self, selected_indices):
        """更新prompts使用统计"""
        if self.prompt_usage_stats is not None:
            batch_usage = torch.zeros(self.num_prompts, device=self.device, dtype=torch.float)
            for indices in selected_indices:
                for idx in indices:
                    batch_usage[idx] += 1
            self.prompt_usage_stats += batch_usage

    def compute_prompts_losses(self, attention_info=None):
        """
        计算prompts相关的训练损失

        Args:
            attention_info: 注意力信息

        Returns:
            losses: 损失字典
        """
        losses = {}
        device = self.prompts_embeddings.device if self.prompts_embeddings is not None else torch.device('cpu')

        if self.prompts_embeddings is None or self.num_prompts == 0:
            return {'diversity_loss': torch.tensor(0.0, device=device),
                    'total': torch.tensor(0.0, device=device)}

        # 1. 多样性损失：鼓励不同prompts之间保持差异性
        prompts_norm = F.normalize(self.prompts_embeddings, dim=1)
        similarity_matrix = torch.matmul(prompts_norm, prompts_norm.T)

        # 移除对角线元素（自相似度）
        mask = torch.eye(self.num_prompts, device=device).bool()
        similarity_matrix.masked_fill_(mask, 0)

        # 多样性损失：最小化非对角元素的相似度
        diversity_loss = similarity_matrix.abs().mean()

        losses['diversity_loss'] = self.diversity_weight * diversity_loss
        losses['total'] = losses['diversity_loss']

        return losses

    def get_prompts_parameters(self):
        """获取prompts相关的可学习参数"""
        params = []
        if self.prompts_embeddings is not None:
            params.append(self.prompts_embeddings)

        # 添加projection层参数
        for param in self.query_proj.parameters():
            params.append(param)
        for param in self.prompts_proj.parameters():
            params.append(param)

        return params

    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'top_k': self.top_k,
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
            'prompts_embeddings': self.prompts_embeddings.cpu() if self.prompts_embeddings is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'top_k': self.top_k,
            'prompt_usage_stats': self.prompt_usage_stats.cpu() if self.prompt_usage_stats is not None else None,
            'query_proj_state': self.query_proj.state_dict(),
            'prompts_proj_state': self.prompts_proj.state_dict(),
        }
        torch.save(state_dict, save_path)
        print(f"Enhanced PromptsEnhancer state saved to {save_path}")

    def load_state(self, load_path):
        """加载状态"""
        state_dict = torch.load(load_path, map_location=self.device)

        if state_dict['prompts_embeddings'] is not None:
            self.prompts_embeddings = nn.Parameter(state_dict['prompts_embeddings'].to(self.device))

        self.num_prompts = state_dict['num_prompts']
        self.top_k = state_dict.get('top_k', self.top_k)

        if state_dict['prompt_usage_stats'] is not None:
            self.register_buffer('prompt_usage_stats', state_dict['prompt_usage_stats'].to(self.device))

        # 加载projection层状态
        self.query_proj.load_state_dict(state_dict['query_proj_state'])
        self.prompts_proj.load_state_dict(state_dict['prompts_proj_state'])

        print(f"Enhanced PromptsEnhancer state loaded from {load_path}")


def create_prompts_enhancer(feature_dim, prompts_pool_path, top_k=5, device='cuda'):
    """
    创建并初始化Enhanced PromptsEnhancer

    Args:
        feature_dim: 特征维度
        prompts_pool_path: prompts pool文件路径
        top_k: 使用的top-k prompts数量
        device: 计算设备

    Returns:
        PromptsEnhancer实例
    """
    enhancer = PromptsEnhancer(feature_dim=feature_dim, top_k=top_k, device=device)

    # 加载预训练的prompts pool
    success = enhancer.load_prompts_pool(prompts_pool_path)

    if success:
        print("Enhanced PromptsEnhancer created successfully!")
        print(f"Statistics: {enhancer.get_statistics()}")
    else:
        print("Failed to create Enhanced PromptsEnhancer with prompts pool")

    return enhancer