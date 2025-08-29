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

        # Key-Value分离的prompts设计
        self.prompt_keys = None  # [num_prompts, feature_dim] - 用于相似度计算和选择
        self.prompt_values = None  # [num_prompts, feature_dim] - 用于实际的特征增强

        # Query projection for prompt selection
        self.query_proj = nn.Linear(feature_dim, feature_dim)

        # Key和Value的独立投影层
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # 使用统计
        self.prompt_usage_stats = None

        # 训练相关参数
        self.diversity_weight = 0.01  # 多样性损失权重
        self.consistency_weight = 0.05  # Key-Value一致性损失权重

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
        从初始prompts初始化可学习的Key-Value参数

        Args:
            initial_prompts: 初始prompt张量 [num_prompts, feature_dim]
        """
        if initial_prompts is not None:
            self.num_prompts = min(len(initial_prompts), self.max_prompts)

            # 初始化Key和Value参数
            self.prompt_keys = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach(),
                requires_grad=True
            )

            noise = torch.randn_like(initial_prompts[:self.num_prompts]) * 0.01
            self.prompt_values = nn.Parameter(
                (initial_prompts[:self.num_prompts] + noise).clone().detach(),
                requires_grad=True
            )

            # 检查是否已存在 buffer，避免重复注册
            if not hasattr(self, 'prompt_usage_stats'):
                self.register_buffer('prompt_usage_stats',
                                     torch.zeros(self.num_prompts, dtype=torch.float))
            else:
                # 如果已存在，重新初始化其值
                self.prompt_usage_stats = torch.zeros(self.num_prompts, dtype=torch.float, device=self.device)
                print("prompt_usage_stats already exists, reinitializing...")

            print(f"Initialized {self.num_prompts} learnable prompts with Key-Value separation")
            print(f"Keys requires_grad: {self.prompt_keys.requires_grad}")
            print(f"Values requires_grad: {self.prompt_values.requires_grad}")

    def forward(self, x, return_attention=False):
        """
        前向传播：使用Key-Value分离机制选择和应用prompts

        Args:
            x: [B, seq_len, D] 输入token序列
            return_attention: 是否返回注意力信息

        Returns:
            enhanced_x: [B, top_k + seq_len, D] prepend后的序列
            attention_info: 注意力信息
        """
        if self.prompt_keys is None or self.num_prompts == 0:
            return x, None

        batch_size, seq_len, _ = x.shape

        # 1. 计算query - 使用CLS token (假设第一个token是CLS)
        cls_token = x[:, 0]  # [B, D]
        query = self.query_proj(cls_token)  # [B, D]

        # 2. 使用Keys计算相似度进行选择
        query_norm = F.normalize(query, dim=1)  # [B, D]

        # 对Keys进行投影和归一化
        projected_keys = self.key_proj(self.prompt_keys)  # [num_prompts, D]
        keys_norm = F.normalize(projected_keys, dim=1)  # [num_prompts, D]

        # 计算query与keys的相似度
        similarity = torch.matmul(query_norm, keys_norm.T)  # [B, num_prompts]

        # 3. 选择top_k个最相关的prompts
        top_k_actual = min(self.top_k, self.num_prompts)
        # =================== 这里不可微 ===================
        # top_k_values, top_k_indices = torch.topk(similarity, top_k_actual, dim=1)  # [B, top_k]
        #
        # # 4. 使用Values进行实际的特征增强
        # selected_values = self.prompt_values[top_k_indices]  # [B, top_k, D]
        #
        # # 对Values进行投影处理
        # enhanced_values = self.value_proj(selected_values)  # [B, top_k, D]
        # 计算相似度分数
        similarity = torch.matmul(query_norm, keys_norm.T)  # [B, num_prompts]

        # 使用temperature softmax得到权重
        attention_weights = F.softmax(similarity / 0.1, dim=1)  # [B, num_prompts]

        # 对所有values进行加权组合，而不是选择top-k
        enhanced_values = torch.matmul(attention_weights.unsqueeze(1),
                                       self.prompt_values.unsqueeze(0))  # [B, 1, D]

        # 5. Prepend enhanced values到输入序列前面
        enhanced_x = torch.cat([enhanced_values, x], dim=1)  # [B, top_k + seq_len, D]

        # 6. 更新使用统计
        self.update_usage_stats_soft(attention_weights)

        attention_info = {
            'attention_weights': attention_weights,  # [B, num_prompts] - 所有prompts的权重
            'similarity_scores': similarity,  # [B, num_prompts] - 原始相似度分数
            'num_prompts_used': self.num_prompts,  # 所有prompts都参与了计算
            'temperature': 0.1,  # softmax温度参数
            'enhanced_values': enhanced_values,  # [B, 1, D] - 加权后的values
        } if return_attention else None

        return enhanced_x, attention_info

    def update_usage_stats_soft(self, attention_weights):
        """基于soft attention权重更新使用统计"""
        if self.prompt_usage_stats is not None:
            # 将attention权重累加到使用统计中
            batch_usage = attention_weights.sum(dim=0)  # [num_prompts]
            self.prompt_usage_stats += batch_usage

    def compute_prompts_losses(self, attention_info=None):
        """
        计算prompts相关的训练损失，包括Key-Value一致性损失

        Args:
            attention_info: 注意力信息

        Returns:
            losses: 损失字典
        """
        losses = {}
        device = self.prompt_keys.device if self.prompt_keys is not None else torch.device('cpu')

        if self.prompt_keys is None or self.num_prompts == 0:
            return {'diversity_loss': torch.tensor(0.0, device=device),
                    'consistency_loss': torch.tensor(0.0, device=device),
                    'total': torch.tensor(0.0, device=device)}

        # 1. Key多样性损失：鼓励不同keys之间保持差异性
        keys_norm = F.normalize(self.prompt_keys, dim=1)
        key_similarity_matrix = torch.matmul(keys_norm, keys_norm.T)

        # 移除对角线元素（自相似度）
        mask = torch.eye(self.num_prompts, device=device).bool()
        key_similarity_matrix.masked_fill_(mask, 0)

        # Key多样性损失：最小化非对角元素的相似度
        key_diversity_loss = key_similarity_matrix.abs().mean()

        # 2. Value多样性损失：鼓励不同values之间保持差异性
        values_norm = F.normalize(self.prompt_values, dim=1)
        value_similarity_matrix = torch.matmul(values_norm, values_norm.T)
        value_similarity_matrix.masked_fill_(mask, 0)

        value_diversity_loss = value_similarity_matrix.abs().mean()

        # 3. Key-Value一致性损失：鼓励对应的key和value保持适度相关
        key_value_similarity = torch.sum(keys_norm * values_norm, dim=1)
        target_similarity = 0.7  # 期望的相似度，不要太高避免退化，不要太低保持相关性
        consistency_loss = F.mse_loss(key_value_similarity,
                                      torch.full_like(key_value_similarity, target_similarity))

        # 4. 选择多样性损失：鼓励选择不同的prompts
        selection_diversity_loss = torch.tensor(0.0, device=device)
        if attention_info is not None and 'attention_weights' in attention_info:
            attention_weights = attention_info['attention_weights']  # [B, num_prompts]

            # 鼓励attention分布的多样性（避免总是关注少数几个prompts）
            avg_attention = attention_weights.mean(dim=0)  # [num_prompts]

            # 计算注意力分布的熵，鼓励更均匀的分布
            entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))
            max_entropy = torch.log(torch.tensor(float(self.num_prompts)))
            selection_diversity_loss = 1.0 - entropy / max_entropy

        # 组合损失
        losses['key_diversity_loss'] = self.diversity_weight * key_diversity_loss
        losses['value_diversity_loss'] = self.diversity_weight * value_diversity_loss
        losses['consistency_loss'] = self.consistency_weight * consistency_loss
        losses['selection_diversity_loss'] = 0.01 * selection_diversity_loss

        losses['total'] = (losses['key_diversity_loss'] +
                           losses['value_diversity_loss'] +
                           losses['consistency_loss'] +
                           losses['selection_diversity_loss'])

        return losses

    def get_prompts_parameters(self):
        """获取prompts相关的可学习参数"""
        params = []

        # Key和Value参数
        if self.prompt_keys is not None:
            params.append(self.prompt_keys)
        if self.prompt_values is not None:
            params.append(self.prompt_values)

        # 投影层参数
        for param in self.query_proj.parameters():
            params.append(param)
        for param in self.key_proj.parameters():
            params.append(param)
        for param in self.value_proj.parameters():
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
            'prompt_keys': self.prompt_keys.cpu() if self.prompt_keys is not None else None,
            'prompt_values': self.prompt_values.cpu() if self.prompt_values is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'top_k': self.top_k,
            'prompt_usage_stats': self.prompt_usage_stats.cpu() if self.prompt_usage_stats is not None else None,
            'query_proj_state': self.query_proj.state_dict(),
            'key_proj_state': self.key_proj.state_dict(),
            'value_proj_state': self.value_proj.state_dict(),
        }
        torch.save(state_dict, save_path)
        print(f"Enhanced PromptsEnhancer (Key-Value) state saved to {save_path}")

    def load_state(self, load_path):
        """加载状态"""
        state_dict = torch.load(load_path, map_location=self.device)

        if state_dict['prompt_keys'] is not None:
            self.prompt_keys = nn.Parameter(state_dict['prompt_keys'].to(self.device))
        if state_dict['prompt_values'] is not None:
            self.prompt_values = nn.Parameter(state_dict['prompt_values'].to(self.device))

        self.num_prompts = state_dict['num_prompts']
        self.top_k = state_dict.get('top_k', self.top_k)

        if state_dict['prompt_usage_stats'] is not None:
            self.register_buffer('prompt_usage_stats', state_dict['prompt_usage_stats'].to(self.device))

        # 加载投影层状态
        self.query_proj.load_state_dict(state_dict['query_proj_state'])
        self.key_proj.load_state_dict(state_dict['key_proj_state'])
        self.value_proj.load_state_dict(state_dict['value_proj_state'])

        print(f"Enhanced PromptsEnhancer (Key-Value) state loaded from {load_path}")


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