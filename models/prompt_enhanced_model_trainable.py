"""
第二步改进：修改PromptEnhancedModel以支持真正的任务驱动训练
主要改动：
1. 增加原始特征的计算路径，用于对比分析
2. 改进prompt损失计算，引入分类任务的直接反馈
3. 建立更紧密的prompt-task联系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptEnhancedModel(nn.Module):
    def __init__(self, backbone, projector, prompt_pool=None, top_k=5, enable_prompt_training=True):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.prompt_pool = prompt_pool
        self.top_k = top_k
        self.enable_prompt_training = enable_prompt_training

        # 用于存储最后一次的计算信息
        self.last_computation_info = None

    def forward(self, x, return_prompt_info=False, return_original_logits=False):
        """
        改进的前向传播，支持更详细的信息返回

        Args:
            x: 输入图像 [B, C, H, W]
            return_prompt_info: 是否返回prompt相关信息
            return_original_logits: 是否返回未增强特征的logits（用于对比）

        Returns:
            proj_features: 投影特征
            logits: 分类logits
            prompt_info: (可选) prompt相关信息
            original_logits: (可选) 原始特征的logits
        """
        # 从backbone提取原始特征
        original_features = self.backbone(x)

        # 存储计算信息
        computation_info = {
            'original_features': original_features,
            'enhanced_features': original_features,  # 默认值
            'attention_info': None,
            'enhancement_applied': False
        }

        enhanced_features = original_features
        attention_info = None

        # 如果启用prompt并且prompt_pool存在，则进行特征增强
        if self.prompt_pool is not None and self.enable_prompt_training:
            enhanced_features, attention_info = self.prompt_pool.forward(
                original_features, top_k=self.top_k, return_attention=True
            )
            computation_info['enhanced_features'] = enhanced_features
            computation_info['attention_info'] = attention_info
            computation_info['enhancement_applied'] = True

        # 通过projector得到最终输出
        proj_features, logits = self.projector(enhanced_features)

        # 存储信息供后续使用
        self.last_computation_info = computation_info

        # 准备返回值
        return_values = [proj_features, logits]

        # 如果需要返回原始logits（用于效果对比）
        if return_original_logits:
            with torch.no_grad():
                original_proj_features, original_logits = self.projector(original_features)
            return_values.append(original_logits)

        # 如果需要返回prompt信息
        if return_prompt_info:
            prompt_info = {
                'original_features': original_features,
                'enhanced_features': enhanced_features,
                'attention_info': attention_info,
                'enhancement_applied': self.prompt_pool is not None and self.enable_prompt_training
            }
            return_values.append(prompt_info)

        return tuple(return_values) if len(return_values) > 2 else (return_values[0], return_values[1])

    def forward_with_prompt_loss(self, x, targets=None):
        """
        改进的前向传播，同时计算任务驱动的prompt损失

        Args:
            x: 输入图像 [B, C, H, W]
            targets: 目标标签 [B] (可选)

        Returns:
            proj_features: 投影特征
            logits: 分类logits
            prompt_losses: prompt损失字典
            original_logits: 原始特征的logits（用于效果评估）
        """
        # 获取原始特征
        original_features = self.backbone(x)

        # 如果没有prompt pool，直接返回
        if self.prompt_pool is None or not self.enable_prompt_training:
            proj_features, logits = self.projector(original_features)
            device = next(self.parameters()).device
            prompt_losses = {
                'task_alignment': torch.tensor(0.0, device=device),
                'usage_efficiency': torch.tensor(0.0, device=device),
                'total': torch.tensor(0.0, device=device)
            }
            return proj_features, logits, prompt_losses, logits  # 返回相同的logits作为original_logits

        # 进行prompt增强
        enhanced_features, attention_info = self.prompt_pool.forward(
            original_features, top_k=self.top_k, return_attention=True
        )

        # 计算增强后的输出
        proj_features, enhanced_logits = self.projector(enhanced_features)

        # 计算原始特征的输出（用于对比和损失计算）
        with torch.no_grad():
            _, original_logits = self.projector(original_features)

        # 计算prompt损失，现在传入logits信息用于任务驱动的优化
        prompt_losses = self.prompt_pool.compute_prompt_losses(
            features=original_features,
            enhanced_features=enhanced_features,
            attention_info=attention_info,
            targets=targets,
            logits=enhanced_logits  # 传入分类器输出用于任务对齐
        )

        # 存储计算信息
        self.last_computation_info = {
            'original_features': original_features,
            'enhanced_features': enhanced_features,
            'attention_info': attention_info,
            'enhancement_applied': True,
            'original_logits': original_logits,
            'enhanced_logits': enhanced_logits
        }

        return proj_features, enhanced_logits, prompt_losses, original_logits

    def enable_prompt_learning(self):
        """启用prompt学习"""
        self.enable_prompt_training = True
        if self.prompt_pool is not None:
            for param in self.prompt_pool.parameters():
                param.requires_grad = True

    def disable_prompt_learning(self):
        """禁用prompt学习"""
        self.enable_prompt_training = False
        if self.prompt_pool is not None:
            for param in self.prompt_pool.parameters():
                param.requires_grad = False

    def get_prompt_parameters(self):
        """获取prompt相关的参数，用于优化器"""
        if self.prompt_pool is not None and self.enable_prompt_training:
            return self.prompt_pool.get_prompt_parameters()
        return []

    def get_num_prompts(self):
        """获取当前prompt数量"""
        if self.prompt_pool is not None:
            return getattr(self.prompt_pool, 'num_prompts', 0)
        return 0

    def prompt_pool_summary(self):
        """返回prompt pool的摘要信息"""
        if self.prompt_pool is None:
            return "No prompt pool attached"

        info = {
            'num_prompts': getattr(self.prompt_pool, 'num_prompts', 0),
            'feature_dim': getattr(self.prompt_pool, 'feature_dim', 0),
            'max_prompts': getattr(self.prompt_pool, 'max_prompts', 'Unknown'),
            'prompt_learning_enabled': self.enable_prompt_training,
            'prompt_parameters': sum(p.numel() for p in self.get_prompt_parameters()),
        }

        # 如果有使用统计，添加统计信息
        if hasattr(self.prompt_pool, 'prompt_usage_stats') and self.prompt_pool.prompt_usage_stats is not None:
            usage_stats = self.prompt_pool.prompt_usage_stats
            info.update({
                'most_used_prompt_usage': usage_stats.max().item(),
                'least_used_prompt_usage': usage_stats.min().item(),
                'average_usage': usage_stats.mean().item(),
                'unused_prompts': (usage_stats == 0).sum().item()
            })

        return info