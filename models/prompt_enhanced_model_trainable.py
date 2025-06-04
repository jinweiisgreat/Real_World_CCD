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

        # 用于存储最后一次的attention信息，用于计算prompt损失
        self.last_attention_info = None

    def forward(self, x, return_prompt_info=False):
        """
        Forward pass with optional prompt enhancement

        Args:
            x: Input images [B, C, H, W]
            return_prompt_info: Whether to return prompt-related information

        Returns:
            proj_features: Projected features
            logits: Classification logits
            prompt_info: (optional) Dictionary containing prompt-related information
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Enhance features using prompt pool if available and enabled
        enhanced_features = features
        attention_info = None

        if self.prompt_pool is not None and self.enable_prompt_training:
            if hasattr(self.prompt_pool, 'forward'):
                # 使用新的可学习prompt pool
                enhanced_features, attention_info = self.prompt_pool.forward(
                    features, top_k=self.top_k, return_attention=True
                )
            else:
                # 向后兼容：使用旧的enhance_features方法
                enhanced_features = self.prompt_pool.enhance_features(features, top_k=self.top_k)

        # 存储attention信息用于损失计算
        self.last_attention_info = attention_info

        # Pass through projector
        proj_features, logits = self.projector(enhanced_features)

        if return_prompt_info:
            prompt_info = {
                'original_features': features,
                'enhanced_features': enhanced_features,
                'attention_info': attention_info,
                'enhancement_applied': self.prompt_pool is not None and self.enable_prompt_training
            }
            return proj_features, logits, prompt_info

        return proj_features, logits

    def compute_prompt_losses(self, targets=None):
        """
        计算prompt相关的损失

        Args:
            targets: 目标标签 [B] (可选)

        Returns:
            Dictionary of prompt losses
        """
        if (self.prompt_pool is None or
                not self.enable_prompt_training or
                not hasattr(self.prompt_pool, 'compute_prompt_losses')):
            return {
                'diversity': torch.tensor(0.0, device=next(self.parameters()).device),
                'alignment': torch.tensor(0.0, device=next(self.parameters()).device),
                'total': torch.tensor(0.0, device=next(self.parameters()).device)
            }

        # 需要从上一次forward获取特征信息
        if not hasattr(self, '_last_forward_info'):
            return {
                'diversity': torch.tensor(0.0, device=next(self.parameters()).device),
                'alignment': torch.tensor(0.0, device=next(self.parameters()).device),
                'total': torch.tensor(0.0, device=next(self.parameters()).device)
            }

        return self.prompt_pool.compute_prompt_losses(
            features=self._last_forward_info['original_features'],
            enhanced_features=self._last_forward_info['enhanced_features'],
            attention_info=self.last_attention_info,
            targets=targets
        )

    def forward_with_prompt_loss(self, x, targets=None):
        """
        Forward pass that also computes prompt losses

        Args:
            x: Input images [B, C, H, W]
            targets: Target labels [B] (optional)

        Returns:
            proj_features: Projected features
            logits: Classification logits
            prompt_losses: Dictionary of prompt losses
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Store for prompt loss computation
        self._last_forward_info = {'original_features': features}

        # Enhance features using prompt pool
        enhanced_features = features
        attention_info = None

        if self.prompt_pool is not None and self.enable_prompt_training:
            if hasattr(self.prompt_pool, 'forward'):
                enhanced_features, attention_info = self.prompt_pool.forward(
                    features, top_k=self.top_k, return_attention=True
                )
            else:
                enhanced_features = self.prompt_pool.enhance_features(features, top_k=self.top_k)

        # Store enhanced features
        self._last_forward_info['enhanced_features'] = enhanced_features
        self.last_attention_info = attention_info

        # Pass through projector
        proj_features, logits = self.projector(enhanced_features)

        # Compute prompt losses
        prompt_losses = {}
        if (self.prompt_pool is not None and
                self.enable_prompt_training and
                hasattr(self.prompt_pool, 'compute_prompt_losses')):
            prompt_losses = self.prompt_pool.compute_prompt_losses(
                features=features,
                enhanced_features=enhanced_features,
                attention_info=attention_info,
                targets=targets
            )
        else:
            device = next(self.parameters()).device
            prompt_losses = {
                'diversity': torch.tensor(0.0, device=device),
                'alignment': torch.tensor(0.0, device=device),
                'total': torch.tensor(0.0, device=device)
            }

        return proj_features, logits, prompt_losses

    def enable_prompt_learning(self):
        """启用prompt学习"""
        self.enable_prompt_training = True
        if self.prompt_pool is not None:
            # 确保prompt pool的参数可以被优化
            for param in self.prompt_pool.parameters():
                param.requires_grad = True

    def disable_prompt_learning(self):
        """禁用prompt学习"""
        self.enable_prompt_training = False
        if self.prompt_pool is not None:
            # 冻结prompt pool的参数
            for param in self.prompt_pool.parameters():
                param.requires_grad = False

    def get_prompt_parameters(self):
        """获取prompt相关的参数，用于优化器"""
        if self.prompt_pool is not None and self.enable_prompt_training:
            return list(self.prompt_pool.parameters())
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

        return info