import torch
import torch.nn as nn
import torch.nn.functional as F
from models import vision_transformer as vits


class EnhancedVisionTransformer(nn.Module):
    """
    支持Prompts的增强版Vision Transformer
    在输入序列前prepend选中的prompts tokens
    """

    def __init__(self, base_vit, prompts_enhancer=None):
        super(EnhancedVisionTransformer, self).__init__()
        self.base_vit = base_vit
        self.prompts_enhancer = prompts_enhancer

        # 存储原始位置编码以便扩展
        self.original_pos_embed = base_vit.pos_embed.clone()

    def interpolate_pos_encoding(self, x, w, h, extra_tokens=0):
        """
        扩展位置编码以支持额外的prompt tokens

        Args:
            x: 输入tensor
            w, h: 图像宽高
            extra_tokens: 额外的token数量（prompts数量）
        """
        npatch = x.shape[1] - 1 - extra_tokens  # 减去CLS token和prompt tokens
        N = self.original_pos_embed.shape[1] - 1  # 原始patch数量

        if npatch == N and w == h and extra_tokens == 0:
            return self.base_vit.pos_embed

        # CLS token的位置编码
        class_pos_embed = self.original_pos_embed[:, 0:1]

        # Patch tokens的位置编码
        patch_pos_embed = self.original_pos_embed[:, 1:]
        dim = x.shape[-1]

        if npatch != N or w != h:
            # 插值patch位置编码
            w0 = w // self.base_vit.patch_embed.patch_size
            h0 = h // self.base_vit.patch_embed.patch_size
            w0, h0 = w0 + 0.1, h0 + 0.1

            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(N ** 0.5), int(N ** 0.5), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / (N ** 0.5), h0 / (N ** 0.5)),
                mode='bicubic',
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # 为prompt tokens创建零位置编码（它们不需要空间位置信息）
        if extra_tokens > 0:
            prompt_pos_embed = torch.zeros(1, extra_tokens, dim, device=x.device, dtype=x.dtype)
            return torch.cat((class_pos_embed, prompt_pos_embed, patch_pos_embed), dim=1)
        else:
            return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens_with_prompts(self, x):
        """
        准备输入tokens，包括prompts的prepending
        """
        B, nc, w, h = x.shape

        # 1. 标准的patch embedding
        x = self.base_vit.patch_embed(x)  # patch linear embedding

        # 2. 添加CLS token
        cls_tokens = self.base_vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, D]

        # 3. 使用prompts enhancer选择和prepend prompts
        extra_tokens = 0
        if self.prompts_enhancer is not None:
            # 使用当前的token序列（包含CLS）来选择prompts
            enhanced_x, attention_info = self.prompts_enhancer(x, return_attention=True)
            x = enhanced_x  # [B, top_k + 1 + num_patches, D]

            if attention_info is not None:
                extra_tokens = attention_info['num_prompts_used']

        # 4. 添加位置编码（考虑额外的prompt tokens）
        pos_embed = self.interpolate_pos_encoding(x, w, h, extra_tokens)
        x = x + pos_embed

        return self.base_vit.pos_drop(x), extra_tokens

    def forward(self, x, return_all_patches=False, return_attention_info=False):
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]
            return_all_patches: 是否返回所有patch tokens
            return_attention_info: 是否返回prompts attention信息
        """
        # 1. 准备tokens（包含prompts）
        x, num_prompts = self.prepare_tokens_with_prompts(x)

        # 2. 通过transformer blocks
        for blk in self.base_vit.blocks:
            x = blk(x)

        x = self.base_vit.norm(x)

        if return_all_patches:
            return x

        # 3. 返回CLS token（现在位置可能因为prompts而改变）
        cls_token_idx = num_prompts  # CLS token现在在prompts之后
        cls_output = x[:, cls_token_idx]  # 获取CLS token

        if return_attention_info:
            # 如果需要，可以返回prompts相关信息
            return cls_output, {'num_prompts_used': num_prompts}

        return cls_output

    def get_last_selfattention(self, x):
        """获取最后一层的自注意力"""
        x, num_prompts = self.prepare_tokens_with_prompts(x)

        for i, blk in enumerate(self.base_vit.blocks):
            if i < len(self.base_vit.blocks) - 1:
                x = blk(x)
            else:
                # 返回最后一个block的attention
                x, attn = blk(x, return_attention=True)
                x = self.base_vit.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        """获取中间层输出"""
        x, _ = self.prepare_tokens_with_prompts(x)

        # 返回最后n个blocks的输出
        output = []
        for i, blk in enumerate(self.base_vit.blocks):
            x = blk(x)
            if len(self.base_vit.blocks) - i <= n:
                output.append(self.base_vit.norm(x))
        return output


def create_enhanced_backbone(original_backbone, prompts_enhancer):
    """
    创建支持prompts的增强backbone

    Args:
        original_backbone: 原始的ViT backbone
        prompts_enhancer: PromptsEnhancer实例

    Returns:
        增强的backbone
    """
    enhanced_backbone = EnhancedVisionTransformer(original_backbone, prompts_enhancer)

    # 确保梯度设置正确
    for name, param in enhanced_backbone.named_parameters():
        if 'base_vit' in name:
            # 保持原有的梯度设置
            continue
        else:
            # prompts enhancer的参数应该可训练
            param.requires_grad = True

    print(f"Enhanced backbone created with prompts support")
    if prompts_enhancer is not None:
        print(f"Prompts enhancer will prepend {prompts_enhancer.top_k} prompts per sample")

    return enhanced_backbone


def get_enhanced_params_groups(model, prompts_enhancer=None, prompts_lr_scale=0.1):
    """
    为包含prompts的模型创建参数组

    Args:
        model: 完整模型
        prompts_enhancer: PromptsEnhancer实例
        prompts_lr_scale: prompts参数的学习率缩放因子

    Returns:
        参数组列表
    """
    regularized = []
    not_regularized = []
    prompts_params = []

    # 模型参数（不包括prompts）
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 跳过prompts enhancer的参数，单独处理
        if 'prompts_enhancer' in name:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    # Prompts相关参数 - 使用较小的学习率
    if prompts_enhancer is not None:
        for param in prompts_enhancer.parameters():
            if param.requires_grad:
                prompts_params.append(param)

    param_groups = [
        {'params': regularized},
        {'params': not_regularized, 'weight_decay': 0.},
    ]

    # 只有当有prompts参数时才添加prompts组
    if prompts_params:
        param_groups.append({
            'params': prompts_params,
            'lr_scale': prompts_lr_scale,  # 学习率缩放标记
            'weight_decay': 0.
        })

    return param_groups