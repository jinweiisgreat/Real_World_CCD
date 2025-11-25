import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveDecisionBoundary(nn.Module):
    """Adaptive Decision Boundary Strategy (ADBS) for FSCIL"""

    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.device = device
        # 初始化决策边界，默认为1.0
        self.boundaries = nn.Parameter(torch.ones(num_classes, device=device))

    def forward(self, logits):
        """应用自适应决策边界到logits"""
        # logits shape: [batch_size, num_classes]
        return logits * self.boundaries.unsqueeze(0)

    def expand_boundaries(self, num_new_classes, init_value=None):
        """为新类别扩展决策边界"""
        if init_value is None:
            # 使用现有边界的平均值初始化新边界
            init_value = self.boundaries.mean().item()

        new_boundaries = torch.full((num_new_classes,), init_value, device=self.device)
        self.boundaries = nn.Parameter(
            torch.cat([self.boundaries.data, new_boundaries])
        )


class InterClassConstraint(nn.Module):
    """Inter-class Constraint Loss for ADBS"""

    def __init__(self):
        super().__init__()

    def forward(self, boundaries, prototypes, classifier_weights):
        """
        计算类间约束损失
        boundaries: [num_classes]
        prototypes: [num_classes, feat_dim]
        classifier_weights: [num_classes, feat_dim]
        """
        num_classes = boundaries.shape[0]
        loss = 0.0

        # 归一化原型
        prototypes = F.normalize(prototypes, dim=-1)

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    # (1 - m_i) * p_i^T * w_i + (m_j - 1) * p_i^T * w_j
                    term1 = (1 - boundaries[i]) * torch.dot(prototypes[i], classifier_weights[i])
                    term2 = (boundaries[j] - 1) * torch.dot(prototypes[i], classifier_weights[j])
                    loss += torch.relu(term1 + term2)

        return loss / (num_classes * (num_classes - 1))