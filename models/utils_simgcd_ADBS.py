import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ADBSHead(nn.Module):
    """
    ADBS增强的分类头，在原有DINOHead基础上添加自适应决策边界
    """

    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()

        # 保持原有的DINOHead结构
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        # ADBS: 自适应决策边界参数
        self.adaptive_boundaries = nn.Parameter(torch.ones(out_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)

        # ADBS: 应用自适应决策边界
        # ϕ(x) = (W · M)^T f(x)，其中M是自适应边界
        adaptive_logits = logits * self.adaptive_boundaries.unsqueeze(0)

        return x_proj, adaptive_logits

    def expand_boundaries(self, new_classes_num):
        """扩展决策边界以适应新类别"""
        current_size = self.adaptive_boundaries.size(0)
        new_size = current_size + new_classes_num

        # 为新类别初始化边界（使用已有类别边界的平均值）
        if current_size > 0:
            mean_boundary = self.adaptive_boundaries.mean().item()
        else:
            mean_boundary = 1.0

        new_boundaries = torch.ones(new_size, device=self.adaptive_boundaries.device)
        new_boundaries[:current_size] = self.adaptive_boundaries.data
        new_boundaries[current_size:] = mean_boundary

        self.adaptive_boundaries = nn.Parameter(new_boundaries)


class ADBSLoss(nn.Module):
    """
    ADBS损失函数：包含分类损失和类间约束损失
    """

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets, prototypes, boundaries, valid_classes_mask=None):
        """
        Args:
            logits: 模型输出 [batch_size, num_classes]
            targets: 目标标签 [batch_size]
            prototypes: 类原型 [num_classes, feature_dim]
            boundaries: 自适应边界 [num_classes]
            valid_classes_mask: 有效类别掩码 [num_classes]
        """
        # 分类损失
        cls_loss = F.cross_entropy(logits, targets)

        # 类间约束损失 (Inter-class Constraint)
        ic_loss = self._compute_inter_class_constraint(prototypes, boundaries, valid_classes_mask)

        # 总损失
        total_loss = cls_loss + self.alpha * ic_loss

        return total_loss, cls_loss, ic_loss

    def _compute_inter_class_constraint(self, prototypes, boundaries, valid_classes_mask=None):
        """
        计算类间约束损失
        实现论文中的约束：(1 - mi)p_i^T w_i + (mj - 1)p_i^T w_j ≤ 0
        """
        if prototypes is None or prototypes.size(0) < 2:
            return torch.tensor(0.0, device=boundaries.device)

        num_classes = prototypes.size(0)
        if valid_classes_mask is not None:
            valid_indices = torch.where(valid_classes_mask)[0]
            if len(valid_indices) < 2:
                return torch.tensor(0.0, device=boundaries.device)
            prototypes = prototypes[valid_indices]
            boundaries = boundaries[valid_indices]
            num_classes = len(valid_indices)

        # 计算类间约束
        ic_loss = 0.0
        count = 0

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    # (1 - mi)p_i^T w_i + (mj - 1)p_i^T w_j
                    term1 = (1 - boundaries[i]) * torch.dot(prototypes[i], prototypes[i])
                    term2 = (boundaries[j] - 1) * torch.dot(prototypes[i], prototypes[j])
                    constraint_value = term1 + term2
                    ic_loss += F.relu(constraint_value)  # max(0, constraint_value)
                    count += 1

        return ic_loss / max(count, 1)


def integrate_adbs_to_model(backbone, num_classes, feature_dim=768, num_mlp_layers=3):
    """
    将ADBS集成到现有模型中
    """
    # 创建ADBS增强的分类头
    adbs_head = ADBSHead(
        in_dim=feature_dim,
        out_dim=num_classes,
        nlayers=num_mlp_layers
    )

    # 组合模型
    model = nn.Sequential(backbone, adbs_head)
    return model


def expand_adbs_model_for_new_session(model, new_classes_num, device):
    """
    为新的增量学习阶段扩展ADBS模型
    """
    # 获取当前分类头
    current_head = model[1]
    current_num_classes = current_head.last_layer.weight.size(0)
    new_num_classes = current_num_classes + new_classes_num

    # 创建新的分类头
    new_head = ADBSHead(
        in_dim=current_head.last_layer.weight.size(1),
        out_dim=new_num_classes,
        nlayers=3  # 可以根据需要调整
    )

    # 复制旧的权重
    with torch.no_grad():
        # 复制MLP权重
        if hasattr(current_head, 'mlp'):
            new_head.mlp.load_state_dict(current_head.mlp.state_dict())

        # 复制分类器权重
        new_head.last_layer.weight_v.data[:current_num_classes] = current_head.last_layer.weight_v.data
        new_head.last_layer.weight_g.data[:current_num_classes] = current_head.last_layer.weight_g.data
        new_head.last_layer.weight.data[:current_num_classes] = current_head.last_layer.weight.data

        # 复制和扩展自适应边界
        new_head.adaptive_boundaries.data[:current_num_classes] = current_head.adaptive_boundaries.data
        # 新类别的边界初始化为已有类别的平均值
        if current_num_classes > 0:
            mean_boundary = current_head.adaptive_boundaries.data.mean()
            new_head.adaptive_boundaries.data[current_num_classes:] = mean_boundary

    # 替换模型的分类头
    model[1] = new_head.to(device)

    return model


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


# offline: ramp-up temperature
class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


# online: fixed sharpened temperature
class DistillLoss_fix(nn.Module):
    def __init__(self, ncrops=2, teacher_temp_final=0.05, student_temp=0.1):
        super().__init__()
        self.ncrops = ncrops
        self.student_temp = student_temp
        self.teacher_temp_final = teacher_temp_final

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_final
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
# class SinkhornKnopp(torch.nn.Module):
#     def __init__(self, num_iters=3, epsilon=0.1):
#         super().__init__()
#         self.num_iters = num_iters
#         self.epsilon = epsilon
#         self.iter = 0
#
#     @torch.no_grad()
#     def forward(self, logits):
#         # Q = torch.exp(logits / self.epsilon).t()
#         Q = torch.exp(logits).t()
#         B = Q.shape[0]
#         K = Q.shape[1]  # how many prototypes
#         sum_Q = torch.sum(Q)
#         Q /= sum_Q
#
#         for it in range(self.num_iters):
#             # normalize each row: total weight per prototype must be 1/K
#             sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
#             Q /= sum_of_rows
#             Q /= K
#
#             # normalize each column: total weight per sample must be 1/B
#             Q /= torch.sum(Q, dim=0, keepdim=True)
#             Q /= B
#
#         Q *= B  # the colomns must sum to 1 so that Q is an assignment
#         return Q.t()
