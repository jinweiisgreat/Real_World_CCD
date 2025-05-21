import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from models.utils_simgcd import DistillLoss  # 导入原始的DistillLoss


class SinkhornKnopp(torch.nn.Module):
    """
    Sinkhorn-Knopp算法实现，用于计算双随机矩阵
    """

    def __init__(self, num_iters=3, epsilon=0.1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        # Q = torch.exp(logits / self.epsilon).t()
        Q = torch.exp(logits).t()
        B = Q.shape[0]
        K = Q.shape[1]  # how many prototypes
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class EnhancedDistillLoss(nn.Module):
    """
    增强版蒸馏损失，使用Sinkhorn-Knopp算法进行约束
    """

    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1, sinkhorn_weight=0.6):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.sinkhorn_weight = sinkhorn_weight

    def forward(self, student_output, teacher_output, epoch, num_iters=3, epsilon=0.05):
        """
        应用带有Sinkhorn-Knopp约束的蒸馏损失

        Args:
            student_output: 学生模型输出 [batch_size, num_classes]
            teacher_output: 教师模型输出 [batch_size, num_classes]
            epoch: 当前训练的epoch
            num_iters: Sinkhorn-Knopp算法的迭代次数
            epsilon: Sinkhorn-Knopp算法的正则化参数
        """
        # 处理学生输出
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # 教师模型调整和锐化
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]

        # 创建Sinkhorn-Knopp算法实例
        sk = SinkhornKnopp(num_iters=num_iters, epsilon=epsilon)

        # 应用Sinkhorn-Knopp获取分配矩阵
        logits_sk = sk(teacher_output / temp)

        # 计算标准的softmax分布
        teacher_out_softmax = F.softmax(teacher_output / temp, dim=-1)

        # 融合softmax和Sinkhorn-Knopp结果
        teacher_out = (1 - self.sinkhorn_weight) * teacher_out_softmax + self.sinkhorn_weight * logits_sk
        teacher_out = teacher_out.detach().chunk(2)

        # 计算蒸馏损失
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # 跳过学生和教师在同一视图上操作的情况
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class EnergyBasedSplitter:
    """
    基于能量的样本分割器，用于区分可能的旧类和新类样本
    """

    def __init__(self, energy_temp=1.0, threshold=None, ratio=0.3, adapt_threshold=True,
                 logger=None, device='cuda'):
        """
        初始化Energy-Based分割器

        Args:
            energy_temp: 计算能量时的温度参数
            threshold: 能量阈值，高于此值的样本被视为新类
            ratio: 当threshold为None时，能量最高的ratio比例的样本被视为新类
            adapt_threshold: 是否在运行时自适应调整阈值
            logger: 日志记录器
            device: 计算设备
        """
        self.energy_temp = energy_temp
        self.threshold = threshold
        self.ratio = ratio
        self.adapt_threshold = adapt_threshold
        self.logger = logger
        self.device = device
        self.session_energy_stats = []

    def compute_energy(self, logits):
        """
        计算样本的能量分数
        低能量 -> 高置信度 -> 可能是old类
        高能量 -> 低置信度 -> 可能是new类

        Args:
            logits: 模型输出的logits [batch_size, num_classes]

        Returns:
            energy: 每个样本的能量分数 [batch_size]
        """
        return -torch.logsumexp(logits / self.energy_temp, dim=1)

    def split_batch_by_energy(self, student_out, teacher_out=None):
        """
        根据能量分数将批次样本分为old和new两部分

        Args:
            student_out: 学生模型输出的logits [batch_size, num_classes]
            teacher_out: 教师模型输出，可选

        Returns:
            old_indices: old类样本的索引
            new_indices: new类样本的索引
            energy: 所有样本的能量分数
        """
        # 计算能量分数
        energy = self.compute_energy(student_out)

        # 确定阈值
        current_threshold = self.threshold
        if current_threshold is None or self.adapt_threshold:
            # 根据ratio参数自动确定阈值
            sorted_energy, _ = torch.sort(energy, descending=True)
            threshold_idx = int(len(sorted_energy) * self.ratio)
            current_threshold = sorted_energy[threshold_idx]

            # 如果启用自适应阈值，更新类属性
            if self.adapt_threshold:
                self.threshold = current_threshold * 0.9 + (self.threshold or 0) * 0.1  # 平滑更新

        # 大于阈值的被认为是new类样本（高能量）
        new_mask = energy > current_threshold
        old_mask = ~new_mask

        old_indices = torch.where(old_mask)[0]
        new_indices = torch.where(new_mask)[0]

        # 记录统计信息
        self.session_energy_stats.append({
            'mean_energy': energy.mean().item(),
            'min_energy': energy.min().item(),
            'max_energy': energy.max().item(),
            'threshold': current_threshold,
            'old_ratio': len(old_indices) / len(energy),
            'new_ratio': len(new_indices) / len(energy)
        })

        return old_indices, new_indices, energy

    def split_dataset_by_energy(self, model, data_loader, num_seen_classes):
        """
        预处理整个数据集，计算能量并将其分为old和new两部分

        Args:
            model: 当前模型(或上一个session的模型)
            data_loader: 数据加载器
            num_seen_classes: 已知类别的数量

        Returns:
            old_indices_dataset: old类样本在数据集中的索引
            new_indices_dataset: new类样本在数据集中的索引
            energy_scores: 所有样本的能量分数
        """
        model.eval()
        all_energies = []
        all_preds = []
        all_indices = []

        if self.logger:
            self.logger.info("计算整个数据集的能量分数，区分old和new类...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Computing energy scores")):
                # 获取图像和索引
                if len(batch) >= 3:
                    images, _, indices = batch[:3]
                    if isinstance(images, list):  # 对比学习中常见的多视图输入
                        images = images[0]
                else:
                    continue  # 跳过格式不正确的批次

                images = images.to(self.device)

                # 获取模型输出
                _, logits = model(images)

                # 计算能量
                energy = self.compute_energy(logits)

                # 保存结果
                all_energies.append(energy.cpu())
                all_preds.append(logits.argmax(1).cpu())
                all_indices.append(indices)

        all_energies = torch.cat(all_energies)
        all_preds = torch.cat(all_preds)
        all_indices = torch.cat(all_indices)

        # 统计预测类别分布
        pred_old = (all_preds < num_seen_classes).sum().item()
        pred_new = (all_preds >= num_seen_classes).sum().item()

        if self.logger:
            self.logger.info(f"模型预测: {pred_old} 个样本属于old类, {pred_new} 个样本属于new类")
            self.logger.info(
                f"能量统计: 平均={all_energies.mean().item():.4f}, 最小={all_energies.min().item():.4f}, 最大={all_energies.max().item():.4f}")

        # 使用能量分数区分old和new
        if self.threshold is None:
            # 根据ratio参数自动确定阈值
            sorted_energy, _ = torch.sort(all_energies, descending=True)
            threshold_idx = int(len(sorted_energy) * self.ratio)
            self.threshold = sorted_energy[threshold_idx].item()

            if self.logger:
                self.logger.info(
                    f"自动确定能量阈值: {self.threshold:.4f} (选取能量最高的{self.ratio * 100:.1f}%样本作为new类)")

        # 大于阈值的被认为是new类样本（高能量）
        new_mask = all_energies > self.threshold
        old_mask = ~new_mask

        old_indices_dataset = all_indices[old_mask].numpy()
        new_indices_dataset = all_indices[new_mask].numpy()

        if self.logger:
            self.logger.info(
                f"能量分割结果: {len(old_indices_dataset)} 个样本归为old类, {len(new_indices_dataset)} 个样本归为new类")

            # 计算能量分割与模型预测的一致性
            old_correct = ((all_preds < num_seen_classes) & old_mask).sum().item()
            new_correct = ((all_preds >= num_seen_classes) & new_mask).sum().item()
            agreement = (old_correct + new_correct) / len(all_indices)
            self.logger.info(f"能量分割与模型预测的一致性: {agreement * 100:.2f}%")

        return old_indices_dataset, new_indices_dataset, all_energies.numpy()

    def visualize_energy_distribution(self, energy, save_path=None):
        """
        可视化能量分布，帮助分析能量阈值的选择

        Args:
            energy: 样本的能量分数
            save_path: 保存图像的路径
        """
        if self.logger:
            self.logger.info("生成能量分布图...")

        import matplotlib.pyplot as plt
        import seaborn as sns

        if not isinstance(energy, np.ndarray):
            energy = energy.cpu().numpy()

        # 创建图形
        plt.figure(figsize=(10, 6))

        # 绘制能量分布直方图
        sns.histplot(energy, bins=50, kde=True)

        # 如果有阈值，绘制阈值线
        if self.threshold is not None:
            plt.axvline(x=self.threshold, color='red', linestyle='--',
                        label=f'Threshold: {self.threshold:.4f}')

            # 计算划分比例
            high_energy = np.sum(energy > self.threshold)
            low_energy = np.sum(energy <= self.threshold)
            high_ratio = high_energy / len(energy) * 100
            low_ratio = low_energy / len(energy) * 100

            # 添加文本标注
            plt.text(0.05, 0.95, f"Low Energy (Old): {low_ratio:.1f}%\nHigh Energy (New): {high_ratio:.1f}%",
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        plt.title('Energy Distribution', fontsize=14)
        plt.xlabel('Energy Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 保存图像
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            if self.logger:
                self.logger.info(f"能量分布图已保存至: {save_path}")

        plt.close()


def apply_dual_distill_losses(student_out, teacher_out, epoch,
                              original_distill_loss, enhanced_distill_loss,
                              energy_splitter):
    """
    基于能量分割，分别对old和new样本应用不同的蒸馏损失

    Args:
        student_out: 学生模型输出 [batch_size, num_classes]
        teacher_out: 教师模型输出 [batch_size, num_classes]
        epoch: 当前训练的epoch
        original_distill_loss: 原始蒸馏损失函数对象(DistillLoss实例)
        enhanced_distill_loss: 增强蒸馏损失函数对象(EnhancedDistillLoss实例)
        energy_splitter: EnergyBasedSplitter实例

    Returns:
        total_loss: 组合的蒸馏损失
        metrics: 包含分割和损失信息的字典
    """
    # 分割样本
    old_indices, new_indices, energy = energy_splitter.split_batch_by_energy(student_out)

    # 计算样本数量
    total_samples = len(student_out)
    num_old = len(old_indices)
    num_new = len(new_indices)

    # 如果全部是old或全部是new
    if num_old == 0:
        # 全部是new类
        loss = enhanced_distill_loss(student_out, teacher_out, epoch)
        return loss, {
            'old_samples': 0,
            'new_samples': total_samples,
            'old_ratio': 0.0,
            'new_ratio': 1.0,
            'mean_energy': energy.mean().item(),
            'is_mixed_batch': False
        }
    elif num_new == 0:
        # 全部是old类
        loss = original_distill_loss(student_out, teacher_out, epoch)
        return loss, {
            'old_samples': total_samples,
            'new_samples': 0,
            'old_ratio': 1.0,
            'new_ratio': 0.0,
            'mean_energy': energy.mean().item(),
            'is_mixed_batch': False
        }

    # 分离old和new样本
    old_student_out = student_out[old_indices]
    old_teacher_out = teacher_out[old_indices]
    new_student_out = student_out[new_indices]
    new_teacher_out = teacher_out[new_indices]

    # 分别计算损失
    old_loss = original_distill_loss(old_student_out, old_teacher_out, epoch)
    new_loss = enhanced_distill_loss(new_student_out, new_teacher_out, epoch)

    # 计算加权总损失
    old_weight = num_old / total_samples
    new_weight = num_new / total_samples
    total_loss = old_weight * old_loss + new_weight * new_loss

    return total_loss, {
        'old_samples': num_old,
        'new_samples': num_new,
        'old_ratio': old_weight,
        'new_ratio': new_weight,
        'old_loss': old_loss.item(),
        'new_loss': new_loss.item(),
        'mean_energy': energy.mean().item(),
        'energy_threshold': energy_splitter.threshold,
        'is_mixed_batch': True
    }