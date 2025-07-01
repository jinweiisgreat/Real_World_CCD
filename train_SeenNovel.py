"""
Modify：添加 Prompt Pool 类 + Prompt训练机制
date:   2025/05/12
author: Wei Jin

update: prompt pool update + prompt training
date:   2025/05/16
author: Wei Jin

update: + prompt training
date:   2025/06/02
author: Wei Jin
"""

import argparse
import os
import math
from tqdm import tqdm
from copy import deepcopy

import numpy as np
from torch.utils.data import DataLoader
import torch

torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
from torch.optim import SGD, lr_scheduler

from project_utils.general_utils import set_seed, init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets

from models.utils_simgcd import DINOHead, get_params_groups, SupConLoss, info_nce_logits, DistillLoss
from models.utils_simgcd_pro_prompt_trainable import get_kmeans_centroid_for_new_head
from models.utils_proto_aug_SeenNovel import ProtoAugManager
from models import vision_transformer as vits
from config import dino_pretrain_path, exp_root
from collections import Counter
# import Enhanced PromptPool and Model
from models.utils_prompt_pool_trainable import LearnablePromptPool, visualize_graph_network
from models.prompt_enhanced_model_trainable import PromptEnhancedModel


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import setproctitle

setproctitle.setproctitle("xiao wei's python process")

'''offline train and test'''
'''====================================================================================================================='''


def train_offline(student, train_loader, test_loader, args):
    """
    改进的离线训练函数，支持prompt pool的任务驱动训练
    """
    assert isinstance(student, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(student)}"

    # 获取所有需要优化的参数
    model_params = list(student.backbone.parameters()) + list(student.projector.parameters())
    prompt_params = student.get_prompt_parameters()

    if prompt_params:
        args.logger.info(f"Found {len(prompt_params)} prompt parameters to optimize")
    else:
        args.logger.info("No prompt parameters found, using only model parameters")

    # 使用分组参数策略
    params_groups = get_params_groups(student)

    # 如果有prompt参数，单独添加到优化器，使用更积极的学习率
    if prompt_params:
        prompt_lr = args.lr * 0.2  # 提高prompt学习率到主学习率的0.2倍
        params_groups.append({
            'params': prompt_params,
            'lr': prompt_lr,
            'weight_decay': 0.0  # prompt不使用weight decay
        })
        args.logger.info(f"Added prompt parameters with lr={prompt_lr}")

    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs_offline,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs_offline,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    best_test_acc_old = 0

    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()
        prompt_loss_record = AverageMeter()
        main_loss_record = AverageMeter()

        student.train()
        # 启用prompt学习
        if student.prompt_pool is not None:
            student.enable_prompt_learning()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs = batch
            mask_lab = torch.ones_like(class_labels)  # 所有样本都是有标签的

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # 使用改进的forward方法，获取prompt损失和原始logits用于对比
            student_proj, student_out, prompt_losses, original_logits = student.forward_with_prompt_loss(
                images, class_labels
            )
            teacher_out = student_out.detach()

            # ========== 主任务损失计算 ==========
            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup: SimGCD Eq.(4)
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup: SimGCD Eq.(1)
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup: SimGCD Eq.(2)
            student_proj_reshaped = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj_reshaped = torch.nn.functional.normalize(student_proj_reshaped, dim=-1)

            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj_reshaped, labels=sup_con_labels)

            # 主任务损失
            main_loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            main_loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # ========== Prompt损失计算 ==========
            # 动态调整prompt权重：早期较小，随训练逐渐增加
            prompt_weight_schedule = min(0.5, (epoch + 1) / (args.epochs_offline * 0.4))
            total_prompt_loss = prompt_weight_schedule * prompt_losses['total']

            # 总损失
            total_loss = main_loss + total_prompt_loss

            # ========== 反向传播和优化 ==========
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录损失
            loss_record.update(total_loss.item(), class_labels.size(0))
            prompt_loss_record.update(total_prompt_loss.item(), class_labels.size(0))
            main_loss_record.update(main_loss.item(), class_labels.size(0))

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t Total Loss: {:.5f}\t Main Loss: {:.5f}\t Prompt Loss: {:.5f}'
                                 .format(epoch, batch_idx, len(train_loader), total_loss.item(),
                                         main_loss.item(), total_prompt_loss.item()))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} Main Loss: {:.4f} Prompt Loss: {:.4f}'.format(
            epoch, loss_record.avg, main_loss_record.avg, prompt_loss_record.avg))

        # 记录prompt pool信息
        if student.prompt_pool is not None:
            prompt_info = student.prompt_pool_summary()
            args.logger.info(f'Prompt Pool: {prompt_info}')

        # 测试
        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

        # 学习率调度
        exp_lr_scheduler.step()

        # 保存模型
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc_test > best_test_acc_old:
            args.logger.info(f'Best ACC on Old Classes on test set: {old_acc_test:.4f}...')
            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            best_test_acc_old = old_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: Old: {best_test_acc_old:.4f}')
        args.logger.info('\n')

    # 训练结束后创建prompt pool
    args.logger.info("Creating prompt pool from offline training data...")

    # 使用干净的数据加载器进行特征提取
    clean_dataset = deepcopy(train_loader.dataset)
    clean_dataset.transform = test_transform
    clean_loader = DataLoader(
        clean_dataset,
        num_workers=args.num_workers_test,
        batch_size=256,
        shuffle=False,
        pin_memory=False
    )

    # 如果prompt pool还没有初始化，现在进行初始化
    if args.prompt_pool.num_prompts == 0:
        prompt_pool_stats = args.prompt_pool.create_prompt_pool(
            model=student,
            data_loader=clean_loader,
            num_classes=args.num_labeled_classes,
            logger=args.logger
        )
    else:
        # 如果已经有了prompt pool，进行更新
        prompt_pool_stats = args.prompt_pool.update_prompt_pool_incrementally(
            model=student,
            data_loader=clean_loader,
            logger=args.logger
        )

    # 保存prompt pool
    prompt_pool_path = os.path.join(args.model_dir, 'prompt_pool.pt')
    args.prompt_pool.save_prompt_pool(prompt_pool_path)
    args.logger.info(f"Prompt pool saved to {prompt_pool_path}")

    # 保存统计信息
    stats_path = os.path.join(args.model_dir, 'prompt_pool_stats.pt')
    torch.save({
        'prompt_pool_stats': prompt_pool_stats
    }, stats_path)

    return best_test_acc_old


def test_offline(model, test_loader, epoch, save_name, args):
    """
    改进的离线测试函数，支持prompt效果分析
    """
    assert isinstance(model, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(model)}"

    model.eval()

    preds, targets = [], []
    mask = np.array([])

    # 用于统计prompt效果
    total_effectiveness_stats = {
        'samples_helped': 0,
        'samples_hurt': 0,
        'total_samples': 0,
        'accuracy_improvement': 0.0,
        'confidence_improvement': 0.0
    }

    # 提取所有特征
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            # 获取预测结果和效果分析
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 输出prompt效果统计
    if total_effectiveness_stats['total_samples'] > 0:
        avg_accuracy_improvement = total_effectiveness_stats['accuracy_improvement'] / total_effectiveness_stats[
            'total_samples']
        avg_confidence_improvement = total_effectiveness_stats['confidence_improvement'] / total_effectiveness_stats[
            'total_samples']
        net_help_ratio = (total_effectiveness_stats['samples_helped'] - total_effectiveness_stats['samples_hurt']) / \
                         total_effectiveness_stats['total_samples']

        args.logger.info(f'Test Prompt Effectiveness - Epoch {epoch}: '
                         f'Acc Improvement: {avg_accuracy_improvement:.4f}, '
                         f'Conf Improvement: {avg_confidence_improvement:.4f}, '
                         f'Net Help Ratio: {net_help_ratio:.4f}, '
                         f'Helped: {total_effectiveness_stats["samples_helped"]}, '
                         f'Hurt: {total_effectiveness_stats["samples_hurt"]}')

    # 评估
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


'''====================================================================================================================='''

'''online train and test'''
'''====================================================================================================================='''


def train_online(student, student_pre, proto_aug_manager, train_loader, test_loader, current_session, args):
    """
    集成seen novel对比学习的改进在线训练函数
    """
    assert isinstance(student, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(student)}"
    assert isinstance(student_pre, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(student_pre)}"

    # ========== 训练前的原型析出（新增） ==========
    seen_novel_prototypes = None
    distillation_info = {}

    if current_session > 1:
        args.logger.info(f"\n=== Session {current_session}: 开始原型析出 ===")

        # 步骤1: 原型析出 - 匈牙利匹配识别seen novel原型
        # seen_novel_prototypes, distillation_info = proto_aug_manager.prototype_distillation_with_hungarian(
        #     student, args.num_labeled_classes, train_loader
        # )

        seen_novel_prototypes, distillation_info = proto_aug_manager.prototype_distillation_with_hungarian(
            student, train_loader
        )

        args.logger.info(f"原型析出完成: {distillation_info}")

        if seen_novel_prototypes is not None:
            args.logger.info(f"✓ 成功识别 {len(seen_novel_prototypes)} 个seen novel原型")
            # 设置到proto_aug_manager中用于后续对比学习
            proto_aug_manager.current_seen_novel_prototypes = seen_novel_prototypes
        else:
            args.logger.info("✗ 未发现seen novel原型")
            proto_aug_manager.current_seen_novel_prototypes = None

    # 加载prompt pool
    if hasattr(args, 'prompt_pool') and args.prompt_pool is not None:
        if current_session == 1:
            # 第一个会话使用离线阶段的prompt pool
            offline_model_dir = os.path.join(exp_root + '_offline', args.dataset_name, args.load_offline_id,
                                             'checkpoints')
            prompt_pool_path = os.path.join(offline_model_dir, 'prompt_pool.pt')
        else:
            # 后续会话使用上一个会话更新后的prompt pool
            prompt_pool_path = os.path.join(args.model_dir, f'prompt_pool_session_{current_session - 1}.pt')

        if os.path.exists(prompt_pool_path):
            args.logger.info(f"Loading prompt pool from {prompt_pool_path}")
            args.prompt_pool.load_prompt_pool(prompt_pool_path, device=args.device)
            args.logger.info(f"Loaded Prompt Pool with {args.prompt_pool.num_prompts} prompts")
        else:
            args.logger.warning(f"Prompt pool not found at {prompt_pool_path}")

    # 获取所有需要优化的参数
    model_params = list(student.backbone.parameters()) + list(student.projector.parameters())
    prompt_params = student.get_prompt_parameters()

    if prompt_params:
        args.logger.info(f"Found {len(prompt_params)} prompt parameters to optimize")
    else:
        args.logger.info("No prompt parameters found, using only model parameters")

    # 使用分组参数策略
    params_groups = get_params_groups(student)

    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs_online_per_session,
        eta_min=args.lr * 1e-3,
    )

    # 损失函数
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs_online_per_session,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # 记录最佳精度
    best_test_acc_all = 0
    best_test_acc_old = 0
    best_test_acc_new = 0
    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0

    # 用于监控prompt效果和seen novel效果
    online_prompt_effectiveness_history = []
    seen_novel_learning_history = []

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()
        prompt_loss_record = AverageMeter()
        main_loss_record = AverageMeter()
        seen_novel_loss_record = AverageMeter()  # 新增：seen novel损失记录

        # 用于统计prompt效果
        epoch_effectiveness_stats = {
            'samples_helped': 0,
            'samples_hurt': 0,
            'total_samples': 0,
            'accuracy_improvement': 0.0,
            'confidence_improvement': 0.0
        }

        student.train()
        student_pre.eval()

        # 启用prompt学习
        if student.prompt_pool is not None:
            student.enable_prompt_learning()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, _ = batch
            mask_lab = torch.zeros_like(class_labels)  # 所有样本都是无标签的

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # 使用改进的forward方法
            student_proj, student_out, prompt_losses, original_logits = student.forward_with_prompt_loss(images,
                                                                                                         class_labels)
            teacher_out = student_out.detach()

            # ========== 主任务损失计算 ==========
            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)

            # 分组熵正则化
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)

            # 1. inter old and new
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new = avg_probs_old_marginal * torch.log(
                avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. old (intra) & new (intra)
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm ** (-avg_probs_old_in_norm))) + math.log(
                float(len(avg_probs_old_in_norm)))
            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(
                    torch.log(avg_probs_new_in_norm ** (-avg_probs_new_in_norm))) + math.log(
                    float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=args.device)

            # 总熵损失
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                            args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in

            # 对比学习
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # ProtoAug损失
            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss(student)

            # 特征蒸馏
            feats = student.backbone(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            with torch.no_grad():
                feats_pre = student_pre.backbone(images)
                feats_pre = torch.nn.functional.normalize(feats_pre, dim=-1)

            feat_distill_loss = (feats - feats_pre).pow(2).sum() / len(feats)

            # ========== Seen Novel对比学习损失（新增核心部分） ==========
            seen_novel_loss = torch.tensor(0.0, device=args.device)
            seen_novel_details = {}

            if (proto_aug_manager.current_seen_novel_prototypes is not None and
                    len(proto_aug_manager.current_seen_novel_prototypes) > 0):
                # 步骤2&3: 软分配 + 对比学习
                seen_novel_loss, seen_novel_details = proto_aug_manager.seen_novel_cl.compute_comprehensive_seen_novel_loss(
                    feats, proto_aug_manager.current_seen_novel_prototypes, student
                )

            # 主任务损失组合
            main_loss = 1 * cluster_loss + 1 * contrastive_loss
            main_loss += args.proto_aug_weight * proto_aug_loss + args.feat_distill_weight * feat_distill_loss

            # 添加seen novel对比学习损失
            seen_novel_weight = getattr(args, 'seen_novel_weight', 0.5)  # 默认权重0.5
            main_loss += seen_novel_weight * seen_novel_loss

            # ========== Prompt损失计算 ==========
            # 在线阶段更积极地训练prompt
            prompt_weight = min(1.0, (epoch + 1) / (args.epochs_online_per_session * 0.15))  # 更快达到最大权重
            total_prompt_loss = prompt_weight * prompt_losses['total']

            # 总损失
            total_loss = main_loss + total_prompt_loss

            # ========== 反向传播和优化 ==========
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录损失
            loss_record.update(total_loss.item(), class_labels.size(0))
            prompt_loss_record.update(total_prompt_loss.item(), class_labels.size(0))
            main_loss_record.update(main_loss.item(), class_labels.size(0))
            seen_novel_loss_record.update(seen_novel_loss.item(), class_labels.size(0))

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t Total Loss: {:.5f}\t Main Loss: {:.5f}\t '
                                 'Prompt Loss: {:.5f}\t SeenNovel Loss: {:.5f}'
                                 .format(epoch, batch_idx, len(train_loader), total_loss.item(),
                                         main_loss.item(), total_prompt_loss.item(), seen_novel_loss.item()))

                # 输出seen novel详细信息
                if seen_novel_details and 'assignment_stats' in seen_novel_details:
                    assign_stats = seen_novel_details['assignment_stats']
                    if assign_stats.get('total_samples', 0) > 0:
                        args.logger.info(f'SeenNovel Stats: Assignment Rate: {assign_stats["assignment_rate"]:.3f}, '
                                         f'Avg Confidence: {assign_stats["avg_confidence"]:.3f}')

                # 输出预测比例信息
                new_true_ratio = len(class_labels[class_labels >= args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds >= args.num_seen_classes]) / len(preds)
                args.logger.info(
                    f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | '
                    f'Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | '
                    f'Pred new ratio: {new_pred_ratio:.4f} | '
                    f'Ground-truth new ratio: {new_true_ratio:.4f}')

        # ========== Epoch总结 ==========
        # 计算平均效果统计
        if epoch_effectiveness_stats['total_samples'] > 0:
            avg_accuracy_improvement = epoch_effectiveness_stats['accuracy_improvement'] / epoch_effectiveness_stats[
                'total_samples']
            avg_confidence_improvement = epoch_effectiveness_stats['confidence_improvement'] / \
                                         epoch_effectiveness_stats['total_samples']
            net_help_ratio = (epoch_effectiveness_stats['samples_helped'] - epoch_effectiveness_stats['samples_hurt']) / \
                             epoch_effectiveness_stats['total_samples']

            online_prompt_effectiveness_history.append({
                'session': current_session,
                'epoch': epoch,
                'avg_accuracy_improvement': avg_accuracy_improvement,
                'avg_confidence_improvement': avg_confidence_improvement,
                'net_help_ratio': net_help_ratio,
                'samples_helped': epoch_effectiveness_stats['samples_helped'],
                'samples_hurt': epoch_effectiveness_stats['samples_hurt']
            })

            args.logger.info(f'Online Prompt Effectiveness - Session {current_session}, Epoch {epoch}: '
                             f'Acc Improvement: {avg_accuracy_improvement:.4f}, '
                             f'Conf Improvement: {avg_confidence_improvement:.4f}, '
                             f'Net Help Ratio: {net_help_ratio:.4f}')

        # 记录seen novel学习历史
        if seen_novel_details:
            seen_novel_learning_history.append({
                'session': current_session,
                'epoch': epoch,
                'seen_novel_loss': seen_novel_loss_record.avg,
                'details': seen_novel_details
            })

        args.logger.info(
            'Train Epoch: {} Avg Loss: {:.4f} Main Loss: {:.4f} Prompt Loss: {:.4f} SeenNovel Loss: {:.4f}'.format(
                epoch, loss_record.avg, main_loss_record.avg, prompt_loss_record.avg, seen_novel_loss_record.avg))

        # 更新seen novel学习策略（新增）
        if proto_aug_manager.current_seen_novel_prototypes is not None:
            proto_aug_manager.update_seen_novel_learning_strategy(epoch)

        # 记录prompt pool信息
        if student.prompt_pool is not None:
            prompt_info = student.prompt_pool_summary()
            args.logger.info(f'Prompt Pool: {prompt_info}')

        # 测试
        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, \
            all_acc_soft_test, seen_acc_test, unseen_acc_test = test_online(student, test_loader, epoch=epoch,
                                                                            save_name='Test ACC',
                                                                            current_session=current_session, args=args)
        args.logger.info(
            'Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                  new_acc_test))
        args.logger.info(
            'Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test,
                                                                                      unseen_acc_test))

        # 学习率调度
        exp_lr_scheduler.step()

        # 保存模型
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'online_prompt_effectiveness_history': online_prompt_effectiveness_history,
            'seen_novel_learning_history': seen_novel_learning_history,  # 新增
            'distillation_info': distillation_info,  # 新增
            'proto_aug_stats': proto_aug_manager.get_comprehensive_statistics() if hasattr(proto_aug_manager,
                                                                                           'get_comprehensive_statistics') else {}
        }

        if all_acc_test > best_test_acc_all:
            args.logger.info(f'Best ACC on All Classes on test set of session-{current_session}: {all_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt')
            args.logger.info(
                "model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt'))

            best_test_acc_all = all_acc_test
            best_test_acc_old = old_acc_test
            best_test_acc_new = new_acc_test
            best_test_acc_soft_all = all_acc_soft_test
            best_test_acc_seen = seen_acc_test
            best_test_acc_unseen = unseen_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(
            f'Metrics with best model on test set (Hard) of session-{current_session}: '
            f'All: {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
        args.logger.info(
            f'Metrics with best model on test set (Soft) of session-{current_session}: '
            f'All: {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        args.logger.info('\n')

    # 更新记录
    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)

    # ========== Session结束后的总结（新增） ==========
    if proto_aug_manager.current_seen_novel_prototypes is not None:
        # 总结seen novel学习效果
        final_stats = proto_aug_manager.seen_novel_cl.assignment_stats
        args.logger.info(f"\n=== Session {current_session} Seen Novel Learning Summary ===")
        args.logger.info(f"总处理样本: {final_stats['total_samples']}")
        args.logger.info(f"分配样本数: {final_stats['assigned_samples']}")
        args.logger.info(f"分配成功率: {final_stats['assignment_rate']:.3f}")
        args.logger.info(f"平均置信度: {final_stats['avg_confidence']:.3f}")

        # 重置seen novel学习统计
        proto_aug_manager.seen_novel_cl.reset_statistics()

    # ========== 更新prompt pool ==========
    if hasattr(args, 'prompt_pool') and args.prompt_pool is not None:
        args.logger.info(f"Incrementally updating prompt pool after session {current_session}...")

        # 准备用于更新的数据集
        clean_dataset = deepcopy(train_loader.dataset)
        clean_dataset.transform = test_transform
        clean_loader = DataLoader(
            clean_dataset,
            num_workers=args.num_workers_test,
            batch_size=256,
            shuffle=False,
            pin_memory=False
        )

        # 获取当前会话的最佳模型
        best_model_path = args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt'
        best_model = student  # 默认使用当前模型

        if os.path.exists(best_model_path):
            try:
                state_dict = torch.load(best_model_path)['model']
                # 创建临时模型用于特征提取
                temp_enhanced_model = PromptEnhancedModel(
                    backbone=deepcopy(backbone),
                    projector=deepcopy(projector_cur),
                    prompt_pool=args.prompt_pool,
                    top_k=5
                )
                temp_enhanced_model.load_state_dict(state_dict)
                temp_enhanced_model = temp_enhanced_model.to(device)
                best_model = temp_enhanced_model
                args.logger.info(f"Loaded best model for prompt pool update from {best_model_path}")
            except Exception as e:
                args.logger.warning(f"Failed to load best model: {e}")

        # 增量更新prompt pool
        similarity_threshold = getattr(args, 'prompt_update_threshold', 0.8)
        ema_alpha = getattr(args, 'prompt_ema_alpha', 0.9)

        update_stats = args.prompt_pool.update_prompt_pool_incrementally(
            model=best_model,
            data_loader=clean_loader,
            similarity_threshold=similarity_threshold,
            ema_alpha=ema_alpha,
            logger=args.logger
        )

        # 保存更新后的prompt pool
        prompt_pool_path = os.path.join(args.model_dir, f'prompt_pool_session_{current_session}.pt')
        args.prompt_pool.save_prompt_pool(prompt_pool_path)
        args.logger.info(f"Updated prompt pool saved to {prompt_pool_path}")

        # 保存综合统计信息（包含seen novel信息）
        update_stats_path = os.path.join(args.model_dir, f'comprehensive_stats_session_{current_session}.pt')
        comprehensive_stats = {
            'prompt_update_stats': update_stats,
            'online_prompt_effectiveness_history': online_prompt_effectiveness_history,
            'seen_novel_learning_history': seen_novel_learning_history,
            'distillation_info': distillation_info,
            'final_seen_novel_stats': proto_aug_manager.seen_novel_cl.assignment_stats,
        }
        torch.save(comprehensive_stats, update_stats_path)
        args.logger.info(f"Comprehensive statistics saved to {update_stats_path}")
        args.logger.info(f"Session {current_session} ends with {args.prompt_pool.num_prompts} prompts in pool")

    return best_test_acc_all


def test_online(model, test_loader, epoch, save_name, current_session, args):
    """
    改进的在线测试函数，支持prompt效果分析
    """
    assert isinstance(model, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(model)}"

    model.eval()

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])

    # 预测分布统计
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    class_prediction_counts = np.zeros(num_classes)

    # 用于统计prompt效果
    total_effectiveness_stats = {
        'samples_helped': 0,
        'samples_hurt': 0,
        'total_samples': 0,
        'accuracy_improvement': 0.0,
        'confidence_improvement': 0.0
    }

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        # 处理不同格式的数据加载器
        if len(batch) == 2:
            images, label = batch
        elif len(batch) >= 3:
            images, label, _ = batch[:3]

        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            # 直接使用PromptEnhancedModel的forward方法
            _, logits = model(images)

            batch_preds = logits.argmax(1).cpu().numpy()
            preds.append(batch_preds)
            targets.append(label.cpu().numpy())

            # 更新预测计数
            for pred in batch_preds:
                class_prediction_counts[pred] += 1

            # 创建不同评估指标的掩码
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                                       else False for x in label]))
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                                       else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 输出prompt效果统计
    if total_effectiveness_stats['total_samples'] > 0:
        avg_accuracy_improvement = total_effectiveness_stats['accuracy_improvement'] / total_effectiveness_stats[
            'total_samples']
        avg_confidence_improvement = total_effectiveness_stats['confidence_improvement'] / total_effectiveness_stats[
            'total_samples']
        net_help_ratio = (total_effectiveness_stats['samples_helped'] - total_effectiveness_stats['samples_hurt']) / \
                         total_effectiveness_stats['total_samples']

        args.logger.info(f'Online Test Prompt Effectiveness - Session {current_session}, Epoch {epoch}: '
                         f'Acc Improvement: {avg_accuracy_improvement:.4f}, '
                         f'Conf Improvement: {avg_confidence_improvement:.4f}, '
                         f'Net Help Ratio: {net_help_ratio:.4f}, '
                         f'Helped: {total_effectiveness_stats["samples_helped"]}, '
                         f'Hurt: {total_effectiveness_stats["samples_hurt"]}')

    # 生成可视化（在最后一个epoch）
    if hasattr(args, 'log_dir') and epoch == args.epochs_online_per_session - 1:
        try:
            # 创建可视化目录
            vis_dir = os.path.join(args.log_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 生成混淆矩阵
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 获取数据中存在的所有类别
            all_classes = sorted(list(set(targets.tolist() + preds.tolist())))

            # 生成混淆矩阵
            cm = confusion_matrix(targets, preds, labels=all_classes)

            # 创建图表
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, fmt='d', cmap='Blues',
                        xticklabels=all_classes,
                        yticklabels=all_classes)

            plt.xlabel('Prediction')
            plt.ylabel('Ground Truth')
            plt.title(f'Confusion Matrix - Session {current_session}')

            # 保存混淆矩阵
            conf_matrix_path = os.path.join(vis_dir, f'confusion_matrix_session_{current_session}.png')
            plt.tight_layout()
            plt.savefig(conf_matrix_path)
            plt.close()
            args.logger.info(f"Confusion matrix saved to: {conf_matrix_path}")

        except Exception as e:
            args.logger.warning(f"Could not generate visualizations: {str(e)}")

    # 计算评估指标
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                             T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                             args=args)

    # 记录结果摘要
    args.logger.info(f"\nTest results summary (Session {current_session}, Epoch {epoch}):")
    args.logger.info(f"Hard metrics: All={all_acc:.4f}, Old={old_acc:.4f}, New={new_acc:.4f}")
    args.logger.info(f"Soft metrics: All={all_acc_soft:.4f}, Seen={seen_acc:.4f}, Unseen={unseen_acc:.4f}")

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc


'''====================================================================================================================='''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', action='store_true', default=False)

    '''group-wise entropy regularization'''
    # memax weight for offline session
    parser.add_argument('--memax_weight', type=float, default=1)
    # memax weight for online session
    parser.add_argument('--memax_old_new_weight', type=float, default=2)
    parser.add_argument('--memax_old_in_weight', type=float, default=1)
    parser.add_argument('--memax_new_in_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature. default = 30')

    '''clustering-guided initialization'''
    parser.add_argument('--init_new_head', action='store_true', default=False)

    '''PASS params'''
    parser.add_argument('--proto_aug_weight', type=float, default=1.0)
    parser.add_argument('--feat_distill_weight', type=float, default=1.0)
    parser.add_argument('--radius_scale', type=float, default=1.0)

    '''hardness-aware sampling temperature'''
    parser.add_argument('--hardness_temp', type=float, default=0.1)

    # Continual GCD params
    parser.add_argument('--num_old_classes', type=int, default=-1)
    parser.add_argument('--prop_train_labels', type=float, default=0.8)
    parser.add_argument('--train_session', type=str, default='offline', help='options: offline, online')
    parser.add_argument('--load_offline_id', type=str, default=None)
    parser.add_argument('--epochs_offline', default=100, type=int)
    parser.add_argument('--epochs_online_per_session', default=30, type=int)
    parser.add_argument('--continual_session_num', default=3, type=int)
    parser.add_argument('--online_novel_unseen_num', default=400, type=int)
    parser.add_argument('--online_old_seen_num', default=50, type=int)
    parser.add_argument('--online_novel_seen_num', default=50, type=int)
    parser.add_argument('--test_mode', type=str, default='cumulative_session',
                        help='options: current_session, cumulative_session')

    # shuffle dataset classes
    parser.add_argument('--shuffle_classes', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)

    # others
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd-pro-v5', type=str)

    '''prompt pool training parameters'''
    parser.add_argument('--enable_prompt_training', action='store_true', default=True,
                        help='Enable prompt pool training')
    parser.add_argument('--prompt_learning_weight', type=float, default=0.1,
                        help='Weight for prompt learning losses')
    parser.add_argument('--prompt_diversity_weight', type=float, default=0.05,
                        help='Weight for prompt diversity loss')
    parser.add_argument('--prompt_alignment_weight', type=float, default=0.1,
                        help='Weight for prompt-feature alignment loss')
    parser.add_argument('--prompt_update_threshold', type=float, default=0.8,
                        help='Similarity threshold for prompt pool updates')
    parser.add_argument('--prompt_ema_alpha', type=float, default=0.9,
                        help='EMA alpha for prompt updates')
    parser.add_argument('--max_prompts', type=int, default=200,
                        help='Maximum number of prompts in the pool')
    parser.add_argument('--prompt_pool', action='store_true', default=True,
                        help='Use prompt pool for feature enhancement')

    parser.add_argument('--seen_novel_weight', type=float, default=0.5,
                        help='Weight for seen novel contrastive learning loss')
    parser.add_argument('--seen_novel_threshold', type=float, default=0.6,
                        help='Similarity threshold for seen novel identification in prototype distillation')
    parser.add_argument('--confidence_threshold', type=float, default=0.65,
                        help='Confidence threshold for soft assignment of seen novel samples')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='Temperature for seen novel contrastive learning')
    parser.add_argument('--alignment_threshold', type=float, default=0.7,
                        help='Similarity threshold for Hungarian alignment of old classes')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    # set_seed(args.seed)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)  # get_dataset

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'happy' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)

    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
                           + '_' + 'ContinualNum' + str(args.continual_session_num) + '_' + 'UnseenNum' + str(
            args.online_novel_unseen_num) \
                           + '_' + 'SeenNum' + str(args.online_novel_seen_num)

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    args.device = device

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()
    args.logger.info(f'Loading weights from {dino_pretrain_path}')
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # 模型参数
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes

    # 初始化可学习的prompt pool
    args.prompt_pool = LearnablePromptPool(
        feature_dim=args.feat_dim,
        similarity_threshold=0.65,  # 降低阈值，更容易匹配
        community_ratio=1.2,  # 减少初始prompt数量，避免过度复杂
        device=device,
        max_prompts=args.max_prompts
    )

    # 设置prompt pool的训练权重
    args.prompt_pool.prompt_learning_weight = args.prompt_learning_weight
    args.prompt_pool.diversity_weight = args.prompt_diversity_weight
    args.prompt_pool.alignment_weight = args.prompt_alignment_weight

    # 设置backbone哪些层需要微调
    for m in backbone.parameters():
        m.requires_grad = False

    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # 创建projector
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = PromptEnhancedModel(
        backbone=backbone,
        projector=projector,
        prompt_pool=args.prompt_pool,
        top_k=5,
        enable_prompt_training=args.enable_prompt_training
    )

    model.to(device)

    args.logger.info(f"Model created with prompt training {'enabled' if args.enable_prompt_training else 'disabled'}")
    if args.enable_prompt_training:
        args.logger.info(f"Prompt pool max size: {args.max_prompts}")
        args.logger.info(
            f"Prompt learning weights - main: {args.prompt_learning_weight}, diversity: {args.prompt_diversity_weight}, alignment: {args.prompt_alignment_weight}")

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # ----------------------
    # 1. OFFLINE TRAIN
    # ----------------------
    if args.train_session == 'offline':
        args.logger.info('========== offline training with labeled old data (old) ==========')
        args.logger.info('loading dataset...')
        offline_session_train_dataset, offline_session_test_dataset, \
            _online_session_train_dataset_list, _online_session_test_dataset_list, \
            datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
            args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'offline_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('offline_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.close()

        offline_session_train_loader = DataLoader(offline_session_train_dataset, num_workers=args.num_workers,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  pin_memory=True)
        offline_session_test_loader = DataLoader(offline_session_test_dataset, num_workers=args.num_workers_test,
                                                 batch_size=256, shuffle=False, pin_memory=False)

        # ----------------------
        # TRAIN
        # ----------------------
        train_offline(model, offline_session_train_loader, offline_session_test_loader, args)


    # ----------------------
    # 2. ONLINE TRAIN
    # ----------------------
    elif args.train_session == 'online':
        args.logger.info(
            '\n\n==================== online continual GCD with unlabeled data (old + novel) ====================')
        args.logger.info('loading dataset...')
        _offline_session_train_dataset, _offline_session_test_dataset, \
            online_session_train_dataset_list, online_session_test_dataset_list, \
            datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
            args.dataset_name, train_transform, test_transform, args)

        # saving dataset dict
        print('save dataset dict...')
        save_dataset_dict_path = os.path.join(args.log_dir, 'online_dataset_dict.txt')
        f_dataset_dict = open(save_dataset_dict_path, 'w')
        f_dataset_dict.write('online_dataset_split_dict: \n')
        f_dataset_dict.write(str(dataset_split_config_dict))
        f_dataset_dict.write('\nnovel_targets_shuffle: \n')
        f_dataset_dict.write(str(novel_targets_shuffle))
        f_dataset_dict.write('\nnum_novel_class_per_session: \n')
        f_dataset_dict.write(str(args.num_unlabeled_classes // args.continual_session_num))
        f_dataset_dict.close()

        # ----------------------
        # CONTINUAL SESSIONS
        # ----------------------
        args.num_novel_class_per_session = args.num_unlabeled_classes // args.continual_session_num
        args.logger.info('number of novel class per session: {}'.format(args.num_novel_class_per_session))

        '''
        v5: ProtoAug Manager
        初始化一个ProtoAugManager实例
        '''
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views * args.batch_size, args.hardness_temp,
                                            args.radius_scale, device, args.logger)

        # best test acc list across continual sessions
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []

        start_session = 0

        '''Continual GCD sessions'''
        for session in range(start_session, args.continual_session_num):
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session + 1))
            # dataset for the current session
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]
            online_session_train_loader = DataLoader(online_session_train_dataset, num_workers=args.num_workers,
                                                     batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                     pin_memory=True)
            online_session_test_loader = DataLoader(online_session_test_dataset, num_workers=args.num_workers_test,
                                                    batch_size=256, shuffle=False, pin_memory=False)

            # number of seen (offline old + previous online new) classes till the beginning of this session
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(
                args.num_seen_classes))

            if args.dataset_name == 'cifar100':
                args.num_cur_novel_classes = len(
                    np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets))
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in
                                    enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in
                                    enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'scars':
                args.num_cur_novel_classes = len(
                    np.unique(online_session_train_dataset.novel_unlabelled_dataset.target))
            else:
                args.num_cur_novel_classes = args.num_novel_class_per_session * (session + 1)
            args.logger.info('number of all novel class (seen novel + unseen novel) in current session: {}'.format(
                args.num_cur_novel_classes))

            '''tunable params in backbone'''
            ####################################################################################################################
            # freeze backbone params
            for m in backbone.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in backbone.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
            ####################################################################################################################

            '''load ckpts from last session (session>0) or offline session (session=0)'''
            ####################################################################################################################
            args.logger.info('loading checkpoints of model_pre...')
            if session == 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes,
                                         nlayers=args.num_mlp_layers)
                model_pre = PromptEnhancedModel(
                    backbone=deepcopy(backbone),
                    projector=projector_pre,
                    prompt_pool=args.prompt_pool,
                    top_k=5,
                    enable_prompt_training=args.enable_prompt_training
                )
                if args.load_offline_id is not None:
                    load_dir_online = os.path.join(exp_root + '_' + 'offline', args.dataset_name, args.load_offline_id,
                                                   'checkpoints', 'model_best.pt')
                    args.logger.info('loading offline checkpoints from: ' + load_dir_online)
                    load_dict = torch.load(load_dir_online)

                    if list(load_dict['model'].keys())[0].startswith('0'):  # 检测Sequential格式
                        new_state_dict = {}
                        for k, v in load_dict['model'].items():
                            if k.startswith('0.'):
                                new_state_dict['backbone' + k[1:]] = v
                            elif k.startswith('1.'):
                                new_state_dict['projector' + k[1:]] = v
                        load_dict['model'] = new_state_dict

                    model_pre.load_state_dict(load_dict['model'])
                    args.logger.info('successfully loaded checkpoints!')

            else:  # session > 0
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_seen_classes,
                                         nlayers=args.num_mlp_layers)
                model_pre = PromptEnhancedModel(
                    backbone=deepcopy(backbone),
                    projector=projector_pre,
                    prompt_pool=args.prompt_pool,
                    top_k=5,
                    enable_prompt_training=args.enable_prompt_training
                )
                load_dir_online = args.model_path[:-3] + '_session-' + str(session) + f'_best.pt'
                args.logger.info('loading checkpoints from last online session: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded checkpoints!')
            ####################################################################################################################

            '''incremental parametric classifier in SimGCD'''
            ####################################################################################################################
            backbone_cur = deepcopy(backbone)
            backbone_cur.load_state_dict(model_pre.backbone.state_dict())
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))
            projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim_cur, nlayers=args.num_mlp_layers)
            args.logger.info('transferring classification head of seen classes...')

            # transfer seen classes' weights
            projector_cur.last_layer.weight_v.data[
            :args.num_seen_classes] = model_pre.projector.last_layer.weight_v.data[:args.num_seen_classes]
            projector_cur.last_layer.weight_g.data[
            :args.num_seen_classes] = model_pre.projector.last_layer.weight_g.data[:args.num_seen_classes]
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = model_pre.projector.last_layer.weight.data[
                                                                           :args.num_seen_classes]

            # initialize new class heads
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform
            online_session_train_loader_for_new_head_init = DataLoader(
                online_session_train_dataset_for_new_head_init,
                num_workers=args.num_workers_test,
                batch_size=256,
                shuffle=False,
                pin_memory=False
            )

            if args.init_new_head:
                new_head = get_kmeans_centroid_for_new_head(model_pre, online_session_train_loader_for_new_head_init,
                                                            args, device)

                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:],
                                                    dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:],
                                                  dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')

                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data

            model_cur = PromptEnhancedModel(
                backbone=backbone_cur,
                projector=projector_cur,
                prompt_pool=args.prompt_pool,
                top_k=5,
                enable_prompt_training=args.enable_prompt_training
            )

            args.logger.info(
                'incremental classifier heads from {} to {}'.format(len(model_pre.projector.last_layer.weight_v),
                                                                    len(model_cur.projector.last_layer.weight_v)))

            model_cur.to(device)
            ####################################################################################################################

            '''compute prototypes offline (session = 0)'''
            if session == 0:
                args.logger.info(
                    'Before Train: compute offline prototypes and radius from {} classes with the best model...'.format(
                        args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_session_train_loader_for_proto_aug = DataLoader(offline_session_train_dataset_for_proto_aug,
                                                                        num_workers=args.num_workers_test,
                                                                        batch_size=256, shuffle=False, pin_memory=False)
                proto_aug_manager.update_prototypes_offline(model_pre, offline_session_train_loader_for_proto_aug,
                                                            args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN
            # ----------------------
            train_online(model_cur, model_pre, proto_aug_manager, online_session_train_loader,
                         online_session_test_loader, session + 1, args)

            '''compute prototypes online after train (session > 0)'''
            args.logger.info(
                'After Train: update online prototypes from {} to {} classes with the best model...'.format(
                    args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes))
            load_dir_online_best = args.model_path[:-3] + '_session-' + str(session + 1) + f'_best.pt'
            args.logger.info('loading best checkpoints current online session: ' + load_dir_online_best)
            load_dict = torch.load(load_dir_online_best)
            model_cur.load_state_dict(load_dict['model'])
            proto_aug_manager.update_prototypes_online(model_cur, online_session_train_loader_for_new_head_init,
                                                       args.num_seen_classes,
                                                       args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session + 1) + f'.pt')
            args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
            proto_aug_manager.save_proto_aug_dict(save_path)

            '''save results dict after each session'''
            best_acc_list_dict = {
                'best_test_acc_all_list': args.best_test_acc_all_list,
                'best_test_acc_old_list': args.best_test_acc_old_list,
                'best_test_acc_new_list': args.best_test_acc_new_list,
                'best_test_acc_soft_all_list': args.best_test_acc_soft_all_list,
                'best_test_acc_seen_list': args.best_test_acc_seen_list,
                'best_test_acc_unseen_list': args.best_test_acc_unseen_list,
            }
            save_results_path = os.path.join(args.model_dir, 'best_acc_list' + '_session-' + str(session + 1) + f'.pt')
            args.logger.info('Saving results (best acc list) to {}.'.format(save_results_path))
            torch.save(best_acc_list_dict, save_results_path)

        # print final results
        args.logger.info(
            '\n\n==================== print final results over {} continual sessions ===================='.format(
                args.continual_session_num))
        for session in range(args.continual_session_num):
            args.logger.info(
                f'Session-{session + 1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')
        for session in range(args.continual_session_num):
            print(
                f'Session-{session + 1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')

    else:
        raise NotImplementedError


