import argparse
import os
import math
from tqdm import tqdm
from copy import deepcopy


import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler

from project_utils.general_utils import set_seed, init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets

from models.utils_simgcd import DINOHead, get_params_groups, SupConLoss, info_nce_logits, DistillLoss
from models.utils_simgcd_pro import get_kmeans_centroid_for_new_head
from models.utils_proto_aug import ProtoAugManager
from models import vision_transformer as vits
from config import dino_pretrain_path, exp_root
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


'''offline train and test'''
'''====================================================================================================================='''
def train_offline(student, train_loader, test_loader, args):

    params_groups = get_params_groups(student)
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
    # best acc log
    best_test_acc_old = 0

    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs = batch   # NOTE!!! no mask lab in this setting
            mask_lab = torch.ones_like(class_labels)   # NOTE!!! all samples are labeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clusterin，g, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup: SimGCD Eq.(4)
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup: SimGCD Eq.(1)
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup: SimGCD Eq.(2)
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)

            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            # Total loss
            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # logs
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
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


def test_offline(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc
'''====================================================================================================================='''




'''online train and test'''
'''====================================================================================================================='''
def train_online(student, student_pre, proto_aug_manager, train_loader, test_loader, current_session, args):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_online_per_session,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_online_per_session,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # best acc log
    best_test_acc_all = 0
    best_test_acc_old = 0
    best_test_acc_new = 0

    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()

        student.train()
        student_pre.eval()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs, _ = batch   # NOTE!!!   mask lab in this setting
            mask_lab = torch.zeros_like(class_labels)   # NOTE!!! all samples are unlabeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            '''
            求每个类的平均预测概率
            例子：
            Softmax probabilities:
                tensor([[9.9995e-01, 4.5398e-05, 2.0611e-09, 2.2603e-07],
                        [2.0611e-09, 9.9999e-01, 9.1188e-07, 1.6702e-10],
                        [2.2603e-07, 2.0611e-09, 9.9995e-01, 4.5398e-05]])
            Average probabilities across samples:
                tensor([0.3333, 0.3333, 0.3333, 0.0000])
            第四个类的平均预测概率是0.0000，表示模型对这个类的预测概率很低。
            '''
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)

            '''
            通过最小化负熵（即最大化熵），鼓励模型在旧类别和新类别之间分配更均匀的注意力，从而减少类别间的预测偏差。
            '''

            # 1. inter old and new
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            # torch.sum(avg_probs_old_in) 将所有旧类别的预测概率加总，得到模型对旧类别的整体关注程度。avg_probs_new_marginal同理。
            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new =  avg_probs_old_marginal * torch.log(avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. old (intra) & new (intra)
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)   # norm
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)   # norm
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm**(-avg_probs_old_in_norm))) + math.log(float(len(avg_probs_old_in_norm)))
            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(torch.log(avg_probs_new_in_norm**(-avg_probs_new_in_norm))) + math.log(float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=device)
            # overall me-max loss
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in


            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss(student)
            feats = student[0](images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            with torch.no_grad():
                feats_pre = student_pre[0](images)
                feats_pre = torch.nn.functional.normalize(feats_pre, dim=-1)

            # 这行代码计算的是特征蒸馏损失，用于让学生网络的特征模仿教师网络的输出，从而在增量学习中帮助保留之前学到的知识，减少灾难性遗忘。
            feat_distill_loss = (feats-feats_pre).pow(2).sum() / len(feats)

            # Total loss
            loss = 0
            loss += 1 * cluster_loss
            loss += 1 * contrastive_loss
            loss += args.proto_aug_weight * proto_aug_loss
            loss += args.feat_distill_weight * feat_distill_loss

            # logs
            pstr = ''
            pstr += f'me_max_loss_old_new: {me_max_loss_old_new.item():.4f} '
            pstr += f'me_max_loss_old_in: {me_max_loss_old_in.item():.4f} '
            pstr += f'me_max_loss_new_in: {me_max_loss_new_in.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'proto_aug_loss: {proto_aug_loss.item():.4f} '
            pstr += f'feat_distill_loss: {feat_distill_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                new_true_ratio = len(class_labels[class_labels>=args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds>=args.num_seen_classes]) / len(preds)
                args.logger.info(f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | Pred new ratio: {new_pred_ratio:.4f} | Ground-truth new ratio: {new_true_ratio:.4f}')

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, \
            all_acc_soft_test, seen_acc_test, unseen_acc_test = test_online(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        args.logger.info('Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test, unseen_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        if all_acc_test > best_test_acc_all:

            args.logger.info(f'Best ACC on All Classes on test set of session-{current_session}: {all_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt')   # NOTE!!! session
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + '_session-' + str(current_session) + f'_best.pt'))

            best_test_acc_all = all_acc_test
            best_test_acc_old = old_acc_test
            best_test_acc_new = new_acc_test

            best_test_acc_soft_all = all_acc_soft_test
            best_test_acc_seen = seen_acc_test
            best_test_acc_unseen = unseen_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Hard): {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
        args.logger.info(f'Metrics with best model on test set (Hard) of session-{current_session}: All (Soft): {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        args.logger.info('\n')


    # log best test acc list
    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)

# 手动计算匈牙利匹配
'''
def test_online(model, test_loader, epoch, save_name, args):
    # ============================ 打印测试集信息 ============================

    def print_testset_info(test_loader, class_names=None, prefix=""):
        all_labels = []
        for _, labels, _ in test_loader:
            all_labels.extend(labels.cpu().numpy())

        counter = Counter(all_labels)
        print(f"\n{prefix}测试集中类别分布:")
        for cls_id in sorted(counter.keys()):
            cls_name = class_names[cls_id] if class_names and cls_id in class_names else str(cls_id)
            print(f"  类别 {cls_id} ({cls_name}): {counter[cls_id]} 个样本")
        print(f"  总测试样本: {len(all_labels)}")

    print_testset_info(test_loader, class_names=getattr(args, 'class_names', None), prefix="Online")
    # ============================ 打印测试集信息 完毕============================

    model.eval()

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())

            # args.train_classes 原始训练时定义的类别
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                                       else False for x in label]))
            # args.num_seen_classes 当前已见类别范围
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                                       else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                             T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                             args=args)

    # # --------- 改进的手动计算方法 ---------
    # from scipy.optimize import linear_sum_assignment
    #
    # # 将数据分为已见和未见类别
    # seen_mask = mask_soft.astype(bool)
    # unseen_mask = ~seen_mask
    #
    # seen_preds = preds[seen_mask]
    # seen_targets = targets[seen_mask]
    # unseen_preds = preds[unseen_mask]
    # unseen_targets = targets[unseen_mask]
    #
    # # 为已见类别构建混淆矩阵和映射
    # seen_classes_pred = np.unique(seen_preds)
    # seen_classes_target = np.unique(seen_targets)
    # seen_classes = np.union1d(seen_classes_pred, seen_classes_target)
    #
    # seen_class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    # seen_w = np.zeros((len(seen_classes), len(seen_classes)), dtype=int)
    #
    # for i in range(len(seen_preds)):
    #     pred_idx = seen_class_to_idx[seen_preds[i]]
    #     target_idx = seen_class_to_idx[seen_targets[i]]
    #     seen_w[pred_idx, target_idx] += 1
    #
    # seen_row_ind, seen_col_ind = linear_sum_assignment(seen_w.max() - seen_w)
    # seen_map = {seen_classes[i]: seen_classes[j] for i, j in zip(seen_row_ind, seen_col_ind)}
    #
    # # 如果有未见类别，为它们构建单独的映射
    # unseen_map = {}
    # if len(unseen_preds) > 0:
    #     unseen_classes_pred = np.unique(unseen_preds)
    #     unseen_classes_target = np.unique(unseen_targets)
    #
    #     # 为未见类别创建频率矩阵
    #     unseen_w = np.zeros((len(unseen_classes_pred), len(unseen_classes_target)), dtype=int)
    #     unseen_pred_to_idx = {c: i for i, c in enumerate(unseen_classes_pred)}
    #     unseen_target_to_idx = {c: i for i, c in enumerate(unseen_classes_target)}
    #
    #     for i in range(len(unseen_preds)):
    #         pred_idx = unseen_pred_to_idx[unseen_preds[i]]
    #         target_idx = unseen_target_to_idx[unseen_targets[i]]
    #         unseen_w[pred_idx, target_idx] += 1
    #
    #     # 应用匈牙利算法找到最佳映射
    #     if unseen_w.size > 0:
    #         unseen_row_ind, unseen_col_ind = linear_sum_assignment(unseen_w.max() - unseen_w)
    #         for i, j in zip(unseen_row_ind, unseen_col_ind):
    #             pred_class = unseen_classes_pred[i]
    #             target_class = unseen_classes_target[j]
    #             # 只有当映射有足够支持时才添加
    #             if unseen_w[i, j] > 0:
    #                 unseen_map[pred_class] = target_class
    #
    # # 合并映射
    # class_map = {**seen_map, **unseen_map}
    #
    # # 应用映射到整个预测集
    # remapped_preds = np.array([class_map.get(p, p) for p in preds])
    #
    # # 计算准确率
    # all_correct = np.sum(remapped_preds == targets)
    # all_total = len(targets)
    # all_Acc = all_correct / all_total
    #
    # # 计算旧类(mask_hard)准确率
    # old_mask = mask_hard.astype(bool)
    # old_preds = remapped_preds[old_mask]
    # old_targets = targets[old_mask]
    # old_correct = np.sum(old_preds == old_targets)
    # old_total = len(old_targets)
    # old_Acc = old_correct / max(old_total, 1)
    #
    # # 计算新类准确率
    # new_mask = ~old_mask
    # new_preds = remapped_preds[new_mask]
    # new_targets = targets[new_mask]
    # new_correct = np.sum(new_preds == new_targets)
    # new_total = len(new_targets)
    # new_Acc = new_correct / max(new_total, 1)
    #
    # # 计算已见类准确率
    # seen_preds_remapped = remapped_preds[seen_mask]
    # seen_correct = np.sum(seen_preds_remapped == seen_targets)
    # seen_total = len(seen_targets)
    # seen_Acc = seen_correct / max(seen_total, 1)
    #
    # # 计算未见类准确率
    # unseen_preds_remapped = remapped_preds[unseen_mask]
    # unseen_correct = np.sum(unseen_preds_remapped == unseen_targets)
    # unseen_total = len(unseen_targets)
    # unseen_Acc = unseen_correct / max(unseen_total, 1)
    #
    # # 记录和输出结果
    # args.logger.info(f"我的计算结果:")
    # args.logger.info(f"All: {all_Acc:.4f}")
    # args.logger.info(f"Old: {old_Acc:.4f}")
    # args.logger.info(f"New: {new_Acc:.4f}")
    # args.logger.info(f"Seen: {seen_Acc:.4f}")
    # args.logger.info(f"Unseen: {unseen_Acc:.4f}")
    #
    # # 与官方结果比较
    # args.logger.info(f"官方计算结果:")
    # args.logger.info(f"All: {all_acc:.4f}")
    # args.logger.info(f"Old: {old_acc:.4f}")
    # args.logger.info(f"New: {new_acc:.4f}")
    # args.logger.info(f"Seen: {seen_acc:.4f}")
    # args.logger.info(f"Unseen: {unseen_acc:.4f}")

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc
'''

# 在线测试函数，支持显示测试图片和标签，并特别展示识别错误的前5个样本
'''
def test_online(model, test_loader, epoch, save_name, args):
    """
    在线测试函数，支持显示测试图片和标签，并特别展示识别错误的前5个样本

    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        epoch: 当前训练轮次
        save_name: 保存名称
        args: 参数对象，需包含logger

    Returns:
        各类评估指标
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import Counter
    import torch
    from tqdm import tqdm

    # 在函数内部直接定义class_names列表
    class_names = [
        'baseball',  # 0
        'bus',  # 1
        'camera',  # 2
        'cosplay',  # 3
        'dress',  # 4
        'hockey',  # 5
        'laptop',  # 6
        'racing',  # 7
        'soccer',  # 8
        'sweater'  # 9
    ]

    # 将class_names添加到args中以便其他地方使用
    args.class_names = class_names

    # ============================ 打印测试集信息 ============================
    def print_testset_info(test_loader, class_names, prefix=""):
        all_labels = []
        for batch in test_loader:
            if len(batch) >= 2:  # 兼容不同的数据加载器格式
                labels = batch[1]
                all_labels.extend(labels.cpu().numpy())

        counter = Counter(all_labels)
        args.logger.info(f"\n{prefix}测试集中类别分布:")
        for cls_id in sorted(counter.keys()):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            args.logger.info(f"  类别 {cls_id} ({cls_name}): {counter[cls_id]} 个样本")
        args.logger.info(f"  总测试样本: {len(all_labels)}")
        return counter

    class_counter = print_testset_info(test_loader, class_names, prefix="Online")

    # ============================ 查看前5张测试图片 ==========================
    args.logger.info("\n显示测试集前5张图片:")

    # 创建一个整合所有图片的大图
    plt.figure(figsize=(15, 5))

    # 获取前5张图片和标签
    images_to_show = []
    labels_to_show = []
    count = 0

    # 从测试集中提取前5张图片
    for batch in test_loader:
        if len(batch) >= 2:  # 兼容不同的数据加载器格式
            images = batch[0]
            labels = batch[1]

            batch_size = min(5 - count, len(images))
            for i in range(batch_size):
                count += 1
                img = images[i].cpu()
                label = labels[i].item()

                # 保存图片和标签
                images_to_show.append(img)
                labels_to_show.append(label)

                # 显示图片信息
                cls_name = class_names[label] if label < len(class_names) else f"未知类别({label})"
                args.logger.info(f"图片 {count}: 标签ID = {label}, 类别名称 = {cls_name}")

                # 在大图中添加这张图片
                plt.subplot(1, 5, count)

                # 处理图像以便显示
                if hasattr(img, 'permute'):  # 处理PyTorch张量
                    # 将CHW转为HWC格式并转换为numpy数组
                    img_np = img.permute(1, 2, 0).numpy()

                    # 标准化图像到[0,1]范围
                    if img_np.max() > 1.0 or img_np.min() < 0.0:
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
                else:
                    img_np = img

                plt.imshow(img_np)
                plt.title(f"{cls_name} (ID:{label})")
                plt.axis('off')

            if count >= 5:
                break

    # 保存合并后的图片
    if hasattr(args, 'log_dir'):
        sample_img_path = os.path.join(args.log_dir, f'test_samples_epoch_{epoch}.png')
        plt.tight_layout()
        plt.savefig(sample_img_path)
        plt.close()
        args.logger.info(f"测试样本图片已保存到: {sample_img_path}")

    # ============================ 模型评估 ============================
    model.eval()

    # 用于收集错误预测的样本
    wrong_samples = []

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])

    # 记录前5个预测结果
    pred_first_5 = []
    true_first_5 = []
    count = 0

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估测试集")):
        # 兼容不同的数据加载器格式
        if len(batch) == 2:
            images, label = batch
        elif len(batch) >= 3:
            images, label, _ = batch[:3]

        # 将图像移至GPU
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            # 获取模型预测
            try:
                _, logits = model(images)
            except:
                logits = model(images)

            batch_preds = logits.argmax(1).cpu()

            # 找出预测错误的样本
            for i in range(len(batch_preds)):
                if batch_preds[i].item() != label[i].item():
                    # 存储错误样本信息
                    wrong_samples.append({
                        'image': images[i].cpu(),
                        'pred': batch_preds[i].item(),
                        'true': label[i].item()
                    })

            preds.append(batch_preds.numpy())
            targets.append(label.cpu().numpy())

            # 记录前5个样本的预测结果
            if count < 5:
                items_to_add = min(5 - count, len(batch_preds))
                pred_first_5.extend(batch_preds[:items_to_add].tolist())
                true_first_5.extend(label.cpu().numpy()[:items_to_add].tolist())
                count += items_to_add

            # args.train_classes 原始训练时定义的类别
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                                       else False for x in label]))
            # args.num_seen_classes 当前已见类别范围
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                                       else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 输出前5个样本的预测结果
    args.logger.info("\n前5个测试样本的预测结果:")
    for i, (pred, true) in enumerate(zip(pred_first_5, true_first_5)):
        pred_class = class_names[pred] if pred < len(class_names) else f"未知({pred})"
        true_class = class_names[true] if true < len(class_names) else f"未知({true})"
        correct = "✓" if pred == true else "✗"
        args.logger.info(f"样本 {i + 1}: 预测 = {pred} ({pred_class}), 真实 = {true} ({true_class}) {correct}")

    # ============================== 显示错误预测的前5个样本 ==============================
    args.logger.info("\n显示错误预测的前5个样本:")

    # 只取前5个错误样本
    wrong_samples = wrong_samples[:5]

    if len(wrong_samples) > 0:
        # 创建一个展示错误样本的大图
        plt.figure(figsize=(15, 5))

        for i, sample in enumerate(wrong_samples):
            img = sample['image']
            pred_id = sample['pred']
            true_id = sample['true']

            pred_class = class_names[pred_id] if pred_id < len(class_names) else f"未知({pred_id})"
            true_class = class_names[true_id] if true_id < len(class_names) else f"未知({true_id})"

            # 在日志中显示错误样本信息
            args.logger.info(f"错误样本 {i + 1}: 预测 = {pred_id} ({pred_class}), 真实 = {true_id} ({true_class})")

            # 在大图中添加这张图片
            plt.subplot(1, len(wrong_samples), i + 1)

            # 处理图像以便显示
            if hasattr(img, 'permute'):  # 处理PyTorch张量
                # 将CHW转为HWC格式并转换为numpy数组
                img_np = img.permute(1, 2, 0).numpy()

                # 标准化图像到[0,1]范围
                if img_np.max() > 1.0 or img_np.min() < 0.0:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
            else:
                img_np = img

            plt.imshow(img_np)
            plt.title(f"pre:{pred_class}\n cor:{true_class}", color='red')
            plt.axis('off')

        # 保存错误样本图片
        if hasattr(args, 'log_dir'):
            wrong_img_path = os.path.join(args.log_dir, f'wrong_samples_epoch_{epoch}.png')
            plt.tight_layout()
            plt.savefig(wrong_img_path)
            plt.close()
            args.logger.info(f"错误样本图片已保存到: {wrong_img_path}")
    else:
        args.logger.info("没有找到错误预测的样本!")

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                             T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                             args=args)

    # 打印总结信息
    args.logger.info(f"\n测试结果总结 (Epoch {epoch}):")
    args.logger.info(f"Hard分类指标: 全部={all_acc:.4f}, 旧类={old_acc:.4f}, 新类={new_acc:.4f}")
    args.logger.info(f"Soft分类指标: 全部={all_acc_soft:.4f}, 已见={seen_acc:.4f}, 未见={unseen_acc:.4f}")
    args.logger.info(f"错误率: {(1 - all_acc) * 100:.2f}% (找到{len(wrong_samples)}个错误样本展示)")

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc
'''

DATASET_CLASS_NAMES = {
    'clear10': [
        'baseball', 'bus', 'camera', 'cosplay', 'dress', 'hockey', 'laptop', 'racing', 'soccer', 'sweater'    # 9
    ],
    'cifar10': [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ],
    'cifar100': [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'cameleon', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
        'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
}


def test_online(model, test_loader, epoch, save_name, args):
    """
    在线测试函数，在session结束时显示测试图片和识别错误的样本
    支持多种数据集的类别名称

    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        epoch: 当前训练轮次
        save_name: 保存名称
        args: 参数对象，需包含logger

    Returns:
        各类评估指标
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import Counter
    import torch
    from tqdm import tqdm

    # 获取当前数据集的类别名称
    if hasattr(args, 'dataset_name') and args.dataset_name in DATASET_CLASS_NAMES:
        class_names = DATASET_CLASS_NAMES[args.dataset_name]
    elif hasattr(args, 'class_names'):
        # 如果args中已经有class_names，则使用它
        class_names = args.class_names
    else:
        # 默认使用数字作为类别名称
        class_names = [f"类别{i}" for i in range(100)]
        args.logger.info("未找到数据集类别名称，使用默认编号")

    # 将class_names添加到args中以便其他地方使用
    args.class_names = class_names

    # 判断是否是session结束以及当前session
    is_session_end = False
    current_session = None

    # 修复session计算逻辑
    # 首先检查是否传入了当前session信息
    if hasattr(args, 'current_session'):
        # 如果直接传入了当前session编号，则使用它
        current_session = args.current_session
    else:
        # 尝试从其他参数推断当前session
        if hasattr(args, 'num_seen_classes') and hasattr(args, 'num_labeled_classes') and hasattr(args,
                                                                                                  'num_novel_class_per_session'):
            # 基于已见类别数量计算当前session
            if args.num_seen_classes > args.num_labeled_classes:
                # (已见类别 - 初始类别) / 每个session的新类别数 = 已完成的session数
                completed_sessions = (
                                                 args.num_seen_classes - args.num_labeled_classes) // args.num_novel_class_per_session
                current_session = completed_sessions + 1  # 当前正在进行的session
            else:
                current_session = 1  # 第一个session

    # 如果仍然无法确定session，使用默认方法
    if current_session is None:
        # 默认从1开始
        current_session = 1
        args.logger.info("无法确定当前session编号，使用默认值1")

    # 判断是否为session结束
    if hasattr(args, 'epochs_online_per_session'):
        # 如果当前epoch是session的最后一个epoch
        if (epoch + 1) % args.epochs_online_per_session == 0 or epoch == args.epochs_online_per_session - 1:
            is_session_end = True
            args.logger.info(f"当前为Session {current_session}的最后一个epoch")

    # ============================ 打印测试集信息 ============================
    def print_testset_info(test_loader, class_names, prefix=""):
        all_labels = []
        for batch in test_loader:
            if len(batch) >= 2:  # 兼容不同的数据加载器格式
                labels = batch[1]
                all_labels.extend(labels.cpu().numpy())

        counter = Counter(all_labels)
        args.logger.info(f"\n{prefix}测试集中类别分布:")
        for cls_id in sorted(counter.keys()):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            args.logger.info(f"  类别 {cls_id} ({cls_name}): {counter[cls_id]} 个样本")
        args.logger.info(f"  总测试样本: {len(all_labels)}")
        return counter

    class_counter = print_testset_info(test_loader, class_names, prefix="Online")

    # ============================ 模型评估 ============================
    model.eval()

    # 用于收集所有样本和错误预测的样本
    all_samples = []
    wrong_samples = []

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估测试集")):
        # 兼容不同的数据加载器格式
        if len(batch) == 2:
            images, label = batch
        elif len(batch) >= 3:
            images, label, _ = batch[:3]

        # 将图像移至GPU
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            # 获取模型预测
            try:
                _, logits = model(images)
            except:
                logits = model(images)

            batch_preds = logits.argmax(1).cpu()

            # 收集所有样本信息（如果是session结束）
            if is_session_end:
                for i in range(len(batch_preds)):
                    sample_info = {
                        'image': images[i].cpu(),
                        'pred': batch_preds[i].item(),
                        'true': label[i].item(),
                        'is_correct': batch_preds[i].item() == label[i].item()
                    }

                    # 添加到所有样本列表
                    all_samples.append(sample_info)

                    # 如果预测错误，添加到错误样本列表
                    if not sample_info['is_correct']:
                        wrong_samples.append(sample_info)

            preds.append(batch_preds.numpy())
            targets.append(label.cpu().numpy())

            # args.train_classes 原始训练时定义的类别
            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                                       else False for x in label]))
            # args.num_seen_classes 当前已见类别范围
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                                       else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # ============================== 如果是session结束，显示样本和错误 ==============================
    if is_session_end and hasattr(args, 'log_dir'):
        # 记录当前session编号到文件名
        session_str = f"session_{current_session}"

        # 1. 显示前5个测试样本
        args.logger.info(f"\n显示{session_str}的前5个测试样本:")

        # 只取前5个样本
        samples_to_show = all_samples[:5]

        if len(samples_to_show) > 0:
            # 创建一个展示样本的大图
            plt.figure(figsize=(15, 5))

            for i, sample in enumerate(samples_to_show):
                img = sample['image']
                pred_id = sample['pred']
                true_id = sample['true']

                pred_class = class_names[pred_id] if pred_id < len(class_names) else f"未知({pred_id})"
                true_class = class_names[true_id] if true_id < len(class_names) else f"未知({true_id})"

                # 在日志中显示样本信息
                status = "✓" if sample['is_correct'] else "✗"
                args.logger.info(f"样本 {i + 1}: 预测={pred_id} ({pred_class}), 真实={true_id} ({true_class}) {status}")

                # 在大图中添加这张图片
                plt.subplot(1, len(samples_to_show), i + 1)

                # 处理图像以便显示
                if hasattr(img, 'permute'):  # 处理PyTorch张量
                    # 将CHW转为HWC格式并转换为numpy数组
                    img_np = img.permute(1, 2, 0).numpy()

                    # 标准化图像到[0,1]范围
                    if img_np.max() > 1.0 or img_np.min() < 0.0:
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
                else:
                    img_np = img

                plt.imshow(img_np)
                plt.title(f"{true_class}\n pre:{pred_class}",
                          color='green' if sample['is_correct'] else 'red')
                plt.axis('off')

            # 保存样本图片
            sample_img_path = os.path.join(args.log_dir, f'test_samples_{session_str}.png')
            plt.tight_layout()
            plt.savefig(sample_img_path)
            plt.close()
            args.logger.info(f"测试样本图片已保存到: {sample_img_path}")

        # 2. 显示所有错误预测的样本信息
        args.logger.info(f"\n显示{session_str}的所有错误预测样本 (总计 {len(wrong_samples)} 个):")

        # 创建一个表格记录所有错误样本
        wrong_info_text = "索引,预测类别ID,预测类别名,真实类别ID,真实类别名\n"
        for i, sample in enumerate(wrong_samples):
            pred_id = sample['pred']
            true_id = sample['true']

            pred_class = class_names[pred_id] if pred_id < len(class_names) else f"未知({pred_id})"
            true_class = class_names[true_id] if true_id < len(class_names) else f"未知({true_id})"

            wrong_info_text += f"{i + 1},{pred_id},{pred_class},{true_id},{true_class}\n"

            # 在日志中显示前20个错误样本
            if i < 20:
                args.logger.info(f"错误样本 {i + 1}: 预测={pred_id} ({pred_class}), 真实={true_id} ({true_class})")

        # 保存错误信息到文本文件
        wrong_info_path = os.path.join(args.log_dir, f'wrong_predictions_{session_str}.csv')
        with open(wrong_info_path, 'w') as f:
            f.write(wrong_info_text)
        args.logger.info(f"所有错误预测信息已保存到: {wrong_info_path}")

        # 3. 显示错误预测的样本图片
        if len(wrong_samples) > 0:
            # 最多显示10个错误样本
            wrong_to_show = wrong_samples[:min(10, len(wrong_samples))]

            # 计算行数和列数
            n_cols = min(5, len(wrong_to_show))
            n_rows = (len(wrong_to_show) + n_cols - 1) // n_cols

            # 创建一个展示错误样本的大图
            plt.figure(figsize=(3 * n_cols, 3 * n_rows))

            for i, sample in enumerate(wrong_to_show):
                img = sample['image']
                pred_id = sample['pred']
                true_id = sample['true']

                pred_class = class_names[pred_id] if pred_id < len(class_names) else f"未知({pred_id})"
                true_class = class_names[true_id] if true_id < len(class_names) else f"未知({true_id})"

                # 在大图中添加这张图片
                plt.subplot(n_rows, n_cols, i + 1)

                # 处理图像以便显示
                if hasattr(img, 'permute'):  # 处理PyTorch张量
                    # 将CHW转为HWC格式并转换为numpy数组
                    img_np = img.permute(1, 2, 0).numpy()

                    # 标准化图像到[0,1]范围
                    if img_np.max() > 1.0 or img_np.min() < 0.0:
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
                else:
                    img_np = img

                plt.imshow(img_np)
                plt.title(f"GT:{true_class}\n Pre:{pred_class}", color='red', fontsize=10)
                plt.axis('off')

            # 保存错误样本图片
            wrong_img_path = os.path.join(args.log_dir, f'wrong_samples_{session_str}.png')
            plt.tight_layout()
            plt.savefig(wrong_img_path)
            plt.close()
            args.logger.info(f"错误样本图片已保存到: {wrong_img_path}")

        # 4. 生成类别混淆矩阵
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            # 获取所有预测结果和真实标签
            y_true = [sample['true'] for sample in all_samples]
            y_pred = [sample['pred'] for sample in all_samples]

            # 获取所有出现的类别
            all_classes = sorted(list(set(y_true + y_pred)))

            # 生成混淆矩阵
            cm = confusion_matrix(y_true, y_pred, labels=all_classes)

            # 计算准确率
            class_acc = np.zeros(len(all_classes))
            for i, cls in enumerate(all_classes):
                cls_samples = [s for s in all_samples if s['true'] == cls]
                if cls_samples:
                    correct = sum(1 for s in cls_samples if s['is_correct'])
                    class_acc[i] = correct / len(cls_samples)

            # 显示混淆矩阵
            plt.figure(figsize=(12, 10))

            # 使用Seaborn绘制更美观的热力图
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            #             xticklabels=[f"{i}:{class_names[i]}" if i < len(class_names) else i for i in all_classes],
            #             yticklabels=[f"{i}:{class_names[i]}" if i < len(class_names) else i for i in all_classes])

            sns.heatmap(cm, fmt='d', cmap='Blues',
                        xticklabels=all_classes,
                        yticklabels=all_classes)

            plt.xlabel(' Prediction')
            plt.ylabel(' Ground Truth')
            plt.title(f'{session_str} Confusion Matrix')

            # 保存混淆矩阵
            conf_matrix_path = os.path.join(args.log_dir, f'confusion_matrix_{session_str}.png')
            plt.tight_layout()
            plt.savefig(conf_matrix_path)
            plt.close()
            args.logger.info(f"类别混淆矩阵已保存到: {conf_matrix_path}")

            # 显示各类别准确率
            args.logger.info("\n各类别准确率:")
            for i, cls in enumerate(all_classes):
                cls_name = class_names[cls] if cls < len(class_names) else f"未知({cls})"
                args.logger.info(f"  类别 {cls} ({cls_name}): {class_acc[i] * 100:.2f}%")
        except Exception as e:
            args.logger.info(f"无法生成混淆矩阵: {str(e)}")

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                             T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                             args=args)

    # 打印总结信息
    args.logger.info(f"\n测试结果总结 (Epoch {epoch}):")
    args.logger.info(f"Hard分类指标: 全部={all_acc:.4f}, 旧类={old_acc:.4f}, 新类={new_acc:.4f}")
    args.logger.info(f"Soft分类指标: 全部={all_acc_soft:.4f}, 已见={seen_acc:.4f}, 未见={unseen_acc:.4f}")

    # 如果是session结束，打印错误率信息
    if is_session_end:
        error_count = len(wrong_samples)
        total_count = len(all_samples)
        error_rate = error_count / total_count if total_count > 0 else 0
        args.logger.info(f"Session {current_session} 错误率: {error_rate * 100:.2f}% ({error_count}/{total_count})")

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc
'''====================================================================================================================='''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
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
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup) of the teacher temperature.')
    #parser.add_argument('--teacher_temp_final', default=0.05, type=float, help='Final value (online session) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

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


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    #set_seed(args.seed)
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes) # get_dataset

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'happy' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)

    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
            + '_' + 'ContinualNum' + str(args.continual_session_num) + '_' + 'UnseenNum' + str(args.online_novel_unseen_num) \
                + '_' + 'SeenNum' + str(args.online_novel_seen_num)

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()

    args.logger.info(f'Loading weights from {dino_pretrain_path}')
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes   # NOTE!!!

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector)

    model.to(device)

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
        offline_session_train_dataset, offline_session_test_dataset,\
            _online_session_train_dataset_list, _online_session_test_dataset_list,\
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
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
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
        args.logger.info('\n\n==================== online continual GCD with unlabeled data (old + novel) ====================')
        args.logger.info('loading dataset...')
        _offline_session_train_dataset, _offline_session_test_dataset,\
            online_session_train_dataset_list, online_session_test_dataset_list,\
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
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views*args.batch_size, args.hardness_temp, args.radius_scale, device, args.logger)

        # best test acc list across continual sessions
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []

        start_session = 0

        '''Continual GCD sessions'''
        #for session in range(args.continual_session_num):
        for session in range(start_session, args.continual_session_num): # cifar100: session 0->4
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session+1))
            # dataset for the current session
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]
            online_session_train_loader = DataLoader(online_session_train_dataset, num_workers=args.num_workers,
                                                     batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
            online_session_test_loader = DataLoader(online_session_test_dataset, num_workers=args.num_workers_test,
                                                    batch_size=256, shuffle=False, pin_memory=False)

            # number of seen (offline old + previous online new) classes till the beginning of this session
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session # example(cifar100)：session 1: old + 10*0 = 50 + 0 = 50
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(args.num_seen_classes))
            if args.dataset_name == 'cifar100':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets))
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                args.num_cur_novel_classes = len(np.unique(novel_cls_labels))
            elif args.dataset_name == 'scars':
                args.num_cur_novel_classes = len(np.unique(online_session_train_dataset.novel_unlabelled_dataset.target))   # NOTE!!! target
            else:
                args.num_cur_novel_classes = args.num_novel_class_per_session * (session+1)
            args.logger.info('number of all novel class (seen novel + unseen novel) in current session: {}'.format(args.num_cur_novel_classes))

            # ---------- 在这里添加新代码 ----------
            # 打印类别详细信息
            args.logger.info("\n详细类别信息:")

            # 1. 打印已知类(Known classes)信息
            known_classes = list(range(args.num_labeled_classes))  # 已知类的索引
            known_class_names = [str(cls) for cls in known_classes]  # 如果没有类名就用索引
            args.logger.info(f"已知类({len(known_classes)}个): {known_class_names}")

            # 2. 打印已见类(Seen classes)信息 - 仅当不是第一个session时
            if session > 0:
                seen_novel_classes = list(range(args.num_labeled_classes, args.num_seen_classes))
                seen_novel_class_names = [str(cls) for cls in seen_novel_classes]
                args.logger.info(f"已见新类({len(seen_novel_classes)}个): {seen_novel_class_names}")

            # 3. 打印当前session的新类信息
            current_novel_classes = []

            # 根据不同数据集获取新类
            if args.dataset_name == 'cifar100':
                current_novel_classes = np.unique(online_session_train_dataset.novel_unlabelled_dataset.targets)
            elif args.dataset_name == 'tiny_imagenet':
                novel_cls_labels = [t for i, (p, t) in
                                    enumerate(online_session_train_dataset.novel_unlabelled_dataset.data)]
                current_novel_classes = np.unique(novel_cls_labels)
            elif args.dataset_name == 'aircraft':
                novel_cls_labels = [t for i, (p, t) in
                                    enumerate(online_session_train_dataset.novel_unlabelled_dataset.samples)]
                current_novel_classes = np.unique(novel_cls_labels)
            elif args.dataset_name == 'scars':
                current_novel_classes = np.unique(online_session_train_dataset.novel_unlabelled_dataset.target)
            else:
                # 如果没有特定处理，使用索引范围
                start_idx = args.num_seen_classes
                end_idx = args.num_labeled_classes + args.num_cur_novel_classes
                current_novel_classes = list(range(start_idx, end_idx))

            # 转换为名称并打印
            current_novel_class_names = [str(cls) for cls in current_novel_classes]
            args.logger.info(f"当前新类({len(current_novel_classes)}个): {current_novel_class_names}")
            # ---------- 新增代码结束 ----------


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
            '''确保每个新阶段都能从之前阶段学到的知识开始，而不是从头学习。'''

            ####################################################################################################################
            args.logger.info('loading checkpoints of model_pre...')
            if session == 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                if args.load_offline_id is not None:
                    # load_dir_online = os.path.join(exp_root + '_' + 'offline', args.dataset_name, args.load_offline_id, 'checkpoints', 'model_best.pt') # session 0 加载的是离线训练的模型
                    load_dir_online = '/home/ps/_jinwei/Happy-CGCD/Official_checkpoints/C100_Stage0/model_best.pt'
                    args.logger.info('loading offline checkpoints from: ' + load_dir_online)
                    load_dict = torch.load(load_dir_online)
                    model_pre.load_state_dict(load_dict['model'])
                    args.logger.info('successfully loaded checkpoints!')
            else:        # session > 0
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_seen_classes, nlayers=args.num_mlp_layers)
                model_pre = nn.Sequential(backbone, projector_pre)
                load_dir_online = args.model_path[:-3] + '_session-' + str(session) + f'_best.pt'   # NOTE!!! session, best; 当 session=1 时，加载的是 session-1_best.pt（即session=0训练完成后保存的模型）
                args.logger.info('loading checkpoints from last online session: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                model_pre.load_state_dict(load_dict['model'])
                args.logger.info('successfully loaded checkpoints!')
            ####################################################################################################################

            '''incremental parametric classifier in SimGCD'''
            ####################################################################################################################
            ####################################################################################################################
            backbone_cur = deepcopy(backbone)   # NOTE!!!
            backbone_cur.load_state_dict(model_pre[0].state_dict())   # NOTE!!!
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes   # total num of classes in the current session 拓展分类器，添加输出结点
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))
            projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim_cur, nlayers=args.num_mlp_layers)
            args.logger.info('transferring classification head of seen classes...')

            # ==============================================================
            # 模型在每个新会话中都以上一个会话的知识为基础，同时扩展自身能力来适应新类别；
            # 权重迁移，模型从一个session进入下一个session时，需要扩展分类器以容纳新类别，同时保留对已见类别的知识；直接转移上一个模型中已知类的分类器权重，保证已知类的知识不会丢失
            # transfer seen classes' weights
            # ==============================================================
            projector_cur.last_layer.weight_v.data[:args.num_seen_classes] = projector_pre.last_layer.weight_v.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight_g.data[:args.num_seen_classes] = projector_pre.last_layer.weight_g.data[:args.num_seen_classes]   # NOTE!!!
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = projector_pre.last_layer.weight.data[:args.num_seen_classes]   # NOTE!!!
            # initialize new class heads
            #############################################
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            """
            online_session_train_dataset
            ├── old_unlabelled_dataset  // 包含旧类别的无标签数据
            └── novel_unlabelled_dataset  // 包含新类别的无标签数据（包括已见类）
            """

            # 使用测试变换而非训练变换的原因：初始化分类器头需要稳定、干净的特征表示，不需要训练时的数据增强

            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform   # NOTE!!!
            online_session_train_loader_for_new_head_init = DataLoader(online_session_train_dataset_for_new_head_init, num_workers=args.num_workers_test,
                                                                    batch_size=256, shuffle=False, pin_memory=False)
            if args.init_new_head:
                new_head = get_kmeans_centroid_for_new_head(model_pre, online_session_train_loader_for_new_head_init, args, device)   # torch.Size([10, 768])

                """
                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] 这部分就是指向分类器中为新类别预留的权重部分。
                在执行K-means初始化之前，这些权重是通过标准的随机初始化方法（如PyTorch默认的初始化）创建的。
                这行代码实际上是在:
                1. 对每个新类别位置的随机初始化权重向量计算范数（向量长度）
                2. 对所有这些范数求平均值
                然后使用这个平均范数来缩放通过K-means获得的新类别表示，使得替换后的权重在数值规模上与随机初始化的权重相似。
                这是一种确保新权重与网络其他部分在数值上兼容的技巧，有助于保持训练过程的稳定性。
                """

                # 保持范数一致性：通过计算这些初始随机权重的平均范数，然后用同样的范数来缩放K-means得到的新类别质心，确保新初始化的权重与分类器其他部分具有相似的规模。
                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:], dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:], dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')

                # 只更新新类别的部分，保留已知类别的权重不变
                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data   # NOTE!!!   # copy
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data   # NOTE!!!
                # 结合[:args.num_seen_classes]，projector_cur全部更新完
                """
                [:args.num_seen_classes]：已知类别的权重，这部分通过从 projector_pre 中直接复制完成更新。
                [args.num_seen_classes:]：新类别的权重，这部分通过 K-means 初始化完成更新。
                """
            ##############################################

            model_cur = nn.Sequential(backbone_cur, projector_cur)   # NOTE!!! backbone_cur

            args.logger.info('incremental classifier heads from {} to {}'.format(len(model_pre[1].last_layer.weight_v), len(model_cur[1].last_layer.weight_v)))
            model_cur.to(device)
            ####################################################################################################################
            ####################################################################################################################

            '''compute prototypes offline (session = 0)'''
            if session == 0:
                args.logger.info('Before Train: compute offline prototypes and radius from {} classes with the best model...'.format(args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_session_train_loader_for_proto_aug = DataLoader(offline_session_train_dataset_for_proto_aug, num_workers=args.num_workers_test,
                                                                        batch_size=256, shuffle=False, pin_memory=False)
                # NOTE!!! use model_pre && offline_session_train_loader
                proto_aug_manager.update_prototypes_offline(model_pre, offline_session_train_loader_for_proto_aug, args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN
            # ----------------------
            train_online(model_cur, model_pre, proto_aug_manager, online_session_train_loader, online_session_test_loader, session+1, args)

            '''compute prototypes online after train (session > 0)'''
            #############################################################################################################
            args.logger.info('After Train: update online prototypes from {} to {} classes with the best model...'.format(args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes))
            # NOTE!!! use model_cur_best && online_session_train_loader
            load_dir_online_best = args.model_path[:-3] + '_session-' + str(session+1) + f'_best.pt'   # NOTE!!! session, best
            args.logger.info('loading best checkpoints current online session: ' + load_dir_online_best)
            load_dict = torch.load(load_dir_online_best)
            model_cur.load_state_dict(load_dict['model'])
            proto_aug_manager.update_prototypes_online(model_cur, online_session_train_loader_for_new_head_init, 
                                                       args.num_seen_classes, args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session+1) + f'.pt')
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
            save_results_path = os.path.join(args.model_dir, 'best_acc_list' + '_session-' + str(session+1) + f'.pt')
            args.logger.info('Saving results (best acc list) to {}.'.format(save_results_path))
            torch.save(best_acc_list_dict, save_results_path)

        # print final results
        args.logger.info('\n\n==================== print final results over {} continual sessions ===================='.format(args.continual_session_num))
        for session in range(args.continual_session_num):
            args.logger.info(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')
        for session in range(args.continual_session_num):
            print(f'Session-{session+1}: All (Hard): {args.best_test_acc_all_list[session]:.4f} Old: {args.best_test_acc_old_list[session]:.4f} New: {args.best_test_acc_new_list[session]:.4f} | All (Soft): {args.best_test_acc_soft_all_list[session]:.4f} Seen: {args.best_test_acc_seen_list[session]:.4f} Unseen: {args.best_test_acc_unseen_list[session]:.4f}')

    else:
        raise NotImplementedError
