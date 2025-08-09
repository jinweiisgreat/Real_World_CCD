import argparse
import os
import math
from tqdm import tqdm
from copy import deepcopy
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F

from project_utils.general_utils import set_seed, init_experiment, AverageMeter
from project_utils.cluster_and_log_utils import log_accs_from_preds

from data.augmentations import get_transform
from data.get_datasets import get_class_splits, ContrastiveLearningViewGenerator, get_datasets

from models.utils_simgcd import DINOHead, get_params_groups, SupConLoss, info_nce_logits, DistillLoss
from models.utils_simgcd_pro import get_kmeans_centroid_for_new_head
from models.utils_proto_aug import ProtoAugManager

from config import exp_root
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import setproctitle

setproctitle.setproctitle("clip continual gcd")

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class WeightedGamma(nn.Module):
    """用于融合图像和文本特征的加权模块"""

    def __init__(self, args):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.5))  # 可学习参数

    def forward(self, img_feats, txt_feats):
        # 融合图像和文本特征
        fusion_feat = self.gamma * img_feats + (1 - self.gamma) * txt_feats
        return F.normalize(fusion_feat, dim=-1)


def get_clip_features_and_fusion(clip_model, weighted_gamma, images):
    """获取CLIP特征并进行融合"""
    with torch.no_grad():
        all_img_feats, all_txt_feats = clip_model(images)

    # 融合图像和文本特征
    fusion_feat = weighted_gamma(all_img_feats, all_txt_feats)
    return fusion_feat


'''offline train and test with CLIP'''
'''====================================================================================================================='''


def train_offline_clip(clip_model, projector, weighted_gamma, train_loader, test_loader, args):
    """离线训练阶段 - 使用CLIP特征，与clip_simgcd保持一致"""

    # 设置优化器 - 包含projector和weighted_gamma的参数
    optimizer = SGD(list(projector.parameters()) + list(weighted_gamma.parameters()),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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

        clip_model.eval()  # CLIP保持eval模式
        projector.train()
        weighted_gamma.train()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs = batch
            mask_lab = torch.ones_like(class_labels)  # 所有样本都是标注的

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

            # 处理两个视图
            fusion_feat_view1 = get_clip_features_and_fusion(clip_model, weighted_gamma, images[0].cuda())
            fusion_feat_view2 = get_clip_features_and_fusion(clip_model, weighted_gamma, images[1].cuda())
            fusion_feat = torch.cat([fusion_feat_view1, fusion_feat_view2], dim=0)

            # 通过投影头
            student_proj, student_out = projector(fusion_feat.float())
            teacher_out = student_out.detach()

            # 超过warm_up_epoch后开始计算损失
            if epoch >= args.warm_up_epoch:
                # 监督分类损失
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # 聚类损失
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch - args.warm_up_epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # 总的分类损失
                loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            else:
                cls_loss = torch.tensor(0.0).cuda()
                cluster_loss = torch.tensor(0.0).cuda()
                loss = torch.tensor(0.0).cuda()

            # 对比学习损失
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # 监督对比学习损失
            student_proj_sup = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj_sup = torch.nn.functional.normalize(student_proj_sup, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj_sup, labels=sup_con_labels)

            # 添加对比学习损失
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # 日志
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'gamma: {weighted_gamma.gamma.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} lr: {:.4f} gamma: {:.4f}'.format(
            epoch, loss_record.avg, exp_lr_scheduler.get_lr()[0], weighted_gamma.gamma.item()))

        # 超过warm_up_epoch后开始测试
        if epoch >= args.warm_up_epoch:
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, _ = test_offline_clip(clip_model, projector, weighted_gamma,
                                                              test_loader, epoch=epoch, save_name='Test ACC', args=args)
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

            save_dict = {
                'projector': projector.state_dict(),
                'weighted_gamma': weighted_gamma.state_dict(),
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

        exp_lr_scheduler.step()

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: Old: {best_test_acc_old:.4f}')
        args.logger.info('\n')


def test_offline_clip(clip_model, projector, weighted_gamma, test_loader, epoch, save_name, args):
    """离线测试函数"""
    clip_model.eval()
    projector.eval()
    weighted_gamma.eval()

    preds, targets = [], []
    mask = np.array([])

    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)

        # 获取CLIP融合特征
        fusion_feat = get_clip_features_and_fusion(clip_model, weighted_gamma, images)

        with torch.no_grad():
            _, logits = projector(fusion_feat.float())
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


'''online train and test with CLIP'''
'''====================================================================================================================='''


def train_online_clip(clip_model, projector_cur, projector_pre, weighted_gamma, proto_aug_manager,
                      train_loader, test_loader, current_session, args):
    """在线训练阶段 - 使用CLIP特征"""

    # 设置优化器
    optimizer = SGD(list(projector_cur.parameters()) + list(weighted_gamma.parameters()),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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

    best_test_acc_all = 0
    best_test_acc_old = 0
    best_test_acc_new = 0
    best_test_acc_soft_all = 0
    best_test_acc_seen = 0
    best_test_acc_unseen = 0

    for epoch in range(args.epochs_online_per_session):
        loss_record = AverageMeter()

        clip_model.eval()  # CLIP保持eval模式
        projector_cur.train()
        projector_pre.eval()
        weighted_gamma.train()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, _ = batch
            mask_lab = torch.zeros_like(class_labels)  # 所有样本都是无标签的

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

            # 处理两个视图
            fusion_feat_view1 = get_clip_features_and_fusion(clip_model, weighted_gamma, images[0].cuda())
            fusion_feat_view2 = get_clip_features_and_fusion(clip_model, weighted_gamma, images[1].cuda())
            fusion_feat = torch.cat([fusion_feat_view1, fusion_feat_view2], dim=0)

            # 通过当前投影头
            student_proj, student_out = projector_cur(fusion_feat.float())
            teacher_out = student_out.detach()

            # 聚类损失
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)

            # 分组最大熵正则化
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)

            # 1. 旧类vs新类之间的熵正则化
            avg_probs_old_in = avg_probs[:args.num_seen_classes]
            avg_probs_new_in = avg_probs[args.num_seen_classes:]

            avg_probs_old_marginal, avg_probs_new_marginal = torch.sum(avg_probs_old_in), torch.sum(avg_probs_new_in)
            me_max_loss_old_new = avg_probs_old_marginal * torch.log(
                avg_probs_old_marginal) + avg_probs_new_marginal * torch.log(avg_probs_new_marginal) + math.log(2)

            # 2. 旧类内部和新类内部的熵正则化
            avg_probs_old_in_norm = avg_probs_old_in / torch.sum(avg_probs_old_in)
            avg_probs_new_in_norm = avg_probs_new_in / torch.sum(avg_probs_new_in)
            me_max_loss_old_in = - torch.sum(torch.log(avg_probs_old_in_norm ** (-avg_probs_old_in_norm))) + math.log(
                float(len(avg_probs_old_in_norm)))

            if args.num_novel_class_per_session > 1:
                me_max_loss_new_in = - torch.sum(
                    torch.log(avg_probs_new_in_norm ** (-avg_probs_new_in_norm))) + math.log(
                    float(len(avg_probs_new_in_norm)))
            else:
                me_max_loss_new_in = torch.tensor(0.0, device=fusion_feat.device)

            # 总的最大熵损失
            cluster_loss += args.memax_old_new_weight * me_max_loss_old_new + \
                            args.memax_old_in_weight * me_max_loss_old_in + args.memax_new_in_weight * me_max_loss_new_in

            # 对比学习损失
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # ProtoAug损失
            proto_aug_loss = proto_aug_manager.compute_proto_aug_hardness_aware_loss_clip(projector_cur)

            # 特征蒸馏损失
            with torch.no_grad():
                _, student_pre_out = projector_pre(fusion_feat.float())
            feat_distill_loss = F.mse_loss(student_out[:, :args.num_seen_classes], student_pre_out)

            # 总损失
            loss = 0
            loss += 1 * cluster_loss
            loss += 1 * contrastive_loss
            loss += args.proto_aug_weight * proto_aug_loss
            loss += args.feat_distill_weight * feat_distill_loss

            # 日志
            pstr = ''
            pstr += f'me_max_loss_old_new: {me_max_loss_old_new.item():.4f} '
            pstr += f'me_max_loss_old_in: {me_max_loss_old_in.item():.4f} '
            pstr += f'me_max_loss_new_in: {me_max_loss_new_in.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
            pstr += f'proto_aug_loss: {proto_aug_loss.item():.4f} '
            pstr += f'feat_distill_loss: {feat_distill_loss.item():.4f} '
            pstr += f'gamma: {weighted_gamma.gamma.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                new_true_ratio = len(class_labels[class_labels >= args.num_seen_classes]) / len(class_labels)
                logits = student_out / 0.1
                preds = logits.argmax(1)
                new_pred_ratio = len(preds[preds >= args.num_seen_classes]) / len(preds)
                args.logger.info(
                    f'Avg old prob: {torch.sum(avg_probs_old_in).item():.4f} | Avg new prob: {torch.sum(avg_probs_new_in).item():.4f} | Pred new ratio: {new_pred_ratio:.4f} | Ground-truth new ratio: {new_true_ratio:.4f}')

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, \
            all_acc_soft_test, seen_acc_test, unseen_acc_test = test_online_clip(clip_model, projector_cur,
                                                                                 weighted_gamma,
                                                                                 test_loader, epoch=epoch,
                                                                                 save_name='Test ACC', args=args)
        args.logger.info(
            'Test Accuracies (Hard): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                  new_acc_test))
        args.logger.info(
            'Test Accuracies (Soft): All {:.4f} | Seen {:.4f} | Unseen {:.4f}'.format(all_acc_soft_test, seen_acc_test,
                                                                                      unseen_acc_test))

        exp_lr_scheduler.step()

        save_dict = {
            'projector': projector_cur.state_dict(),
            'weighted_gamma': weighted_gamma.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
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
            f'Metrics with best model on test set (Hard) of session-{current_session}: All (Hard): {best_test_acc_all:.4f} Old: {best_test_acc_old:.4f} New: {best_test_acc_new:.4f}')
        args.logger.info(
            f'Metrics with best model on test set (Soft) of session-{current_session}: All (Soft): {best_test_acc_soft_all:.4f} Seen: {best_test_acc_seen:.4f} Unseen: {best_test_acc_unseen:.4f}')
        args.logger.info('\n')

    # 记录最佳结果
    args.best_test_acc_all_list.append(best_test_acc_all)
    args.best_test_acc_old_list.append(best_test_acc_old)
    args.best_test_acc_new_list.append(best_test_acc_new)
    args.best_test_acc_soft_all_list.append(best_test_acc_soft_all)
    args.best_test_acc_seen_list.append(best_test_acc_seen)
    args.best_test_acc_unseen_list.append(best_test_acc_unseen)


def test_online_clip(clip_model, projector, weighted_gamma, test_loader, epoch, save_name, args):
    """在线测试函数，使用CLIP特征"""
    clip_model.eval()
    projector.eval()
    weighted_gamma.eval()

    preds, targets = [], []
    mask_hard = np.array([])
    mask_soft = np.array([])

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if len(batch) == 2:
            images, label = batch
        elif len(batch) >= 3:
            images, label, _ = batch[:3]

        images = images.cuda(non_blocking=True)

        # 获取CLIP融合特征
        fusion_feat = get_clip_features_and_fusion(clip_model, weighted_gamma, images)

        with torch.no_grad():
            _, logits = projector(fusion_feat.float())

            batch_preds = logits.argmax(1).cpu().numpy()
            preds.append(batch_preds)
            targets.append(label.cpu().numpy())

            mask_hard = np.append(mask_hard, np.array([True if x.item() in range(len(args.train_classes))
                                                       else False for x in label]))
            mask_soft = np.append(mask_soft, np.array([True if x.item() in range(args.num_seen_classes)
                                                       else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 生成混淆矩阵（在最后一个epoch）
    if hasattr(args, 'log_dir') and hasattr(args,
                                            'epochs_online_per_session') and epoch == args.epochs_online_per_session - 1:
        try:
            all_classes = sorted(list(set(targets.tolist() + preds.tolist())))
            cm = confusion_matrix(targets, preds, labels=all_classes)

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, fmt='d', cmap='Blues',
                        xticklabels=all_classes,
                        yticklabels=all_classes)
            plt.xlabel('Prediction')
            plt.ylabel('Ground Truth')
            plt.title(f'Confusion Matrix (Epoch {epoch})')

            current_session = 1
            if hasattr(args, 'num_seen_classes') and hasattr(args, 'num_labeled_classes') and hasattr(args,
                                                                                                      'num_novel_class_per_session'):
                if args.num_seen_classes > args.num_labeled_classes:
                    completed_sessions = (
                                                     args.num_seen_classes - args.num_labeled_classes) // args.num_novel_class_per_session
                    current_session = completed_sessions + 1

            session_str = f"session_{current_session}"
            conf_matrix_path = os.path.join(args.log_dir, f'confusion_matrix_{session_str}.png')
            plt.tight_layout()
            plt.savefig(conf_matrix_path)
            plt.close()
            args.logger.info(f"Confusion matrix saved to: {conf_matrix_path}")

        except Exception as e:
            args.logger.info(f"Could not generate confusion matrix: {str(e)}")

    # 计算指标
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_hard,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    all_acc_soft, seen_acc, unseen_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask_soft,
                                                             T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                             args=args)

    return all_acc, old_acc, new_acc, all_acc_soft, seen_acc, unseen_acc


'''====================================================================================================================='''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CLIP Enhanced Continual GCD',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_workers_test', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='options: cifar10, cifar100, tiny_imagenet, cub, imagenet_100')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

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

    # 熵正则化权重
    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--memax_old_new_weight', type=float, default=2)
    parser.add_argument('--memax_old_in_weight', type=float, default=1)
    parser.add_argument('--memax_new_in_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup) of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    # 聚类引导初始化
    parser.add_argument('--init_new_head', action='store_true', default=False)

    # ProtoAug参数
    parser.add_argument('--proto_aug_weight', type=float, default=1.0)
    parser.add_argument('--feat_distill_weight', type=float, default=1.0)
    parser.add_argument('--radius_scale', type=float, default=1.0)
    parser.add_argument('--hardness_temp', type=float, default=0.1)

    # 增量GCD参数
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

    # CLIP相关参数
    parser.add_argument('--clip_model_path', type=str, default="/home/ps/_jinwei/SimGCD/result/pretrained_model_save/2024_09_13_20_07_28/cifar100_clip_ep100.pth", help='Path to pre-trained CLIP model')
    parser.add_argument('--warm_up_epoch', default=0, type=int,
                        help='Warm up epochs before computing classification loss')

    # 其他参数
    parser.add_argument('--shuffle_classes', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='clip-simgcd-continual', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    args.exp_root = args.exp_root + '_' + args.train_session
    args.exp_name = 'CLIP_Enhanced_CGCD' + '-' + args.train_session

    if args.train_session == 'offline':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels)

    elif args.train_session == 'online':
        args.base_exp_id = 'Old' + str(args.num_labeled_classes) + '_' + 'Ratio' + str(args.prop_train_labels) \
                           + '_' + 'ContinualNum' + str(args.continual_session_num) + '_' + 'UnseenNum' + str(
            args.online_novel_unseen_num) \
                           + '_' + 'SeenNum' + str(args.online_novel_seen_num)

    else:
        raise NotImplementedError

    init_experiment(args, runner_name=['CLIP_Happy'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # 加载CLIP模型
    # ----------------------
    args.logger.info(f'Loading CLIP model from {args.clip_model_path}')
    clip_model = torch.load(args.clip_model_path, map_location='cpu').cuda()
    clip_model.eval()  # CLIP模型保持在eval模式
    for param in clip_model.parameters():
        param.requires_grad = False  # 冻结CLIP参数
    args.logger.info('CLIP model loaded and frozen')

    # 创建特征融合模块
    weighted_gamma = WeightedGamma(args).cuda()

    # ----------------------
    # 模型配置
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768  # CLIP特征维度
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # ----------------------
    # 1. OFFLINE TRAIN
    # ----------------------
    if args.train_session == 'offline':
        args.logger.info('========== offline training with CLIP features ==========')
        args.logger.info('loading dataset...')

        offline_session_train_dataset, offline_session_test_dataset, \
            _online_session_train_dataset_list, _online_session_test_dataset_list, \
            datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
            args.dataset_name, train_transform, test_transform, args)

        # 保存数据集配置
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
        # PROJECTION HEAD
        # ----------------------
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).cuda()

        # ----------------------
        # TRAIN
        # ----------------------
        train_offline_clip(clip_model, projector, weighted_gamma, offline_session_train_loader,
                           offline_session_test_loader, args)

    # ----------------------
    # 2. ONLINE TRAIN
    # ----------------------
    elif args.train_session == 'online':
        args.logger.info('\n\n==================== online continual GCD with CLIP enhancement ====================')
        args.logger.info('loading dataset...')

        _offline_session_train_dataset, _offline_session_test_dataset, \
            online_session_train_dataset_list, online_session_test_dataset_list, \
            datasets, dataset_split_config_dict, novel_targets_shuffle = get_datasets(
            args.dataset_name, train_transform, test_transform, args)

        # 保存数据集配置
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

        # 初始化ProtoAug Manager
        proto_aug_manager = ProtoAugManager(args.feat_dim, args.n_views * args.batch_size,
                                            args.hardness_temp, args.radius_scale, device, args.logger)


        # 为ProtoAug Manager添加CLIP支持的方法
        def compute_proto_aug_hardness_aware_loss_clip(self, projector):
            """为CLIP特征定制的ProtoAug损失计算"""
            prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

            # 难度感知采样
            sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
            sampling_prob = sampling_prob.cpu().numpy()
            prototypes_labels = np.random.choice(len(prototypes), size=(self.batch_size,), replace=True,
                                                 p=sampling_prob)
            prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

            prototypes_sampled = prototypes[prototypes_labels]
            prototypes_augmented = prototypes_sampled + torch.randn((self.batch_size, self.feature_dim),
                                                                    device=self.device) * self.radius * self.radius_scale

            # 通过投影头获取logits
            _, prototypes_output = projector(prototypes_augmented)
            proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

            return proto_aug_loss


        # 绑定新方法到proto_aug_manager
        proto_aug_manager.compute_proto_aug_hardness_aware_loss_clip = compute_proto_aug_hardness_aware_loss_clip.__get__(
            proto_aug_manager, ProtoAugManager)

        # 最佳测试准确率列表
        args.best_test_acc_all_list = []
        args.best_test_acc_old_list = []
        args.best_test_acc_new_list = []
        args.best_test_acc_soft_all_list = []
        args.best_test_acc_seen_list = []
        args.best_test_acc_unseen_list = []

        start_session = 0

        # 增量GCD会话
        for session in range(start_session, args.continual_session_num):
            args.logger.info('\n\n========== begin online continual session-{} ==============='.format(session + 1))

            # 当前会话的数据集
            online_session_train_dataset = online_session_train_dataset_list[session]
            online_session_test_dataset = online_session_test_dataset_list[session]
            online_session_train_loader = DataLoader(online_session_train_dataset, num_workers=args.num_workers,
                                                     batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                     pin_memory=True)
            online_session_test_loader = DataLoader(online_session_test_dataset, num_workers=args.num_workers_test,
                                                    batch_size=256, shuffle=False, pin_memory=False)

            # 已见类别数量
            args.num_seen_classes = args.num_labeled_classes + args.num_novel_class_per_session * session
            args.logger.info('number of seen class (old + seen novel) at the beginning of current session: {}'.format(
                args.num_seen_classes))

            # 当前会话的新类别数量
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

            # 加载前一个会话的检查点
            args.logger.info('loading checkpoints of projector_pre...')
            if session == 0:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes,
                                         nlayers=args.num_mlp_layers).cuda()

                if args.load_offline_id is not None:
                    load_dir_online = os.path.join(exp_root + '_' + 'offline', args.dataset_name, args.load_offline_id,
                                                   'checkpoints', 'model_best.pt')
                    args.logger.info('loading offline checkpoints from: ' + load_dir_online)
                    load_dict = torch.load(load_dir_online)
                    projector_pre.load_state_dict(load_dict['projector'])
                    weighted_gamma.load_state_dict(load_dict['weighted_gamma'])
                    args.logger.info('successfully loaded checkpoints!')
            else:
                projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_seen_classes,
                                         nlayers=args.num_mlp_layers).cuda()

                load_dir_online = args.model_path[:-3] + '_session-' + str(session) + f'_best.pt'
                args.logger.info('loading checkpoints from last online session: ' + load_dir_online)
                load_dict = torch.load(load_dir_online)
                projector_pre.load_state_dict(load_dict['projector'])
                weighted_gamma.load_state_dict(load_dict['weighted_gamma'])
                args.logger.info('successfully loaded checkpoints!')

            # 增量分类器
            args.mlp_out_dim_cur = args.num_labeled_classes + args.num_cur_novel_classes
            args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim_cur))

            projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim_cur,
                                     nlayers=args.num_mlp_layers).cuda()
            args.logger.info('transferring classification head of seen classes...')

            # 转移已见类别的权重
            projector_cur.last_layer.weight_v.data[:args.num_seen_classes] = projector_pre.last_layer.weight_v.data[
                                                                             :args.num_seen_classes]
            projector_cur.last_layer.weight_g.data[:args.num_seen_classes] = projector_pre.last_layer.weight_g.data[
                                                                             :args.num_seen_classes]
            projector_cur.last_layer.weight.data[:args.num_seen_classes] = projector_pre.last_layer.weight.data[
                                                                           :args.num_seen_classes]

            # 初始化新类别头部
            online_session_train_dataset_for_new_head_init = deepcopy(online_session_train_dataset)
            online_session_train_dataset_for_new_head_init.old_unlabelled_dataset.transform = test_transform
            online_session_train_dataset_for_new_head_init.novel_unlabelled_dataset.transform = test_transform
            online_session_train_loader_for_new_head_init = DataLoader(online_session_train_dataset_for_new_head_init,
                                                                       num_workers=args.num_workers_test,
                                                                       batch_size=256, shuffle=False, pin_memory=False)

            if args.init_new_head:
                # 创建用于K-means初始化的临时模型
                temp_model_for_kmeans = lambda images: (None, projector_pre(
                    get_clip_features_and_fusion(clip_model, weighted_gamma, images)))
                new_head = get_kmeans_centroid_for_new_head_clip(temp_model_for_kmeans,
                                                                 online_session_train_loader_for_new_head_init, args,
                                                                 device)

                norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[args.num_seen_classes:],
                                                    dim=-1).mean()
                norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[args.num_seen_classes:],
                                                  dim=-1).mean()
                new_head_weight_v = new_head * norm_new_head_weight_v
                new_head_weight = new_head * norm_new_head_weight
                args.logger.info('initializing classification head of unseen novel classes...')

                projector_cur.last_layer.weight_v.data[args.num_seen_classes:] = new_head_weight_v.data
                projector_cur.last_layer.weight.data[args.num_seen_classes:] = new_head_weight.data

            # 计算离线原型（session = 0时）
            if session == 0:
                args.logger.info('Before Train: compute offline prototypes and radius from {} classes...'.format(
                    args.num_labeled_classes))
                offline_session_train_dataset_for_proto_aug = deepcopy(_offline_session_train_dataset)
                offline_session_train_dataset_for_proto_aug.transform = test_transform
                offline_session_train_loader_for_proto_aug = DataLoader(offline_session_train_dataset_for_proto_aug,
                                                                        num_workers=args.num_workers_test,
                                                                        batch_size=256, shuffle=False, pin_memory=False)

                # 创建用于原型计算的临时模型
                temp_model_for_proto = lambda images: (get_clip_features_and_fusion(clip_model, weighted_gamma, images),
                                                       projector_pre(
                                                           get_clip_features_and_fusion(clip_model, weighted_gamma,
                                                                                        images)))
                proto_aug_manager.update_prototypes_offline_clip(temp_model_for_proto,
                                                                 offline_session_train_loader_for_proto_aug,
                                                                 args.num_labeled_classes)
                save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_offline' + f'.pt')
                args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
                proto_aug_manager.save_proto_aug_dict(save_path)

            # ----------------------
            # TRAIN
            # ----------------------
            train_online_clip(clip_model, projector_cur, projector_pre, weighted_gamma, proto_aug_manager,
                              online_session_train_loader, online_session_test_loader, session + 1, args)

            # 训练后更新在线原型
            args.logger.info(
                'After Train: update online prototypes from {} to {} classes...'.format(args.num_seen_classes,
                                                                                        args.num_labeled_classes + args.num_cur_novel_classes))

            load_dir_online_best = args.model_path[:-3] + '_session-' + str(session + 1) + f'_best.pt'
            args.logger.info('loading best checkpoints current online session: ' + load_dir_online_best)
            load_dict = torch.load(load_dir_online_best)
            projector_cur.load_state_dict(load_dict['projector'])
            weighted_gamma.load_state_dict(load_dict['weighted_gamma'])

            # 创建用于原型更新的临时模型
            temp_model_for_proto_update = lambda images: (
                get_clip_features_and_fusion(clip_model, weighted_gamma, images),
                projector_cur(get_clip_features_and_fusion(clip_model, weighted_gamma, images)))
            proto_aug_manager.update_prototypes_online_clip(temp_model_for_proto_update,
                                                            online_session_train_loader_for_new_head_init,
                                                            args.num_seen_classes,
                                                            args.num_labeled_classes + args.num_cur_novel_classes)
            save_path = os.path.join(args.model_dir, 'ProtoAugDict' + '_session-' + str(session + 1) + f'.pt')
            args.logger.info('Saving ProtoAugDict to {}.'.format(save_path))
            proto_aug_manager.save_proto_aug_dict(save_path)

            # 保存每个会话后的结果
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

        # 打印最终结果
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


def get_kmeans_centroid_for_new_head_clip(model_func, online_session_train_loader, args, device):
    """为CLIP特征定制的K-means初始化函数"""
    from sklearn.cluster import KMeans

    all_feats = []
    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')

    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            feats, _ = model_func(images)  # 使用临时模型函数获取特征
            all_feats.append(feats.cpu().numpy())

    # K-MEANS
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_cur_novel_classes, random_state=0).fit(all_feats)
    centroids_np = kmeans.cluster_centers_
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    with torch.no_grad():
        # 使用前一个投影头来判断哪些质心属于新类别
        from models.utils_simgcd_pro import get_kmeans_centroid_for_new_head
        # 这里需要创建一个临时的包装来兼容原始函数的接口
        class TempModel:
            def __init__(self, model_func):
                self.model_func = model_func

            def __call__(self, images):
                return self.model_func(images)

            def __getitem__(self, idx):
                if idx == 1:  # projector
                    return lambda x: self.model_func(torch.zeros_like(x))[1]  # 返回logits

        temp_model = TempModel(model_func)
        _, logits = temp_model[1](centroids)
        max_logits, _ = torch.max(logits, dim=-1)
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)
        new_head = centroids[proto_idx]

    return new_head


# 为ProtoAugManager添加CLIP支持的方法
def update_prototypes_offline_clip(self, model_func, train_loader, num_labeled_classes):
    """为CLIP特征定制的离线原型更新"""
    all_feats_list = []
    all_labels_list = []

    for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats, _ = model_func(images)
            all_feats_list.append(feats)
            all_labels_list.append(label)

    all_feats = torch.cat(all_feats_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)

    # 计算原型和半径
    prototypes_list = []
    radius_list = []
    for c in range(num_labeled_classes):
        feats_c = all_feats[all_labels == c]
        feats_c_mean = torch.mean(feats_c, dim=0)
        prototypes_list.append(feats_c_mean)
        feats_c_center = feats_c - feats_c_mean
        cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)
        radius = torch.trace(cov) / self.feature_dim
        radius_list.append(radius)

    avg_radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
    prototypes_all = torch.stack(prototypes_list, dim=0)
    prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

    # 更新
    self.radius = avg_radius
    self.prototypes = prototypes_all

    # 更新平均相似度
    similarity = prototypes_all @ prototypes_all.T
    for i in range(len(similarity)):
        similarity[i, i] -= similarity[i, i]
    mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
    self.mean_similarity = mean_similarity


def update_prototypes_online_clip(self, model_func, train_loader, num_seen_classes, num_all_classes):
    """为CLIP特征定制的在线原型更新"""
    all_preds_list = []
    all_feats_list = []

    for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feats, logits = model_func(images)
            all_feats_list.append(feats)
            all_preds_list.append(logits.argmax(1))

    all_feats = torch.cat(all_feats_list, dim=0)
    all_preds = torch.cat(all_preds_list, dim=0)

    # 计算新类别原型
    prototypes_list = []
    for c in range(num_seen_classes, num_all_classes):
        feats_c = all_feats[all_preds == c]
        if len(feats_c) == 0:
            self.logger.info('No pred of this class, using random initialization...')
            feats_c_mean = torch.randn(self.feature_dim, device=self.device)
        else:
            self.logger.info('computing (predicted) class-wise mean...')
            feats_c_mean = torch.mean(feats_c, dim=0)
        prototypes_list.append(feats_c_mean)

    prototypes_cur = torch.stack(prototypes_list, dim=0)
    prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
    prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

    # 更新
    self.prototypes = prototypes_all

    # 更新平均相似度
    similarity = prototypes_all @ prototypes_all.T
    for i in range(len(similarity)):
        similarity[i, i] -= similarity[i, i]
    mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity) - 1)
    self.mean_similarity = mean_similarity


# 将新方法绑定到ProtoAugManager类
ProtoAugManager.update_prototypes_offline_clip = update_prototypes_offline_clip
ProtoAugManager.update_prototypes_online_clip = update_prototypes_online_clip