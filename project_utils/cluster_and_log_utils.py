from project_utils.cluster_utils import cluster_acc, np, linear_assignment
from typing import List


# def split_cluster_acc_v1(y_true, y_pred, mask, args):

#     """
#     Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
#     (Mask usually corresponding to `Old' and `New' classes in GCD setting)
#     :param targets: All ground truth labels
#     :param preds: All predictions
#     :param mask: Mask defining two subsets
#     :return:
#     """

#     mask = mask.astype(bool)
#     y_true = y_true.astype(int)
#     y_pred = y_pred.astype(int)
#     weight = mask.mean()

#     old_acc = cluster_acc(y_true[mask], y_pred[mask])
#     if args.train_session == 'online':
#         new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
#         total_acc = weight * old_acc + (1 - weight) * new_acc
#     else:
#         new_acc = None
#         total_acc = old_acc

#     return total_acc, old_acc, new_acc


# def split_cluster_acc_v2(y_true, y_pred, mask, args):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     First compute linear assignment on all data, then look at how good the accuracy is on subsets
#
#     # Arguments
#         mask: Which instances come from old classes (True) and which ones come from new classes (False)
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(int)
#
#     old_classes_gt = set(y_true[mask])
#     new_classes_gt = set(y_true[~mask])
#
#     assert y_pred.size == y_true.size
#     #
#     # # 检查是否在累积会话模式，且使用mask_soft
#     # if hasattr(args, 'test_mode') and args.test_mode == 'cumulative_session' and hasattr(args,
#     #                                                                                      'mask_type') and args.mask_type == 'soft':
#     #     # 分离已见和未见类别的映射计算
#     #     from scipy.optimize import linear_sum_assignment
#     #
#     #     # 已见类别的映射
#     #     seen_mask = mask
#     #     unseen_mask = ~mask
#     #
#     #     seen_preds = y_pred[seen_mask]
#     #     seen_targets = y_true[seen_mask]
#     #     unseen_preds = y_pred[unseen_mask]
#     #     unseen_targets = y_true[unseen_mask]
#     #
#     #     # 为已见类别构建映射
#     #     seen_classes_pred = np.unique(seen_preds)
#     #     seen_classes_target = np.unique(seen_targets)
#     #     seen_classes = np.union1d(seen_classes_pred, seen_classes_target)
#     #
#     #     if len(seen_classes) > 0:
#     #         seen_class_to_idx = {c: i for i, c in enumerate(seen_classes)}
#     #         seen_w = np.zeros((len(seen_classes), len(seen_classes)), dtype=int)
#     #
#     #         for i in range(len(seen_preds)):
#     #             pred_idx = seen_class_to_idx[seen_preds[i]]
#     #             target_idx = seen_class_to_idx[seen_targets[i]]
#     #             seen_w[pred_idx, target_idx] += 1
#     #
#     #         seen_row_ind, seen_col_ind = linear_sum_assignment(seen_w.max() - seen_w)
#     #         seen_map = {seen_classes[i]: seen_classes[j] for i, j in zip(seen_row_ind, seen_col_ind)}
#     #     else:
#     #         seen_map = {}
#     #
#     #     # 为未见类别构建映射
#     #     unseen_map = {}
#     #     if len(unseen_preds) > 0:
#     #         unseen_classes_pred = np.unique(unseen_preds)
#     #         unseen_classes_target = np.unique(unseen_targets)
#     #
#     #         if len(unseen_classes_pred) > 0 and len(unseen_classes_target) > 0:
#     #             unseen_w = np.zeros((len(unseen_classes_pred), len(unseen_classes_target)), dtype=int)
#     #             unseen_pred_to_idx = {c: i for i, c in enumerate(unseen_classes_pred)}
#     #             unseen_target_to_idx = {c: i for i, c in enumerate(unseen_classes_target)}
#     #
#     #             for i in range(len(unseen_preds)):
#     #                 pred_idx = unseen_pred_to_idx[unseen_preds[i]]
#     #                 target_idx = unseen_target_to_idx[unseen_targets[i]]
#     #                 unseen_w[pred_idx, target_idx] += 1
#     #
#     #             unseen_row_ind, unseen_col_ind = linear_sum_assignment(unseen_w.max() - unseen_w)
#     #             for i, j in zip(unseen_row_ind, unseen_col_ind):
#     #                 pred_class = unseen_classes_pred[i]
#     #                 target_class = unseen_classes_target[j]
#     #                 # 确保映射有足够支持
#     #                 if unseen_w[i, j] > 0:
#     #                     unseen_map[pred_class] = target_class
#     #
#     #     # 合并映射
#     #     class_map = {**seen_map, **unseen_map}
#     #
#     #     # 应用映射到整个预测集
#     #     remapped_preds = np.array([class_map.get(p, p) for p in y_pred])
#     #
#     #     # 计算总体准确率
#     #     total_acc = np.sum(remapped_preds == y_true) / len(y_true)
#     #
#     #     # 计算旧类准确率
#     #     old_preds = remapped_preds[mask]
#     #     old_targets = y_true[mask]
#     #     old_acc = np.sum(old_preds == old_targets) / max(len(old_targets), 1)
#     #
#     #     # 计算新类准确率
#     #     if args.train_session == 'online':
#     #         new_preds = remapped_preds[~mask]
#     #         new_targets = y_true[~mask]
#     #         new_acc = np.sum(new_preds == new_targets) / max(len(new_targets), 1)
#     #     else:
#     #         new_acc = None
#     #
#     #     return total_acc, old_acc, new_acc
#     #
#     # # 标准匈牙利算法处理（原始代码）
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=int)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     ind = linear_assignment(w.max() - w)
#     ind = np.vstack(ind).T
#
#     ind_map = {j: i for i, j in ind}
#     total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
#
#     old_acc = 0
#     total_old_instances = 0
#     for i in old_classes_gt:
#         old_acc += w[ind_map[i], i]
#         total_old_instances += sum(w[:, i])
#     old_acc /= total_old_instances
#
#     if args.train_session == 'online':
#         new_acc = 0
#         total_new_instances = 0
#         for i in new_classes_gt:
#             new_acc += w[ind_map[i], i]
#             total_new_instances += sum(w[:, i])
#         new_acc /= total_new_instances
#     else:
#         new_acc = None
#
#     return total_acc, old_acc, new_acc
#
#
# # EVAL_FUNCS保持不变
# EVAL_FUNCS = {
#     # 'v1': split_cluster_acc_v1,
#     'v2': split_cluster_acc_v2,
# }
#
# def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int = None, print_output=False,
#                         args=None):
#     """
#     Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results
#
#     :param y_true: GT labels
#     :param y_pred: Predicted indices
#     :param mask: Which instances belong to Old and New classes
#     :param T: Epoch
#     :param eval_funcs: Which evaluation functions to use
#     :param save_name: What are we evaluating ACC on
#     :return:
#     """
#
#     # 添加一个标记来表明当前使用的是哪种mask (hard或soft)
#     if hasattr(args, 'test_mode') and args.test_mode == 'cumulative_session':
#         # 检查这是否是第二次调用（处理soft mask的调用）
#         if save_name == 'Test ACC' and not hasattr(args, 'first_call_done'):
#             args.mask_type = 'hard'
#             args.first_call_done = True
#         else:
#             args.mask_type = 'soft'
#
#     mask = mask.astype(bool)
#     y_true = y_true.astype(int)
#     y_pred = y_pred.astype(int)
#
#     for i, f_name in enumerate(eval_funcs):
#         acc_f = EVAL_FUNCS[f_name]
#         all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask, args)
#         log_name = f'{save_name}_{f_name}'
#
#         if i == 0:
#             to_return = (all_acc, old_acc, new_acc)
#
#         if print_output:
#             print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
#             print(print_str)
#
#     # 重置标记，以便下一轮评估
#     if hasattr(args, 'mask_type'):
#         delattr(args, 'mask_type')
#
#     return to_return

def split_cluster_acc_v2(y_true, y_pred, mask, args):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    if args.train_session == 'online':
        new_acc = 0
        total_new_instances = 0
        for i in new_classes_gt:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
        new_acc /= total_new_instances
    else:
        new_acc = None

    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    #'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, print_output=False, args=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask, args)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return