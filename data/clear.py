#############################################
# File: clear.py
# Description: Example code for loading CLEAR10
#              in a CGCD-like fashion with
#              1 offline + 3 online sessions.
#############################################

import os
import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder


def subsample_dataset(dataset, indices):
    """
    从给定 dataset 中抽取指定 indices 的子集，返回一个新的 Subset。
    如果 indices 为空，返回 None。
    """
    if len(indices) == 0:
        return None
    return Subset(dataset, indices)


def subsample_classes_by_index(dataset, include_classes):
    """
    在一个 ImageFolder / Subset 上，保留指定类别 (include_classes)，
    并返回对应的 Subset。

    注意：ImageFolder 的 targets 可以通过
          dataset.samples[i][1] 或 dataset.dataset.samples[...] 访问，
          如果是多重 Subset，需要递归索引。
    """

    # 拿到所有 (image_path, label) 以及 label
    all_indices = []
    for idx in range(len(dataset)):
        # 取该 idx 对应的全局标签
        img_path, label = get_item_path_label(dataset, idx)
        if label in include_classes:
            all_indices.append(idx)

    return subsample_dataset(dataset, all_indices)


def split_labeled_unlabeled(dataset, prop=0.8, seed=0):
    """
    将 dataset 中的样本打乱后拆分成两部分：labeled(比例=prop) + unlabeled(比例=1-prop)。
    返回 (labeled_indices, unlabeled_indices)。
    """
    all_indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(all_indices)

    split_pos = int(len(all_indices) * prop)
    labeled_indices = all_indices[:split_pos]
    unlabeled_indices = all_indices[split_pos:]
    return labeled_indices, unlabeled_indices


def subsample_old_classes_unlabeled(dataset, old_classes, num_per_class=250, seed=0):
    """
    从 dataset 中筛选出指定旧类 old_classes 的样本，
    然后对每个类随机选出 num_per_class 条做无标签数据。
    返回合并后的 Subset。
    """
    random.seed(seed)
    selected_indices = []
    # 将 dataset 中属于 old_classes 的样本按类区分
    cls_to_indices = {}
    for idx in range(len(dataset)):
        _, label = get_item_path_label(dataset, idx)
        if label in old_classes:
            if label not in cls_to_indices:
                cls_to_indices[label] = []
            cls_to_indices[label].append(idx)

    # 对每个旧类随机抽取 num_per_class
    for c in old_classes:
        if c in cls_to_indices:  # 防止 dataset 里没出现某类
            candidates = cls_to_indices[c]
            if len(candidates) < num_per_class:
                chosen = candidates  # 如果数据不够就全取
            else:
                chosen = random.sample(candidates, num_per_class)
            selected_indices.extend(chosen)

    return subsample_dataset(dataset, selected_indices)


def subsample_novel_classes_unlabeled(dataset, novel_classes, unseen_num=4000, seen_num=250, session_idx=0, seed=0):
    """
    假设 novel_classes[:session_idx] 是已出现过的新类(novel seen)，
    novel_classes[session_idx] 及之后的类是本轮新出现的(novel unseen)。
    这里演示思路：对“见过的类”抽 seen_num，对“新出现的类”抽 unseen_num。
    具体逻辑可根据你的需求调整。

    注意：如果你希望一次会话出现多个新类，可根据 session_idx 做更复杂拆分。
    """
    random.seed(seed)
    selected_indices = []

    # 将 dataset 中属于 novel_classes 的样本按类区分
    cls_to_indices = {}
    for idx in range(len(dataset)):
        _, label = get_item_path_label(dataset, idx)
        if label in novel_classes:
            if label not in cls_to_indices:
                cls_to_indices[label] = []
            cls_to_indices[label].append(idx)

    # 简单示例: novel_classes[:session_idx] -> “seen”，
    #          novel_classes[session_idx:]   -> “unseen”
    # 你可根据实际需求来决定哪些类算 seen / unseen
    seen_cls = novel_classes[:session_idx]
    unseen_cls = novel_classes[session_idx:]

    for c in novel_classes:
        if c not in cls_to_indices:
            continue  # 该类可能在本文件夹中没有数据
        candidates = cls_to_indices[c]

        # 判断是 seen 还是 unseen
        if c in seen_cls:
            num_pick = seen_num
        else:
            num_pick = unseen_num

        if len(candidates) < num_pick:
            chosen = candidates
        else:
            chosen = random.sample(candidates, num_pick)

        selected_indices.extend(chosen)

    return subsample_dataset(dataset, selected_indices)


def get_item_path_label(dataset, idx):
    """
    通用函数：给定一个 Subset 或 ImageFolder，
    返回该样本对应 (image_path, label)。

    注意如果是多重 Subset，需要一直追溯到底层 dataset。
    """
    # 如果是 Subset，就递归查它的 underlying dataset
    if isinstance(dataset, Subset):
        return get_item_path_label(dataset.dataset, dataset.indices[idx])
    elif isinstance(dataset, ImageFolder):
        # ImageFolder.samples[idx] = (path, class_index)
        path, label = dataset.samples[idx]
        return path, label
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def get_clear_10_datasets(
        root_dir,
        train_transform,
        test_transform,
        config_dict,
        train_classes=range(7),
        prop_train_labels=1.0,
        is_shuffle=False,
        seed=0
):
    """
    强行把 CLEAR10 拆分成:
      - Offline: folder=1 (有标签)
      - Online: folders=2,3,4 (无标签) => 3 个增量会话
    """

    # ---------------------
    # 1) 离线阶段: folder=1
    # ---------------------
    offline_train_all = ImageFolder(
        root=os.path.join(root_dir, "train", "1"),
        transform=train_transform
    )
    offline_test_all = ImageFolder(
        root=os.path.join(root_dir, "test", "1"),
        transform=test_transform
    )

    # 如果 prop_train_labels < 1.0，就拆分一下有标签/无标签
    if prop_train_labels < 1.0:
        labeled_indices, _ = split_labeled_unlabeled(offline_train_all, prop=prop_train_labels, seed=seed)
        offline_train_dataset = Subset(offline_train_all, labeled_indices)
    else:
        offline_train_dataset = offline_train_all

    # 只保留旧类 (train_classes) => 模仿你的 CIFAR 流程
    offline_train_dataset = subsample_classes_by_index(offline_train_dataset, train_classes)
    offline_test_dataset = subsample_classes_by_index(offline_test_all, train_classes)

    # ---------------------
    # 2) 在线阶段: folder=2,3,4 => 3个session
    # ---------------------
    # 读取一些超参
    continual_session_num = 3  # 强制 3 个增量会话
    online_old_seen_num = config_dict.get('online_old_seen_num', 250)
    online_novel_unseen_num = config_dict.get('online_novel_unseen_num', 4000)
    online_novel_seen_num = config_dict.get('online_novel_seen_num', 250)

    # 找出新类
    all_classes = list(range(10))  # CLEAR10
    novel_classes = [c for c in all_classes if c not in train_classes]
    if is_shuffle:
        random.seed(seed)
        random.shuffle(novel_classes)

    # 分别加载 folders=2,3,4
    online_folders = [2, 3, 4]
    online_old_dataset_unlabelled_list = []
    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []

    for s, folder_idx in enumerate(online_folders):
        ds_train = ImageFolder(
            root=os.path.join(root_dir, "train", str(folder_idx)),
            transform=train_transform
        )
        ds_test = ImageFolder(
            root=os.path.join(root_dir, "test", str(folder_idx)),
            transform=test_transform
        )

        # 旧类无标签
        old_unlabeled_ds = subsample_old_classes_unlabeled(
            ds_train,
            old_classes=train_classes,
            num_per_class=online_old_seen_num,
            seed=seed
        )

        # 新类无标签
        # 简单示例：s=0 => novel[:1] => unseen
        #          s=1 => novel[:2], 前一个 seen + 1个 unseen
        #          s=2 => novel[:3], 前两个 seen + 1个 unseen
        novel_unlabeled_ds = subsample_novel_classes_unlabeled(
            ds_train,
            novel_classes=novel_classes,
            unseen_num=online_novel_unseen_num,
            seen_num=online_novel_seen_num,
            session_idx=s,
            seed=seed
        )

        online_old_dataset_unlabelled_list.append(old_unlabeled_ds)
        online_novel_dataset_unlabelled_list.append(novel_unlabeled_ds)

        # 测试集：旧类 + 目前出现的新类 => novel_classes[: s+1]
        current_novel = novel_classes[: s + 1]
        combined_test = subsample_classes_by_index(
            ds_test,
            list(train_classes) + current_novel
        )
        online_test_dataset_list.append(combined_test)

    all_datasets = {
        'offline_train_dataset': offline_train_dataset,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list
    }

    return all_datasets, novel_classes


# # =========== Optional: a quick usage demo ============
#
# if __name__ == "__main__":
#     # 你可在此简单测试一下
#     import torchvision.transforms as T
#
#     # 配置
#     clear_root = "/path/to/CLEAR10"
#     config_dict = {
#         'online_old_seen_num': 250,
#         'online_novel_unseen_num': 4000,
#         'online_novel_seen_num': 250
#     }
#
#     train_transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor()
#     ])
#     test_transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor()
#     ])
#
#     # 获取数据
#     all_datasets, novel_classes = get_clear_10_datasets(
#         root_dir=clear_root,
#         train_transform=train_transform,
#         test_transform=test_transform,
#         config_dict=config_dict,
#         train_classes=range(7),  # 旧类 0~6
#         prop_train_labels=1.0,  # 离线部分全部标注
#         is_shuffle=False,
#         seed=42
#     )
#
#     # 查看返回内容
#     print("离线训练集样本数:", len(all_datasets['offline_train_dataset']))
#     print("离线测试集样本数:", len(all_datasets['offline_test_dataset']))
#
#     for s in range(3):
#         old_ds = all_datasets['online_old_dataset_unlabelled_list'][s]
#         novel_ds = all_datasets['online_novel_dataset_unlabelled_list'][s]
#         test_ds = all_datasets['online_test_dataset_list'][s]
#
#         old_len = len(old_ds) if old_ds else 0
#         novel_len = len(novel_ds) if novel_ds else 0
#         test_len = len(test_ds) if test_ds else 0
#         print(f"[Session {s}] Old unlabeled: {old_len}, Novel unlabeled: {novel_len}, Test: {test_len}")
#
#     print("Novel classes:", novel_classes)