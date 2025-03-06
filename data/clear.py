import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import random
from copy import deepcopy
import glob


class CLEAR10Dataset(Dataset):
    def __init__(self, root_dir, time_buckets=None, classes=None, transform=None, is_train=True):
        """
        CLEAR10数据集加载器

        参数:
            root_dir: 数据集根目录
            time_buckets: 要加载的时间bucket列表，如[1,2,3]
            classes: 要加载的类别列表，如['computer', 'camera']
            transform: 图像变换
            is_train: 是否加载训练集(True)或测试集(False)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 如果未指定时间bucket，则加载所有
        if time_buckets is None:
            self.time_buckets = list(range(1, 11))  # 1-10
        else:
            self.time_buckets = time_buckets

        # 获取所有可用类别
        all_classes = self._get_all_classes()

        # 如果未指定类别，则加载所有类别
        if classes is None:
            self.classes = all_classes
        else:
            self.classes = [c for c in classes if c in all_classes]

        # 类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 加载所有图像路径和标签
        self.samples = []
        self.targets = []
        self.time_ids = []
        self.uq_idxs = []  # 唯一索引，用于子采样

        idx = 0
        for time_id in self.time_buckets:
            for class_name in self.classes:
                img_dir = os.path.join(self.root_dir, "labeled_images", str(time_id), class_name)

                # 获取所有图像路径
                img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))

                # 区分训练集和测试集(每类前250张为训练，后50张为测试)
                if self.is_train:
                    img_paths = img_paths[:250]
                else:
                    img_paths = img_paths[250:300]

                for img_path in img_paths:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.targets.append(self.class_to_idx[class_name])
                    self.time_ids.append(time_id)
                    self.uq_idxs.append(idx)
                    idx += 1

        self.targets = np.array(self.targets)
        self.uq_idxs = np.array(self.uq_idxs)

    def _get_all_classes(self):
        """获取数据集中所有可用的类别"""
        classes = []
        # 从第一个时间bucket中获取所有类别
        class_dirs = glob.glob(os.path.join(self.root_dir, "labeled_images", "1", "*"))
        for class_dir in class_dirs:
            if os.path.isdir(class_dir):
                classes.append(os.path.basename(class_dir))
        return sorted(classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target


def subsample_classes(dataset, include_classes=None):
    """
    只保留指定类别的样本

    参数:
        dataset: 原始数据集
        include_classes: 要包含的类别列表
    """
    if include_classes is None:
        return dataset

    include_classes_idx = [dataset.class_to_idx[cls] for cls in include_classes
                           if cls in dataset.class_to_idx]

    mask = np.isin(dataset.targets, include_classes_idx)
    indices = np.where(mask)[0]

    subset = deepcopy(dataset)
    subset.samples = [dataset.samples[i] for i in indices]
    subset.targets = dataset.targets[indices]
    subset.uq_idxs = dataset.uq_idxs[indices]
    if hasattr(dataset, 'time_ids'):
        subset.time_ids = [dataset.time_ids[i] for i in indices]

    return subset


def subsample_instances(dataset, prop_indices_to_subsample=0.8, seed=0):
    """
    随机子采样指定比例的样本

    参数:
        dataset: 原始数据集
        prop_indices_to_subsample: 要采样的比例
        seed: 随机种子
    """
    np.random.seed(seed)
    n_to_sample = int(prop_indices_to_subsample * len(dataset))
    indices = np.random.choice(range(len(dataset)), n_to_sample, replace=False)
    return indices


def subsample_dataset(dataset, indices):
    """
    根据索引子采样数据集

    参数:
        dataset: 原始数据集
        indices: 要保留的样本索引
    """
    subset = deepcopy(dataset)
    subset.samples = [dataset.samples[i] for i in indices]
    subset.targets = dataset.targets[indices]
    subset.uq_idxs = dataset.uq_idxs[indices]
    if hasattr(dataset, 'time_ids'):
        subset.time_ids = [dataset.time_ids[i] for i in indices]

    return subset


class subDataset_wholeDataset(Dataset):
    """
    将多个子数据集合并为一个数据集
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum([0] + self.lengths)

        # 合并属性
        self.samples = []
        targets = []
        uq_idxs = []
        time_ids = []

        for dataset in self.datasets:
            self.samples.extend(dataset.samples)
            targets.append(dataset.targets)
            uq_idxs.append(dataset.uq_idxs)
            if hasattr(dataset, 'time_ids'):
                time_ids.extend(dataset.time_ids)

        self.targets = np.concatenate(targets)
        self.uq_idxs = np.concatenate(uq_idxs)
        if time_ids:
            self.time_ids = time_ids

        # 继承第一个数据集的类别信息
        self.classes = self.datasets[0].classes
        self.class_to_idx = self.datasets[0].class_to_idx

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.offsets, idx, side='right') - 1
        sample_idx = idx - self.offsets[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]


def get_clear10_datasets(root_dir, train_transform, test_transform, config_dict,
                         offline_classes=None, is_shuffle=True, seed=0):
    """
    为CGCD任务准备CLEAR10数据集

    参数:
        root_dir: CLEAR10数据集根目录
        train_transform: 训练集图像变换
        test_transform: 测试集图像变换
        config_dict: 配置字典
        offline_classes: 离线阶段的类别（如果为None，则使用前7个类别）
        is_shuffle: 是否打乱新类别的顺序
        seed: 随机种子
    """
    continual_session_num = config_dict['continual_session_num']  # 3
    online_novel_unseen_num = config_dict['online_novel_unseen_num']  # 每个新类别的样本数
    online_old_seen_num = config_dict['online_old_seen_num']  # 每个旧类别的样本数
    online_novel_seen_num = config_dict['online_novel_seen_num']  # 每个已见过新类别的样本数
    prop_train_labels = 0.8  # 标记数据的比例

    # 加载完整的训练集（所有时间bucket，所有类别）
    whole_training_set = CLEAR10Dataset(root_dir=root_dir, transform=train_transform, is_train=True)

    # 获取所有类别
    all_classes = whole_training_set.classes
    print(f"CLEAR10数据集共有 {len(all_classes)} 个类别: {all_classes}")

    # 如果未指定离线类别，则使用前7个类别
    if offline_classes is None:
        offline_classes = all_classes[:7]

    # 获取旧类别（离线阶段）的所有样本
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=offline_classes)

    # 为每个旧类别创建子集
    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[cls])
                            for cls in offline_classes]

    # 为每个类别的样本划分为有标签和无标签部分
    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels, seed=seed)
                               for samples in each_old_all_samples]

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.targets)))) - set(each_old_labeled_slices[i])))
        for i, samples in enumerate(each_old_all_samples)
    ]

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                for i, samples in enumerate(each_old_all_samples)]

    each_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                  for i, samples in enumerate(each_old_all_samples)]

    # 离线阶段的训练数据（旧类别的标记样本）
    offline_train_dataset_samples = subDataset_wholeDataset(each_old_labeled_samples)

    # 测试集（所有时间bucket，所有类别）
    test_dataset = CLEAR10Dataset(root_dir=root_dir, transform=test_transform, is_train=False)

    # 离线阶段的测试数据（仅包含旧类别）
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=offline_classes)

    # 在线阶段的旧类别未标记样本
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # 随机采样每个旧类别的样本
        online_session_each_old_slices = [
            np.random.choice(np.array(list(range(len(samples.targets)))),
                             min(online_old_seen_num, len(samples.targets)),
                             replace=False)
            for samples in each_old_unlabeled_samples
        ]

        online_session_old_samples = [
            subsample_dataset(deepcopy(samples), online_session_each_old_slices[i])
            for i, samples in enumerate(each_old_unlabeled_samples)
        ]

        online_session_old_dataset = subDataset_wholeDataset(online_session_old_samples)
        online_old_dataset_unlabelled_list.append(online_session_old_dataset)

    # 获取新类别（在线阶段）
    novel_classes = [cls for cls in all_classes if cls not in offline_classes]

    # 如果需要，打乱新类别的顺序
    if is_shuffle:
        np.random.seed(seed)
        np.random.shuffle(novel_classes)

    # 在线阶段的新类别未标记样本
    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []

    # 每个会话的新类别数量
    novel_classes_per_session = len(novel_classes) // continual_session_num

    for s in range(continual_session_num):
        # 当前会话要处理的所有新类别（包括之前的和新的）
        current_session_novel_classes = novel_classes[:s * novel_classes_per_session + novel_classes_per_session]

        # 按类别分组获取样本
        online_session_novel_samples = []

        for i, cls in enumerate(current_session_novel_classes):
            # 获取该类别的所有样本
            cls_samples = subsample_classes(deepcopy(whole_training_set), include_classes=[cls])

            # 决定采样数量（已见过的类别少采样，新类别多采样）
            if (s >= 1) and (i < s * novel_classes_per_session):
                # 已见过的新类别
                sample_size = min(online_novel_seen_num, len(cls_samples))
            else:
                # 首次出现的新类别
                sample_size = min(online_novel_unseen_num, len(cls_samples))

            # 随机采样
            indices = np.random.choice(np.array(list(range(len(cls_samples)))), sample_size, replace=False)
            cls_subset = subsample_dataset(deepcopy(cls_samples), indices)
            online_session_novel_samples.append(cls_subset)
            online_session_novel_samples.append(cls_subset)

        # 合并该会话的所有新类别样本
        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)

        # 创建该会话的测试数据集（包含所有旧类别和到目前为止的新类别）
        current_test_classes = offline_classes + current_session_novel_classes
        online_session_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=current_test_classes)
        online_test_dataset_list.append(online_session_test_dataset)

    # 汇总所有数据集
    all_datasets = {
        'offline_train_dataset': offline_train_dataset_samples,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    return all_datasets, novel_classes


# 示例使用
if __name__ == "__main__":
    # 配置参数
    config_dict = {
        'continual_session_num': 3,
        'online_novel_unseen_num': 200,  # 每个新类别的样本数（首次出现）
        'online_old_seen_num': 50,  # 每个旧类别的样本数
        'online_novel_seen_num': 50,  # 每个已见过新类别的样本数
    }

    # 图像变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取数据集
    datasets, novel_classes = get_clear10_datasets(
        root_dir="/path/to/CLEAR10",
        train_transform=train_transform,
        test_transform=test_transform,
        config_dict=config_dict,
        seed=42
    )

    # 打印数据集信息
    print(f"离线训练集大小: {len(datasets['offline_train_dataset'])}")
    print(f"离线测试集大小: {len(datasets['offline_test_dataset'])}")

    for i, dataset in enumerate(datasets['online_old_dataset_unlabelled_list']):
        print(f"会话 {i + 1} 旧类别未标记数据大小: {len(dataset)}")

    for i, dataset in enumerate(datasets['online_novel_dataset_unlabelled_list']):
        print(f"会话 {i + 1} 新类别未标记数据大小: {len(dataset)}")

    for i, dataset in enumerate(datasets['online_test_dataset_list']):
        print(f"会话 {i + 1} 测试集大小: {len(dataset)}")

    print(f"新类别顺序: {novel_classes}")