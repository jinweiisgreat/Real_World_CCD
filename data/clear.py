import os
import torch
import numpy as np
import random
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

# 配置参数
clear_root = '~/Dataset/CLEAR/CLEAR10_restructured'  # 修改为您的CLEAR数据集路径

# 默认配置
# dataset_split_config_dict = {
#     'clear10': {
#         'offline_old_cls_num': 7,  # 离线阶段已知类别数
#         'offline_prop_train_labels': 1.0,  # 离线阶段标记数据比例
#         'continual_session_num': 3,  # 持续学习阶段数
#         'online_novel_unseen_num': 480,  # 每个新类别样本数
#         'online_old_seen_num': 40,  # 每个已知类别样本数
#         'online_novel_seen_num': 50,  # 每个已见过新类别样本数
#     }
# }

# 类别名称映射(用于加载数据)
class_names = ["baseball", "bus", "camera", "cosplay", "dress",
               "hockey", "laptop", "racing", "soccer", "sweater"]


class CustomCLEAR(Dataset):
    """CLEAR数据集的自定义类，支持类别筛选和采样"""

    def __init__(self, root_dir, folder, transform=None, train=True,
                 include_classes=None, is_labeled=True):
        """
        初始化CLEAR数据集

        Args:
            root_dir: 数据集根目录
            folder: 使用哪个文件夹(1,2,3,4)
            transform: 图像转换
            train: 是否为训练集
            include_classes: 要包含的类别列表
            is_labeled: 数据是否有标签
        """
        self.root_dir = root_dir
        self.folder = str(folder)
        self.transform = transform
        self.phase = 'train' if train else 'test'
        self.include_classes = include_classes if include_classes is not None else list(range(len(class_names)))
        self.is_labeled = is_labeled

        # 设置路径
        self.data_path = os.path.join(root_dir, self.phase, 'labeled_images', self.folder)

        # 加载数据
        self.data = []
        self.targets = []
        self.uq_idxs = []
        self._load_data()

    def _load_data(self):
        """加载指定文件夹和类别的数据"""
        idx = 0

        for class_idx in self.include_classes:
            class_dir = os.path.join(self.data_path, class_names[class_idx])
            if not os.path.exists(class_dir):
                continue

            # 获取所有图像文件
            files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            for file in files:
                self.data.append(os.path.join(class_dir, file))
                self.targets.append(class_idx if self.is_labeled else -1)  # -1表示无标签
                self.uq_idxs.append(idx)
                idx += 1

    def __getitem__(self, idx):
        """获取数据集中的一项"""
        img_path = self.data[idx]
        target = self.targets[idx]
        uq_idx = self.uq_idxs[idx]

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 应用转换
        if self.transform:
            img = self.transform(img)

        return img, target, uq_idx

    def __len__(self):
        """返回数据集长度"""
        return len(self.data)


def subsample_dataset(dataset, idxs):
    """从数据集中选择指定索引的样本"""
    if len(idxs) > 0:
        dataset_copy = deepcopy(dataset)
        dataset_copy.data = [dataset.data[i] for i in idxs]
        dataset_copy.targets = [dataset.targets[i] for i in idxs]
        dataset_copy.uq_idxs = [dataset.uq_idxs[i] for i in idxs]
        return dataset_copy
    else:
        return None


def subsample_classes(dataset, include_classes):
    """从数据集中选择指定类别的样本"""
    cls_idxs = [i for i, t in enumerate(dataset.targets) if t in include_classes]

    return subsample_dataset(dataset, cls_idxs)


def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    """随机选择数据集中的一部分样本索引"""
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    # 收集所有要子采样的索引
    idxs = []

    for c in classes:
        class_idxs = np.where(targets == c)[0]
        num_to_sample = int(prop_indices_to_subsample * len(class_idxs))
        sampled_idxs = np.random.choice(class_idxs, size=num_to_sample, replace=False)
        idxs.extend(sampled_idxs)

    return np.array(idxs)


def subDataset_wholeDataset(datalist):
    """将多个数据集合并为一个大数据集"""
    wholeDataset = deepcopy(datalist[0])
    wholeDataset.data = []
    wholeDataset.targets = []
    wholeDataset.uq_idxs = []

    idx_offset = 0
    for d in datalist:
        wholeDataset.data.extend(d.data)
        wholeDataset.targets.extend(d.targets)
        # 重新编号唯一索引以避免冲突
        wholeDataset.uq_idxs.extend([idx + idx_offset for idx in d.uq_idxs])
        idx_offset += len(d.uq_idxs)

    return wholeDataset


def get_clear_10_datasets(train_transform, test_transform, config_dict, train_classes=range(7),
                          prop_train_labels=1.0, is_shuffle=False, seed=0):
    """
    获取CLEAR-10数据集用于CGCD任务

    Args:
        train_transform: 训练图像转换
        test_transform: 测试图像转换
        config_dict: 配置字典
        train_classes: 已知类别列表
        prop_train_labels: 标记数据比例
        is_shuffle: 是否打乱新类别
        seed: 随机种子

    Returns:
        所有数据集的字典
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 获取配置参数
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # 已知类别和新类别
    train_classes = list(train_classes)
    all_classes = list(range(len(class_names)))
    novel_classes = [c for c in all_classes if c not in train_classes]

    if is_shuffle:
        random.shuffle(novel_classes)

    '''----------------------------- 离线阶段：已知类别的标记样本 -----------------------------------------------'''
    # 离线阶段(文件夹1)的数据集，使用所有已知类别
    offline_train_dataset = CustomCLEAR(
        root_dir=clear_root,
        folder=1,
        transform=train_transform,
        train=True,
        include_classes=train_classes,
        is_labeled=True
    )

    # 离线阶段的测试数据集
    offline_test_dataset = CustomCLEAR(
        root_dir=clear_root,
        folder=1,
        transform=test_transform,
        train=False,
        include_classes=train_classes,
        is_labeled=True
    )

    '''----------------------------- 在线阶段：所有类别的无标签样本 ----------------------------------------------'''
    # 为每个session创建无标签数据集
    online_old_dataset_unlabelled_list = []  # 已知类别
    online_novel_dataset_unlabelled_list = []  # 新类别
    online_test_dataset_list = []  # 测试集

    for s in range(continual_session_num):
        folder = s + 2  # 文件夹2,3,4对应session 1,2,3

        # 当前session要引入的新类别
        current_novel_classes = novel_classes[:s + 1]
        # 真正新引入的类别
        new_class = [novel_classes[s]]
        # 已见过的新类别
        seen_novel_classes = novel_classes[:s]

        '''--- 已知类别的无标签样本 ---'''
        # 从当前文件夹采样已知类别数据
        session_old_dataset = CustomCLEAR(
            root_dir=clear_root,
            folder=folder,
            transform=train_transform,
            train=True,
            include_classes=train_classes,
            is_labeled=False  # 无标签
        )

        # 为每个类别随机采样
        old_class_samples = {}
        for cls in train_classes:
            cls_idxs = [i for i, t in enumerate(session_old_dataset.targets) if
                        session_old_dataset.data[i].find(class_names[cls]) != -1]

            # 随机采样
            if len(cls_idxs) > online_old_seen_num:
                sampled_idxs = random.sample(cls_idxs, online_old_seen_num)
                old_class_samples[cls] = sampled_idxs
            else:
                old_class_samples[cls] = cls_idxs

        # 合并所有采样的索引
        all_old_idxs = []
        for idxs in old_class_samples.values():
            all_old_idxs.extend(idxs)

        # 子采样数据集
        session_old_sampled = subsample_dataset(session_old_dataset, all_old_idxs)
        online_old_dataset_unlabelled_list.append(session_old_sampled)

        '''--- 新类别的无标签样本 ---'''
        # 从当前文件夹采样新类别和已见新类别
        all_novel_dataset = CustomCLEAR(
            root_dir=clear_root,
            folder=folder,
            transform=train_transform,
            train=True,
            include_classes=current_novel_classes,
            is_labeled=False  # 无标签
        )

        # 为每个新类别和已见新类别随机采样
        novel_samples = []

        # 处理已见新类别
        for cls in seen_novel_classes:
            cls_idxs = [i for i, t in enumerate(all_novel_dataset.targets) if
                        all_novel_dataset.data[i].find(class_names[cls]) != -1]

            # 随机采样
            if len(cls_idxs) > online_novel_seen_num:
                sampled_idxs = random.sample(cls_idxs, online_novel_seen_num)
                novel_samples.extend(sampled_idxs)
            else:
                novel_samples.extend(cls_idxs)

        # 处理新引入的类别
        for cls in new_class:
            cls_idxs = [i for i, t in enumerate(all_novel_dataset.targets) if
                        all_novel_dataset.data[i].find(class_names[cls]) != -1]

            # 随机采样
            if len(cls_idxs) > online_novel_unseen_num:
                sampled_idxs = random.sample(cls_idxs, online_novel_unseen_num)
                novel_samples.extend(sampled_idxs)
            else:
                novel_samples.extend(cls_idxs)

        # 子采样数据集
        session_novel_sampled = subsample_dataset(all_novel_dataset, novel_samples)
        online_novel_dataset_unlabelled_list.append(session_novel_sampled)

        '''--- 测试集 ---'''
        # 包含所有已知类别和当前已见的新类别
        test_classes = train_classes + current_novel_classes
        session_test_dataset = CustomCLEAR(
            root_dir=clear_root,
            folder=folder,
            transform=test_transform,
            train=False,
            include_classes=test_classes,
            is_labeled=True
        )
        online_test_dataset_list.append(session_test_dataset)

    '''---------------------------------- 所有CGCD任务的数据集 -----------------------------------------------'''
    all_datasets = {
        'offline_train_dataset': offline_train_dataset,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    return all_datasets, novel_classes


# def get_transform():
#     """获取标准的图像转换"""
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     return train_transform, test_transform


# if __name__ == '__main__':
#     # 示例用法
#     train_transform, test_transform = get_transform()
#
#     datasets, novel_classes = get_clear_10_datasets(
#         train_transform=train_transform,
#         test_transform=test_transform,
#         config_dict=dataset_split_config_dict['clear10'],
#         train_classes=range(7),  # 前7个类别作为已知类别
#         prop_train_labels=1.0,
#         is_shuffle=True,
#         seed=42
#     )
#
#     # 打印数据集信息
#     print(f"离线训练集样本数: {len(datasets['offline_train_dataset'])}")
#     print(f"离线测试集样本数: {len(datasets['offline_test_dataset'])}")
#
#     for i in range(3):
#         print(f"\nSession {i + 1}:")
#         print(f"  已知类别样本数: {len(datasets['online_old_dataset_unlabelled_list'][i])}")
#         print(f"  新类别样本数: {len(datasets['online_novel_dataset_unlabelled_list'][i])}")
#         print(f"  测试集样本数: {len(datasets['online_test_dataset_list'][i])}")