from PIL import Image
import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from data.data_utils import subsample_instances
from config import clear_10_root


class CustomCLEAR10Dataset(Dataset):
    """
    自定义CLEAR10数据集类，用于支持唯一索引和域信息
    """

    def __init__(self, root=None, transform=None, data=None, targets=None, domain=None):
        self.root = root
        self.transform = transform
        self.domain = domain

        # 如果直接提供了数据和标签
        if data is not None and targets is not None:
            self.data = data
            self.targets = targets
            self.uq_idxs = np.array(range(len(self.targets)))
            return

        # 从文件系统加载数据
        self.data = []
        self.targets = []

        # 遍历指定域文件夹下的所有类别
        domain_path = os.path.join(root, str(domain))
        for class_id, class_name in enumerate(sorted(os.listdir(domain_path))):
            class_path = os.path.join(domain_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.data.append(img_path)  # 存储图片路径
                        self.targets.append(class_id)

        self.targets = np.array(self.targets)
        self.uq_idxs = np.array(range(len(self.targets)))

    def __getitem__(self, item):
        if isinstance(self.data[item], str):  # 如果是路径
            img = Image.open(self.data[item]).convert('RGB')
        else:  # 如果是已加载的图像
            img = self.data[item]

        label = self.targets[item]

        if self.transform is not None:
            img = self.transform(img)

        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    """
    基于索引子采样数据集
    """
    if len(idxs) > 0:
        new_dataset = deepcopy(dataset)
        new_dataset.data = [new_dataset.data[i] for i in idxs]
        new_dataset.targets = new_dataset.targets[idxs]
        new_dataset.uq_idxs = new_dataset.uq_idxs[idxs]
        return new_dataset
    else:
        return None


def subsample_classes(dataset, include_classes=(0, 1, 2, 3, 4, 5, 6)):
    """
    仅保留特定类别的样本
    """
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, np.array(cls_idxs))

    return dataset


def subDataset_wholeDataset(datalist):
    """
    将多个数据集合并为一个
    """
    if not datalist:
        return None

    wholeDataset = deepcopy(datalist[0])

    if isinstance(wholeDataset.data[0], str):  # 图片路径
        wholeDataset.data = []
        for d in datalist:
            wholeDataset.data.extend(d.data)
    else:  # 已加载的图像
        all_data = []
        for d in datalist:
            all_data.extend(d.data)
        wholeDataset.data = all_data

    wholeDataset.targets = np.concatenate([d.targets for d in datalist], axis=0)
    wholeDataset.uq_idxs = np.concatenate([d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_clear_datasets(train_transform, test_transform, config_dict,
                          train_classes=(0, 1, 2, 3, 4, 5, 6),
                          novel_classes=(7, 8, 9),
                          prop_train_labels=1.0,
                          split_train_val=False, is_shuffle=False, seed=0):
    """
    为CLEAR10数据集创建适用于Continual-GCD的数据加载器

    参数:
    - train_transform: 训练集变换
    - test_transform: 测试集变换
    - config_dict: 配置字典
    - train_classes: 初始训练类别索引(0-6)
    - novel_classes: 增量学习类别索引(7-9)
    - split_train_val: 是否分割训练和验证集
    - is_shuffle: 是否打乱新类顺序
    - seed: 随机种子

    返回:
    - 适用于Happy框架的数据集字典
    """
    # 配置在线会话参数
    continual_session_num = config_dict.get('continual_session_num', 3)

    # 使用指定的random seed
    if seed is not None:
        np.random.seed(seed)

    # ===========================================
    # 1. 离线阶段 - 文件夹1前7个类
    # ===========================================

    # 加载文件夹1的训练数据
    domain1_train_dataset = CustomCLEAR10Dataset(
        root=os.path.join(clear_10_root, 'train'),
        transform=train_transform,
        domain=1
    )

    # 提取前7个类作为训练数据
    offline_train_dataset = subsample_classes(
        deepcopy(domain1_train_dataset),
        include_classes=train_classes
    )

    # 加载测试集 - 文件夹1
    test_dataset_full = CustomCLEAR10Dataset(
        root=os.path.join(clear_10_root, 'test'),
        transform=test_transform,
        domain=1
    )

    # 离线测试集 - 只包含7个旧类
    offline_test_dataset = subsample_classes(
        deepcopy(test_dataset_full),
        include_classes=train_classes
    )

    # ===========================================
    # 2. 在线增量阶段
    # ===========================================

    # 每个会话的新类映射
    session_novel_class_map = {
        0: [novel_classes[0]],  # Session 1: racing
        1: [novel_classes[0], novel_classes[1]],  # Session 2: racing, soccer
        2: list(novel_classes)  # Session 3: racing, soccer, sweater
    }

    # 创建在线会话数据集
    online_old_dataset_unlabelled_list = []
    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []

    for session in range(continual_session_num):
        domain_id = session + 2  # 会话1对应文件夹2，依此类推

        # 当前会话需要使用的所有类别
        current_novel_classes = session_novel_class_map[session]

        # 加载当前域的训练数据
        domain_train_dataset = CustomCLEAR10Dataset(
            root=os.path.join(clear_10_root, 'train'),
            transform=train_transform,
            domain=domain_id
        )

        # 1. 提取旧类样本
        old_classes_dataset = subsample_classes(
            deepcopy(domain_train_dataset),
            include_classes=train_classes
        )

        # 为每个旧类随机选择40个样本
        old_samples_list = []
        for cls in train_classes:
            cls_idxs = np.where(old_classes_dataset.targets == cls)[0]
            if len(cls_idxs) > 40:
                selected_idxs = np.random.choice(cls_idxs, 40, replace=False)
                old_samples_list.append(subsample_dataset(deepcopy(old_classes_dataset), selected_idxs))
            else:
                old_samples_list.append(subsample_dataset(deepcopy(old_classes_dataset), cls_idxs))

        # 合并所有旧类样本
        session_old_dataset = subDataset_wholeDataset(old_samples_list)

        # 2. 提取新类样本
        novel_samples_list = []

        for i, novel_cls in enumerate(current_novel_classes):
            # 筛选出该类别的所有样本
            novel_cls_dataset = subsample_classes(
                deepcopy(domain_train_dataset),
                include_classes=[novel_cls]
            )

            # 依据类别在当前会话中的状态决定采样数量
            if novel_cls == novel_classes[session]:  # 当前会话的新类
                sample_count = config_dict['online_novel_unseen_num']
            else:  # 已见过的新类
                sample_count = config_dict['online_novel_seen_num']

            cls_idxs = np.where(novel_cls_dataset.targets == novel_cls)[0]
            if len(cls_idxs) > sample_count:
                selected_idxs = np.random.choice(cls_idxs, sample_count, replace=False)
                novel_samples_list.append(subsample_dataset(deepcopy(novel_cls_dataset), selected_idxs))
            else:
                novel_samples_list.append(subsample_dataset(deepcopy(novel_cls_dataset), cls_idxs))

        # 合并所有新类样本
        session_novel_dataset = subDataset_wholeDataset(novel_samples_list)

        # 添加到会话列表
        online_old_dataset_unlabelled_list.append(session_old_dataset)
        online_novel_dataset_unlabelled_list.append(session_novel_dataset)

        # 3. 创建当前会话的测试集
        # 包含旧类和当前所有出现过的新类
        test_classes = list(train_classes) + current_novel_classes
        session_test_dataset = subsample_classes(
            deepcopy(test_dataset_full),
            include_classes=test_classes
        )
        online_test_dataset_list.append(session_test_dataset)

    # ===========================================
    # 3. 组织返回结果
    # ===========================================

    all_datasets = {
        'offline_train_dataset': offline_train_dataset,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    # 如果需要打乱类别顺序
    if is_shuffle:
        novel_targets_shuffle = list(novel_classes)
        np.random.shuffle(novel_targets_shuffle)
    else:
        novel_targets_shuffle = list(novel_classes)

    return all_datasets, novel_targets_shuffle


# 测试代码，使用时请注释掉

if __name__ == '__main__':
    clear_10_root = '/home/ps/_jinwei/Dataset/CLEAR/CLEAR10_CGCD'
    config_dict = {
        'continual_session_num': 3,
        'online_novel_unseen_num': 600,
        'online_old_seen_num': 40,
        'online_novel_seen_num': 50
    }

    # 类别索引到名称的映射
    class_names = {
        0: 'baseball',
        1: 'bus',
        2: 'camera',
        3: 'cosplay',
        4: 'dress',
        5: 'hockey',
        6: 'laptop',
        7: 'racing',
        8: 'soccer',
        9: 'sweater'
    }

    train_classes = (0, 1, 2, 3, 4, 5, 6)
    novel_classes = (7, 8, 9)

    datasets, novel_shuffle = get_clear_datasets(
        train_transform=None,
        test_transform=None,
        config_dict=config_dict,
        train_classes=train_classes,
        novel_classes=novel_classes
    )

    print("Offline train dataset size:", len(datasets['offline_train_dataset']))
    print("Offline train classes:", [class_names[i] for i in train_classes])
    print("Offline test dataset size:", len(datasets['offline_test_dataset']))
    print("Offline test classes:", [class_names[i] for i in train_classes])

    # 每个会话的新类映射
    session_novel_class_map = {
        0: [novel_classes[0]],  # Session 1: racing
        1: [novel_classes[0], novel_classes[1]],  # Session 2: racing, soccer
        2: list(novel_classes)  # Session 3: racing, soccer, sweater
    }

    for i, dataset in enumerate(datasets['online_old_dataset_unlabelled_list']):
        print(f"Session {i + 1} old dataset size:", len(dataset))
        print(f"Session {i + 1} old classes:", [class_names[c] for c in train_classes])

    for i, dataset in enumerate(datasets['online_novel_dataset_unlabelled_list']):
        print(f"Session {i + 1} novel dataset size:", len(dataset))

        # 区分新类和已见新类
        if i == 0:
            # Session 1只有新类，没有已见新类
            print(f"Session {i + 1} novel classes:")
            print(f"  - 新类: [{class_names[novel_classes[0]]}]")
        elif i == 1:
            # Session 2有一个新类和一个已见新类
            print(f"Session {i + 1} novel classes:")
            print(f"  - 已见新类: [{class_names[novel_classes[0]]}]")
            print(f"  - 新类: [{class_names[novel_classes[1]]}]")
        else:
            # Session 3有一个新类和两个已见新类
            print(f"Session {i + 1} novel classes:")
            print(f"  - 已见新类: [{class_names[novel_classes[0]]}, {class_names[novel_classes[1]]}]")
            print(f"  - 新类: [{class_names[novel_classes[2]]}]")

    for i, dataset in enumerate(datasets['online_test_dataset_list']):
        print(f"Session {i + 1} test dataset size:", len(dataset))
        test_classes = list(train_classes) + session_novel_class_map[i]
        print(f"Session {i + 1} test classes:", [class_names[c] for c in test_classes])