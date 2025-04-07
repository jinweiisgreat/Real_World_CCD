import os
import pandas as pd
import numpy as np
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from config import clear_10_root


def build_class_mapping(root, split='train', domain=1):
    base_path = os.path.join(root, split, str(domain))
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    return {name: idx for idx, name in enumerate(class_names)}


class CustomCLEAR(Dataset):
    def __init__(self, root, split='train', domains=(1,), transform=None, loader=default_loader, class_to_id=None):
        self.root = root
        self.split = split
        self.domains = domains if isinstance(domains, (list, tuple)) else [domains]
        self.transform = transform
        self.loader = loader
        self.class_to_id = class_to_id or build_class_mapping(root, split, self.domains[0])

        self._load_metadata()
        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def _load_metadata(self):
        records = []
        for domain in self.domains:
            domain_path = os.path.join(self.root, self.split, str(domain))
            if not os.path.exists(domain_path):
                continue
            for class_name in sorted(os.listdir(domain_path)):
                if class_name not in self.class_to_id:
                    continue
                class_id = self.class_to_id[class_name]
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        records.append({
                            "filepath": os.path.join(class_path, fname),
                            "target": class_id,
                            "domain": domain
                        })
        self.data = pd.DataFrame(records)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = sample.filepath
        target = sample.target
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True
    dataset.data = dataset.data[mask].reset_index(drop=True)
    dataset.uq_idxs = dataset.uq_idxs[mask]
    return dataset


def subsample_classes(dataset, include_classes):
    include_classes = np.array(include_classes)
    cls_idxs = [i for i, (_, row) in enumerate(dataset.data.iterrows()) if row['target'] in include_classes]

    target_xform_dict = {k: i for i, k in enumerate(include_classes)}
    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]
    return dataset


def subDataset_wholeDataset(datalist):
    whole = deepcopy(datalist[0])
    whole.data = pd.concat([d.data for d in datalist], axis=0).reset_index(drop=True)
    whole.uq_idxs = np.concatenate([d.uq_idxs for d in datalist], axis=0)
    return whole

def subsample_per_class(dataset, num_samples=100, seed=0):
    """
    针对某个 `CustomCLEAR` 数据集，对每个类别只保留最多 `num_samples` 条样本
    并返回一个新的（或就地修改的）dataset。
    """
    # 为了可复现，可以手动设置随机种子
    np.random.seed(seed)

    # dataset.data 是一个 pandas DataFrame，包含 "target" 列
    df = dataset.data
    new_idxs = []

    # 按照 'target' 分组，每个类只保留指定数量的样本
    grouped = df.groupby('target')
    for cls, group_df in grouped:
        all_indices = group_df.index.values
        if len(all_indices) > num_samples:
            selected = np.random.choice(all_indices, size=num_samples, replace=False)
        else:
            selected = all_indices
        new_idxs.extend(selected)

    # 使用 subsample_dataset 来根据 new_idxs 进行真正的子集裁剪
    dataset = subsample_dataset(dataset, new_idxs)
    return dataset


def get_clear_datasets(train_transform, test_transform, config_dict,
                       train_classes=(0, 1, 2, 3, 4, 5, 6),
                       novel_classes=(7, 8, 9),
                       prop_train_labels=1.0,
                       split_train_val=False,
                       is_shuffle=False, seed=0,
                       test_mode='current_session',
                       root=clear_10_root,
                       max_test_per_class=100):

    continual_session_num = config_dict.get('continual_session_num', 3)
    if seed is not None:
        np.random.seed(seed)

    class_to_id = build_class_mapping(root, split='train', domain=1)

    # -----------------------------
    # 1) 构造离线训练/测试集
    # -----------------------------

    train_dataset = CustomCLEAR(root=root, split='train', domains=[1], transform=train_transform, class_to_id=class_to_id)
    offline_train_dataset = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)

    test_dataset = CustomCLEAR(root=root, split='test', domains=[1], transform=test_transform, class_to_id=class_to_id)
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=train_classes)

    # -----------------------------
    # 2) 确定各 session 的新类
    # -----------------------------

    session_novel_class_map = {
        0: [novel_classes[0]],
        1: [novel_classes[0], novel_classes[1]],
        2: list(novel_classes)
    }


    online_old_dataset_unlabelled_list = []
    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    cumulative_test_datasets = [offline_test_dataset]

    # -----------------------------
    # 3) 构造每个 session 的数据集
    # -----------------------------

    for session in range(continual_session_num):
        domain_id = session + 2
        train_domain_dataset = CustomCLEAR(root=root, split='train', domains=[domain_id], transform=train_transform, class_to_id=class_to_id)
        test_domain_dataset = CustomCLEAR(root=root, split='test', domains=[domain_id], transform=test_transform, class_to_id=class_to_id)

        old_samples = []
        for cls in train_classes:
            cls_subset = subsample_classes(deepcopy(train_domain_dataset), include_classes=[cls])
            cls_idxs = list(range(len(cls_subset.data)))
            sample_count = config_dict['online_old_seen_num']
            selected = np.random.choice(cls_idxs, min(sample_count, len(cls_idxs)), replace=False)
            old_samples.append(subsample_dataset(cls_subset, selected))
        session_old_dataset = subDataset_wholeDataset(old_samples)

        novel_samples = []
        for novel_cls in session_novel_class_map[session]:
            novel_subset = subsample_classes(deepcopy(train_domain_dataset), include_classes=[novel_cls])
            sample_count = config_dict['online_novel_unseen_num'] if novel_cls == novel_classes[session] else config_dict['online_novel_seen_num']
            cls_idxs = list(range(len(novel_subset.data)))
            selected = np.random.choice(cls_idxs, min(sample_count, len(cls_idxs)), replace=False)
            novel_samples.append(subsample_dataset(novel_subset, selected))
        session_novel_dataset = subDataset_wholeDataset(novel_samples)

        online_old_dataset_unlabelled_list.append(session_old_dataset)
        online_novel_dataset_unlabelled_list.append(session_novel_dataset)

        test_classes = list(train_classes) + session_novel_class_map[session]
        session_test_dataset = subsample_classes(deepcopy(test_domain_dataset), include_classes=test_classes)

        if test_mode == 'cumulative_session':
            cumulative_test_datasets.append(session_test_dataset)
            combined = subDataset_wholeDataset(cumulative_test_datasets)
            if max_test_per_class is not None:
                combined = subsample_per_class(combined, num_samples=max_test_per_class, seed=seed)
            online_test_dataset_list.append(combined)
        else:
            online_test_dataset_list.append(session_test_dataset)

    all_datasets = {
        'offline_train_dataset': offline_train_dataset,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    novel_targets_shuffle = list(novel_classes)
    if is_shuffle:
        np.random.shuffle(novel_targets_shuffle)

    return all_datasets, novel_targets_shuffle


# Example usage
"""
if __name__ == '__main__':
    from torchvision import transforms

    dummy_transform = transforms.Compose([transforms.ToTensor()])

    config_dict = {
        'continual_session_num': 3,
        'online_novel_unseen_num': 300,
        'online_old_seen_num': 50,
        'online_novel_seen_num': 50,
    }

    all_datasets, novel_targets_shuffle = get_clear_datasets(
        train_transform=dummy_transform,
        test_transform=dummy_transform,
        config_dict=config_dict,
        is_shuffle=False
    )

    novel_datasets = all_datasets['online_novel_dataset_unlabelled_list']

    for session_id, dataset in enumerate(novel_datasets):
        print(f"\n[Session {session_id}] Online Novel Unlabelled Info")
        class_counts = dataset.data['filepath'].apply(lambda p: os.path.basename(os.path.dirname(p)))
        counts = class_counts.value_counts()
        for class_name, count in counts.items():
            print(f"Class: {class_name}, Count: {count}")
"""