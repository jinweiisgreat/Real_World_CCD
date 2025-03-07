import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

# --------------------- #
# 1) 数据加载类 (模仿 CustomCub2011)
# --------------------- #
class CustomCLEAR10(Dataset):
    """
    示例：用于加载 CLEAR10 数据集的 Dataset，与 CustomCub2011 结构相似。
    假设数据目录结构:
        root/
          train/
            1/<class_name>/*.jpg
            2/<class_name>/*.jpg
            3/<class_name>/*.jpg
            4/<class_name>/*.jpg
          test/
            1/<class_name>/*.jpg
            2/<class_name>/*.jpg
            3/<class_name>/*.jpg
            4/<class_name>/*.jpg
    """
    base_folder = None   # 不一定需要
    url = None           # 如无下载链接，可留空
    filename = None
    tgz_md5 = None

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 download=False):
        """
        :param root: 根目录。例如 /path/to/CLEAR10
        :param train: 是否是训练集 (True/False)
        :param transform: 图像变换
        :param target_transform: 标签变换
        :param loader: 用于加载图像的函数
        :param download: 如需要在线下载可实现; 否则忽略
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('CLEAR10 dataset not found or corrupted. '
                               'Please verify your dataset path.')

        # 创建索引并加载元数据
        self._load_metadata()
        # 记下 [0..N-1] 的唯一索引
        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        """
        目标：构建 self.data (pandas DataFrame)，
             包含至少: [img_path, target] 以及是否训练/测试等信息
        """
        # train or test
        split_folder = 'train' if self.train else 'test'
        data_records = []

        # 假设有 4 个分桶: 1, 2, 3, 4
        # 如果您只想在离线阶段拿 1 号桶，也可在后续 pipeline 再做过滤
        bucket_ids = [1, 2, 3, 4]

        # 这里给出10个类别名称的示例（可以根据自己文件夹名称进行修改）
        clear10_classes = [
            "baseball", "bus", "camera", "cosplay", "dress",
            "hockey",   "laptop", "racing", "soccer", "sweater"
        ]
        class_to_label = {cls_name: i for i, cls_name in enumerate(clear10_classes)}

        for bucket_id in bucket_ids:
            bucket_str = str(bucket_id)
            bucket_path = os.path.join(self.root, split_folder, bucket_str)
            if not os.path.isdir(bucket_path):
                # 若该桶不存在，可跳过 或 raise 警告
                continue

            # 遍历bucket_path下的所有子文件夹(类)
            for cls_name in os.listdir(bucket_path):
                cls_dir = os.path.join(bucket_path, cls_name)
                if not os.path.isdir(cls_dir):
                    continue

                label = class_to_label.get(cls_name, None)
                if label is None:
                    # 未知类别，可根据需求做处理
                    continue

                # 扫描该类别文件夹下的所有jpg/png
                for fname in os.listdir(cls_dir):
                    fpath = os.path.join(cls_dir, fname)
                    if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
                        continue

                    # 记录: 图像路径, 类别(0~9), 以及桶ID等信息
                    data_records.append({
                        'img_path': fpath,
                        'target': label,
                        'bucket': bucket_id
                    })

        # 转成 pandas DataFrame, 类似 cub.py 里的 self.data
        self.data = pd.DataFrame(data_records)

    def _check_integrity(self):
        # 如无校验需求，这里直接返回 True 即可
        # 也可做一些简单的路径检查
        return True

    def _download(self):
        # CLEAR10 若无官方 tar 包下载，可不实现
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回 (image, target, uq_idx)
        与 cub.py 保持同样的三元组格式。
        """
        record = self.data.iloc[idx]
        img_path = record['img_path']
        target = record['target']
        img = self.loader(img_path)  # default_loader默认调用 PIL.Image.open

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


# --------------------- #
# 2) 工具函数 (与 cub.py 对齐)
# --------------------- #
def subsample_dataset(dataset, idxs):
    """
    根据给定索引 idxs 对 dataset.data 和 dataset.uq_idxs 做过滤。
    """
    mask = np.zeros(len(dataset), dtype=bool)
    mask[idxs] = True

    dataset.data = dataset.data[mask].reset_index(drop=True)
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(7)):
    """
    只保留 include_classes 里的类别样本，
    并把它们的标签重新映射到连续 [0..len(include_classes)-1] 区间。
    """
    include_classes = list(include_classes)
    cls_idxs = [i for i, row in dataset.data.iterrows() if row['target'] in include_classes]

    dataset = subsample_dataset(dataset, cls_idxs)

    # 类别映射
    target_xform_dict = {}
    for i, old_cls in enumerate(include_classes):
        target_xform_dict[old_cls] = i

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def subDataset_wholeDataset(datalist):
    """
    把 datalist 里若干子数据集的 DataFrame 和 uq_idxs 拼接起来，
    返回一个新的大 dataset。
    """
    from copy import deepcopy
    if len(datalist) == 0:
        return None
    wholeDataset = deepcopy(datalist[0])
    # pandas concat
    frames = [d.data for d in datalist]
    wholeDataset.data = pd.concat(frames, axis=0, ignore_index=True)
    # uq_idxs
    all_uq_idxs = [d.uq_idxs for d in datalist]
    wholeDataset.uq_idxs = np.concatenate(all_uq_idxs)
    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):
    """
    与 cub.py 类似，对 train_dataset 按类别分割出 train/val 索引
    """
    train_classes = np.unique(train_dataset.data['target'])

    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]
        val_count = int(val_split * len(cls_idxs))
        v_ = np.random.choice(cls_idxs, replace=False, size=val_count)
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


# --------------------- #
# 3) 示例：get_clear_datasets
# --------------------- #
def get_clear_datasets(train_transform,
                       test_transform,
                       config_dict,
                       old_classes=range(7),
                       prop_train_labels=0.8,
                       is_shuffle=False,
                       seed=0):
    """
    这里示例把 CLEAR10 做成 CGCD 形式，模仿 get_cub_datasets。
    假设:
      - old_classes=range(7) 代表初始 7 个旧类(0~6)
      - 其余的 3 个类(7,8,9) 依次在多个 Session 中引入
      - config_dict 里可能包含:
          config_dict['continual_session_num'] = 3
          config_dict['online_novel_unseen_num'] = 600  # 每次新类出现时抽多少(示例)
          config_dict['online_old_seen_num'] = 40       # 每次旧类抽多少(示例)
          config_dict['online_novel_seen_num'] = 40     # 对已见新类再出现时抽多少(示例)
    """
    # 读取重要字段
    continual_session_num = config_dict.get('continual_session_num', 3)
    online_novel_unseen_num = config_dict.get('online_novel_unseen_num', 600)
    online_old_seen_num     = config_dict.get('online_old_seen_num', 40)
    online_novel_seen_num   = config_dict.get('online_novel_seen_num', 40)

    # 1) 初始化整个训练集(含全部桶)
    whole_train = CustomCLEAR10(root=config_dict['clear_root'],
                                train=True,
                                transform=train_transform,
                                download=False)

    # 2) 抽取旧类: 先保留 old_classes
    old_dataset_all = subsample_classes(deepcopy(whole_train), include_classes=old_classes)

    # 类别内再拆分: 80%有标签 / 20%无标签
    from data.data_utils import subsample_instances
    each_old_all_samples = [
        subsample_classes(deepcopy(old_dataset_all), include_classes=[c])
        for c in old_classes
    ]
    each_old_labeled_slices = [
        subsample_instances(ds, prop_indices_to_subsample=prop_train_labels)
        for ds in each_old_all_samples
    ]
    each_old_unlabeled_slices = [
        np.array(list(set(range(len(ds.data))) - set(each_old_labeled_slices[i])))
        for i, ds in enumerate(each_old_all_samples)
    ]
    each_old_labeled_samples = [
        subsample_dataset(deepcopy(ds), each_old_labeled_slices[i])
        for i, ds in enumerate(each_old_all_samples)
    ]
    each_old_unlabeled_samples = [
        subsample_dataset(deepcopy(ds), each_old_unlabeled_slices[i])
        for i, ds in enumerate(each_old_all_samples)
    ]

    # 把所有旧类的 labeled 样本拼成 offline train
    offline_train_dataset = subDataset_wholeDataset(each_old_labeled_samples)

    # 对应 offline test dataset (假设只测旧类)
    whole_test = CustomCLEAR10(root=config_dict['clear_root'],
                               train=False,
                               transform=test_transform)
    offline_test_dataset = subsample_classes(deepcopy(whole_test), include_classes=old_classes)

    # 3) 为 online old classes 的无标签数据准备多个会话
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # 从 each_old_unlabeled_samples 里抽 offline_old_seen_num
        session_old_samples_list = []
        for ds in each_old_unlabeled_samples:
            idxs_ = np.random.choice(len(ds.data), size=online_old_seen_num, replace=False)
            session_old_samples_list.append(subsample_dataset(deepcopy(ds), idxs_))

        session_old_dataset = subDataset_wholeDataset(session_old_samples_list)
        online_old_dataset_unlabelled_list.append(session_old_dataset)

    # 4) 剩余类别(假设 range(10) 中去掉 0~6 => 7,8,9) 为 novel
    all_novel = set(range(10)) - set(old_classes)  # 7,8,9
    novel_dataset_unlabelled = subsample_classes(deepcopy(whole_train), include_classes=all_novel)

    novel_labels = np.unique(novel_dataset_unlabelled.data['target'])
    # 如果需要随机顺序
    if is_shuffle:
        np.random.seed(seed)
        np.random.shuffle(novel_labels)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []

    # 如果要平均分配 novel 到 sessions，
    # 这里做个简单示例: 3 个类 => 3 个 session, 每个 session 引入一个类
    # (若实际有更多类, 需做更灵活的分配)
    novel_labels = list(novel_labels)  # e.g. [7,8,9]
    if len(novel_labels) < continual_session_num:
        print("Warning: novel类数量少于session数，可能需要调整策略")

    # session i 要引入 novel_labels[i]
    for s in range(continual_session_num):
        current_novel = novel_labels[s] if s < len(novel_labels) else None
        # 构建 session s 的 novel 样本
        if current_novel is not None:
            ds_novel_single = subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[current_novel])
            # 如果是第一次出现 => unseen_num
            # 若后面再次出现 => seen_num
            if s == 0:
                chosen_idx = np.random.choice(len(ds_novel_single.data), online_novel_unseen_num, replace=False)
            else:
                chosen_idx = np.random.choice(len(ds_novel_single.data), online_novel_seen_num, replace=False)
            ds_novel_single = subsample_dataset(ds_novel_single, chosen_idx)
        else:
            # 没有更多新类可以引入
            ds_novel_single = None

        # 拼成一个 dataset
        if ds_novel_single is not None:
            session_novel_dataset = ds_novel_single
        else:
            # 如果没有新类，则返回空 dataset
            session_novel_dataset = None

        # 收集
        online_novel_dataset_unlabelled_list.append(session_novel_dataset)

        # session 的测试集: 旧类 + 目前出现过的新类
        # 目前出现过的新类 => novel_labels[:s+1]
        seen_novel_so_far = novel_labels[:s+1]
        session_test_classes = list(old_classes) + list(seen_novel_so_far)
        session_test = subsample_classes(deepcopy(whole_test), include_classes=session_test_classes)
        online_test_dataset_list.append(session_test)

    # 最终打包
    all_datasets = {
        'offline_train_dataset': offline_train_dataset,
        'offline_test_dataset': offline_test_dataset,
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,
        'online_test_dataset_list': online_test_dataset_list,
    }

    return all_datasets