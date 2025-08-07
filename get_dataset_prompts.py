#!/usr/bin/env python3
"""
专门用于Prompts Pool创建的数据集加载器
加载完整的训练数据集，无需复杂的连续学习划分
"""

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from data.tiny_imagenet import get_tiny_imagenet_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.stanford_cars import get_scars_datasets
from config import cifar_10_root, cifar_100_root


class SimpleDatasetWrapper(Dataset):
    """简单的数据集包装器，兼容现有的prompts pool创建流程"""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.uq_idxs = np.array(range(len(base_dataset)))

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx

    def __len__(self):
        return len(self.base_dataset)


def get_full_dataset_for_prompts(dataset_name, transform):
    """
    获取完整的训练数据集用于prompts pool创建

    Args:
        dataset_name: 数据集名称
        transform: 数据变换

    Returns:
        dataset: 包装后的数据集
    """

    if dataset_name == 'cifar10':
        base_dataset = CIFAR10(root=cifar_10_root, train=True, transform=transform, download=True)

    elif dataset_name == 'cifar100':
        base_dataset = CIFAR100(root=cifar_100_root, train=True, transform=transform, download=True)

    elif dataset_name == 'tiny_imagenet':
        # 对于其他数据集，需要稍微适配一下
        # 这里简化处理，你可以根据需要扩展
        raise NotImplementedError("TinyImageNet需要单独实现")

    elif dataset_name == 'cub':
        raise NotImplementedError("CUB需要单独实现")

    elif dataset_name == 'aircraft':
        raise NotImplementedError("Aircraft需要单独实现")

    elif dataset_name == 'scars':
        raise NotImplementedError("SCARS需要单独实现")

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 包装数据集使其兼容现有代码
    wrapped_dataset = SimpleDatasetWrapper(base_dataset)

    print(f"加载完整 {dataset_name} 训练数据集:")
    print(f"  - 样本数量: {len(wrapped_dataset)}")
    print(f"  - 类别数量: {len(set(base_dataset.targets))}")
    print(f"  - 类别范围: {min(base_dataset.targets)} - {max(base_dataset.targets)}")

    return wrapped_dataset


# 示例使用
if __name__ == "__main__":
    import torchvision.transforms as transforms

    # 测试变换
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载CIFAR-100完整数据集
    dataset = get_full_dataset_for_prompts('cifar100', test_transform)

    # 测试数据加载
    img, label, uq_idx = dataset[0]
    print(f"样本形状: {img.shape}, 标签: {label}, 索引: {uq_idx}")

    print("数据集加载成功！")