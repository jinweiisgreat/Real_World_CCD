#!/usr/bin/env python3
"""
独立生成Prompt Pool的脚本
从训练数据中创建prompt pool，可以单独运行

Usage:
    python generate_prompt_pool.py --dataset cifar10 --data_root ./data --output_dir ./prompt_pools
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目路径到sys.path (根据你的项目结构调整)
sys.path.append('.')
sys.path.append('..')

# 导入必要的模块
from models.utils_prompt_pool_trainable import LearnablePromptPool
from models.prompt_enhanced_model_trainable import PromptEnhancedModel
from models.utils_simgcd import DINOHead
from models import vision_transformer as vits
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets
from config import dino_pretrain_path, exp_root


def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate Prompt Pool')

    # 数据集相关
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet_100'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root')
    parser.add_argument('--transform', type=str, default='imagenet',
                        help='Image transform type')

    # 模型相关
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model. If None, use DINO backbone only')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--feat_dim', type=int, default=768,
                        help='Feature dimension')
    parser.add_argument('--num_mlp_layers', type=int, default=3,
                        help='Number of MLP layers in projector')

    # Prompt Pool相关
    parser.add_argument('--max_prompts', type=int, default=200,
                        help='Maximum number of prompts in the pool')
    parser.add_argument('--similarity_threshold', type=float, default=0.65,
                        help='Similarity threshold for community detection')
    parser.add_argument('--community_ratio', type=float, default=1.2,
                        help='Community ratio parameter')

    # 输出相关
    parser.add_argument('--output_dir', type=str, default='./prompt_pools',
                        help='Output directory for prompt pool')
    parser.add_argument('--save_prefix', type=str, default='prompt_pool',
                        help='Prefix for saved files')

    # 训练相关
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # 类别分割相关 (从原代码复制)
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='Proportion of training labels')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True,
                        help='Use standard splits')

    return parser.parse_args()


def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_model(args, device, logger):
    """加载模型"""
    # 加载backbone
    backbone = vits.__dict__['vit_base']()
    logger.info(f'Loading DINO weights from {dino_pretrain_path}')
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # 冻结backbone参数
    for param in backbone.parameters():
        param.requires_grad = False

    backbone.to(device)

    # 如果有预训练模型路径，加载完整模型
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f'Loading pre-trained model from {args.model_path}')

        # 创建projector
        mlp_out_dim = args.num_labeled_classes  # 这个需要从数据集获得
        projector = DINOHead(in_dim=args.feat_dim, out_dim=mlp_out_dim, nlayers=args.num_mlp_layers)

        # 创建PromptEnhancedModel (不带prompt pool)
        model = PromptEnhancedModel(
            backbone=backbone,
            projector=projector,
            prompt_pool=None,  # 先不加载prompt pool
            enable_prompt_training=False
        )

        # 加载权重
        checkpoint = torch.load(args.model_path, map_location='cpu')

        # 只加载backbone和projector的权重，忽略prompt pool部分
        model_state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith('prompt_pool.'):
                model_state_dict[key] = value

        model.load_state_dict(model_state_dict, strict=False)
        model.to(device)
        logger.info('Pre-trained model loaded successfully')
    else:
        # 只使用DINO backbone
        logger.info('Using DINO backbone only for feature extraction')
        model = backbone

    return model


def load_dataset(args, logger):
    """加载数据集"""
    # 获取类别分割
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'Labeled classes: {args.num_labeled_classes}')
    logger.info(f'Unlabeled classes: {args.num_unlabeled_classes}')

    # 获取数据变换
    _, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)

    # 获取数据集
    datasets = get_datasets(args.dataset, test_transform, train_classes=args.train_classes,
                            unlabeled_classes=args.unlabeled_classes, args=args)

    # 创建训练数据加载器 (用于特征提取)
    train_dataset = datasets['train_labelled']
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f'Training samples: {len(train_dataset)}')

    return train_loader, args


def extract_features(model, data_loader, device, logger):
    """提取特征"""
    logger.info("Extracting features from training data...")

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
            try:
                # 处理不同的batch格式
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch[0], batch[1]

                images = images.to(device)

                # 提取特征
                if hasattr(model, 'backbone'):
                    features = model.backbone(images)
                else:
                    features = model(images)

                # 归一化特征
                features = torch.nn.functional.normalize(features, dim=-1)

                all_features.append(features.cpu())
                all_labels.append(labels)

            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue

    if not all_features:
        raise ValueError("No features extracted! Check your data loader and model.")

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logger.info(f"Extracted {len(all_features)} feature vectors of dimension {all_features.shape[1]}")

    return all_features, all_labels


def create_prompt_pool(args, device, logger):
    """创建Prompt Pool"""
    logger.info("Creating Prompt Pool...")

    # 初始化prompt pool
    prompt_pool = LearnablePromptPool(
        feature_dim=args.feat_dim,
        similarity_threshold=args.similarity_threshold,
        community_ratio=args.community_ratio,
        device=device,
        max_prompts=args.max_prompts
    )

    logger.info(f"Prompt Pool initialized with max_prompts={args.max_prompts}")
    return prompt_pool


def generate_prompt_pool_from_features(prompt_pool, features, labels, num_classes, logger):
    """从特征生成prompt pool"""
    logger.info("Generating prompt pool from extracted features...")

    # 创建一个简单的特征提取器mock对象
    class FeatureExtractorMock:
        def __init__(self, features, labels):
            self.features = features.numpy()
            self.labels = labels.numpy()

        def extract_features(self, *args, **kwargs):
            return self.features, self.labels

    # 转换为numpy格式
    features_np = features.numpy()
    labels_np = labels.numpy()

    # 调用prompt pool的创建方法
    try:
        # 使用社区检测方法创建prompt pool
        result = prompt_pool.create_from_features(
            all_features=features_np,
            all_labels=labels_np,
            num_classes=num_classes,
            logger=logger
        )

        logger.info("Prompt pool created successfully using community detection")

    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        logger.info("Falling back to K-means initialization...")

        # 备用方案：使用K-means
        result = prompt_pool._fallback_kmeans_initialization(
            all_features=features_np,
            num_classes=num_classes,
            logger=logger
        )

    return result


def save_prompt_pool(prompt_pool, result, args, logger):
    """保存prompt pool"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存prompt pool
    prompt_pool_path = output_dir / f"{args.save_prefix}_{args.dataset}.pt"
    prompt_pool.save_prompt_pool(str(prompt_pool_path))
    logger.info(f"Prompt pool saved to: {prompt_pool_path}")

    # 保存统计信息
    stats_path = output_dir / f"{args.save_prefix}_{args.dataset}_stats.pt"
    torch.save({
        'creation_result': result,
        'dataset': args.dataset,
        'num_labeled_classes': args.num_labeled_classes,
        'max_prompts': args.max_prompts,
        'similarity_threshold': args.similarity_threshold,
        'community_ratio': args.community_ratio,
        'feature_dim': args.feat_dim,
    }, str(stats_path))
    logger.info(f"Statistics saved to: {stats_path}")

    # 保存配置文件
    config_path = output_dir / f"{args.save_prefix}_{args.dataset}_config.txt"
    with open(config_path, 'w') as f:
        f.write("Prompt Pool Generation Configuration\n")
        f.write("=" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\nGeneration Results:\n")
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Configuration saved to: {config_path}")

    return prompt_pool_path, stats_path, config_path


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    device = setup_device()

    # 设置日志
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'prompt_pool_generation_{args.dataset}.log')
    logger = setup_logger('PromptPoolGenerator', log_file)

    logger.info("Starting Prompt Pool generation...")
    logger.info(f"Arguments: {args}")

    try:
        # 1. 加载数据集
        data_loader, args = load_dataset(args, logger)

        # 2. 加载模型
        model = load_model(args, device, logger)

        # 3. 提取特征
        features, labels = extract_features(model, data_loader, device, logger)

        # 4. 创建prompt pool
        prompt_pool = create_prompt_pool(args, device, logger)

        # 5. 生成prompt pool
        result = generate_prompt_pool_from_features(
            prompt_pool, features, labels, args.num_labeled_classes, logger
        )

        # 6. 保存结果
        prompt_pool_path, stats_path, config_path = save_prompt_pool(prompt_pool, result, args, logger)

        # 7. 输出摘要
        logger.info("\n" + "=" * 50)
        logger.info("Prompt Pool Generation Completed!")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Number of prompts created: {result.get('num_prompts', 'Unknown')}")
        logger.info(f"Method used: {result.get('method', 'Unknown')}")
        if 'avg_purity' in result:
            logger.info(f"Average community purity: {result['avg_purity']:.4f}")
        logger.info(f"Prompt pool saved to: {prompt_pool_path}")
        logger.info(f"Statistics saved to: {stats_path}")
        logger.info("=" * 50)

        print(f"\n✓ Success! Prompt pool saved to: {prompt_pool_path}")

    except Exception as e:
        logger.error(f"Error during prompt pool generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()