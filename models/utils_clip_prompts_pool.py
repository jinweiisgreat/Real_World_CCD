#!/usr/bin/env python3
"""
独立创建CLIP社区Prompts池的脚本
使用CLIP ViT分支提取特征，通过Louvain社区发现算法构建prompts池
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
from transformers import CLIPModel, CLIPProcessor
from collections import defaultdict
import pickle
from PIL import Image
import time
from datetime import datetime

# 添加项目路径以便导入现有模块
import sys

sys.path.append('.')

from data.get_datasets import get_class_splits, get_datasets
from data.augmentations import get_transform
from torch.utils.data import DataLoader




class CLIPPromptsPoolCreator:
    """
    CLIP社区Prompts池创建器
    """

    def __init__(self, clip_model_path, similarity_threshold=0.7, device='cuda'):
        """
        初始化

        Args:
            clip_model_path: CLIP模型路径
            similarity_threshold: 构建图时的相似度阈值
            device: 计算设备
        """
        self.device = device
        self.similarity_threshold = similarity_threshold

        print(f"Loading CLIP model from {clip_model_path}...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.clip_model.to(device)
        self.clip_model.eval()

        # 获取CLIP的特征维度
        self.feature_dim = self.clip_model.config.vision_config.hidden_size
        print(f"CLIP feature dimension: {self.feature_dim}")

        print("CLIP model loaded successfully!")

    def extract_clip_features(self, dataloader, max_samples=None):
        """
        使用CLIP ViT分支提取图像特征

        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数量（用于调试）

        Returns:
            features: 提取的特征 [N, feature_dim]
            labels: 对应的标签 [N]
            indices: 样本索引 [N]
        """
        print("Extracting CLIP features...")

        all_features = []
        all_labels = []
        all_indices = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting CLIP features")):
                # 处理不同格式的batch
                if len(batch) >= 3:
                    images, labels, indices = batch[:3]
                else:
                    images, labels = batch
                    indices = torch.arange(batch_idx * dataloader.batch_size,
                                           batch_idx * dataloader.batch_size + len(labels))


                images = images.to(self.device)
                pixels_values = images

                inputs = self.clip_processor(images=images_pil, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(self.device)

                # 使用CLIP视觉编码器提取特征
                vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)

                # 取pooled output或CLS token特征
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    features = vision_outputs.pooler_output
                else:
                    # 如果没有pooler_output，使用CLS token (第一个token)
                    features = vision_outputs.last_hidden_state[:, 0, :]

                # L2归一化
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                all_features.append(features.cpu())
                all_labels.append(labels)
                all_indices.append(indices)

                sample_count += len(labels)

                if max_samples and sample_count >= max_samples:
                    break




        if not all_features:
            raise RuntimeError("No features extracted! Check your data loader and CLIP model.")

        # 合并所有特征和标签
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()

        print(f"Successfully extracted {len(features)} CLIP features with shape {features.shape}")

        return features, labels, indices

    def build_similarity_graph(self, features):
        """
        基于特征相似度构建图

        Args:
            features: 特征矩阵 [N, feature_dim]

        Returns:
            G: NetworkX图对象
        """
        print(f"Building similarity graph with threshold {self.similarity_threshold}...")

        # 计算余弦相似度矩阵
        print("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(features)
        n_samples = len(features)

        # 创建NetworkX图
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))

        # 添加边（只保留相似度高于阈值的边）
        print("Adding edges to graph...")
        edge_count = 0
        for i in tqdm(range(n_samples), desc="Building graph"):
            for j in range(i + 1, n_samples):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    edge_count += 1

        print(f"Graph built with {n_samples} nodes and {edge_count} edges")

        # 检查图的连通性
        num_components = nx.number_connected_components(G)
        print(f"Graph has {num_components} connected components")

        return G

    def discover_communities(self, G):
        """
        使用Louvain算法进行社区发现

        Args:
            G: NetworkX图对象

        Returns:
            communities: 社区分配字典
        """
        print("Discovering communities using Louvain algorithm...")

        # 使用Louvain算法进行社区发现
        communities = community_louvain.best_partition(G, random_state=42)

        # 统计社区信息
        community_sizes = defaultdict(int)
        for node, community_id in communities.items():
            community_sizes[community_id] += 1

        num_communities = len(community_sizes)
        print(f"Discovered {num_communities} communities")

        # 显示社区大小分布
        sizes = list(community_sizes.values())
        print(f"Community size statistics:")
        print(f"  Min size: {min(sizes)}")
        print(f"  Max size: {max(sizes)}")
        print(f"  Mean size: {np.mean(sizes):.2f}")
        print(f"  Std size: {np.std(sizes):.2f}")

        return communities

    def compute_community_centroids(self, features, labels, communities):
        """
        计算每个社区的中心点作为prompts

        Args:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            communities: 社区分配字典

        Returns:
            prompts_pool: prompts池 [num_communities, feature_dim]
            community_info: 社区信息列表
        """
        print("Computing community centroids as prompts...")

        # 按社区分组
        community_groups = defaultdict(list)
        for node, community_id in communities.items():
            community_groups[community_id].append(node)

        prompts_pool = []
        community_info = []

        for community_id, nodes in tqdm(community_groups.items(), desc="Computing centroids"):
            # 获取该社区的所有特征
            community_features = features[nodes]
            community_labels = labels[nodes]

            # 计算中心点（均值）
            centroid = np.mean(community_features, axis=0)

            # L2归一化
            centroid = centroid / np.linalg.norm(centroid)

            prompts_pool.append(centroid)

            # 计算标签分布
            unique_labels, counts = np.unique(community_labels, return_counts=True)
            label_distribution = dict(zip(unique_labels, counts))

            # 记录社区信息
            info = {
                'community_id': community_id,
                'size': len(nodes),
                'centroid': centroid,
                'member_indices': nodes,
                'member_labels': community_labels.tolist(),
                'label_distribution': label_distribution,
                'dominant_label': unique_labels[np.argmax(counts)],
                'label_purity': np.max(counts) / len(nodes)  # 标签纯度
            }
            community_info.append(info)

        prompts_pool = np.array(prompts_pool)

        print(f"Generated prompts pool with {len(prompts_pool)} prompts")

        return prompts_pool, community_info

    def create_prompts_pool(self, dataloader, max_samples=None, save_dir=None):
        """
        完整的prompts池创建流程

        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数量
            save_dir: 保存目录

        Returns:
            prompts_pool: 构建好的prompts池
            community_info: 社区信息
        """
        print("=" * 20)
        print("Creating CLIP Community Prompts Pool")
        print("=" * 20)

        start_time = time.time()

        # 1. 提取CLIP特征
        features, labels, indices = self.extract_clip_features(dataloader, max_samples)

        # 2. 构建相似度图
        G = self.build_similarity_graph(features)

        # 3. 社区发现
        communities = self.discover_communities(G)

        # 4. 计算社区中心点作为prompts
        prompts_pool, community_info = self.compute_community_centroids(features, labels, communities)

        # 5. 保存结果
        if save_dir:
            vis_path = os.path.join(save_dir, "prompts_pool.png")
            self.visualize_graph_network(G, communities, vis_path)
            self.save_results(save_dir, prompts_pool, community_info, features, labels, indices, communities)

        elapsed_time = time.time() - start_time
        print("=" * 20)
        print(f"Prompts pool creation completed in {elapsed_time:.2f} seconds!")
        print(f"Created {len(prompts_pool)} prompts with dimension {prompts_pool.shape[1]}")
        print("=" * 20)

        return prompts_pool, community_info

    def save_results(self, save_dir, prompts_pool, community_info, features, labels, indices, communities):
        """
        保存所有结果

        Args:
            save_dir: 保存目录
            prompts_pool: prompts池
            community_info: 社区信息
            features: 原始特征
            labels: 标签
            indices: 索引
            communities: 社区分配
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存主要结果
        main_results = {
            'prompts_pool': prompts_pool,
            'community_info': community_info,
            'num_communities': len(community_info),
            'feature_dim': prompts_pool.shape[1],
            'similarity_threshold': self.similarity_threshold,
            'creation_time': datetime.now().isoformat()
        }

        main_save_path = os.path.join(save_dir, 'prompts_pool.pkl')
        with open(main_save_path, 'wb') as f:
            pickle.dump(main_results, f)
        print(f"Main results saved to {main_save_path}")

        # 保存详细数据（可选，用于后续分析）
        detailed_results = {
            'features': features,
            'labels': labels,
            'indices': indices,
            'communities': communities,
            'similarity_threshold': self.similarity_threshold
        }

        detailed_save_path = os.path.join(save_dir, 'detailed_data.pkl')
        with open(detailed_save_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        print(f"Detailed data saved to {detailed_save_path}")

        # 保存社区统计信息为文本文件
        stats_path = os.path.join(save_dir, 'community_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("CLIP Community Prompts Pool Statistics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total samples: {len(features)}\n")
            f.write(f"Feature dimension: {features.shape[1]}\n")
            f.write(f"Number of communities: {len(community_info)}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"Creation time: {datetime.now().isoformat()}\n\n")

            f.write("Community Details:\n")
            f.write("-" * 30 + "\n")
            for i, info in enumerate(community_info):
                f.write(f"Community {i} (ID: {info['community_id']}):\n")
                f.write(f"  Size: {info['size']}\n")
                f.write(f"  Dominant label: {info['dominant_label']}\n")
                f.write(f"  Label purity: {info['label_purity']:.3f}\n")
                f.write(f"  Label distribution: {info['label_distribution']}\n")
                f.write("\n")

        print(f"Statistics saved to {stats_path}")

    def visualize_graph_network(self, G, save_path, max_nodes=5000):
        """
        可视化相似度图网络

        Args:
            G: NetworkX图对象
            save_path: 保存路径
            max_nodes: 最大显示节点数
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print(f"Visualizing graph network, max {max_nodes} nodes...")

            # 设置样式
            sns.set(style="whitegrid", context="paper", font_scale=1.2)

            # 如果图太大，采样节点
            n = G.number_of_nodes()
            if n > max_nodes:
                nodes_to_sample = np.random.choice(list(G.nodes()), max_nodes, replace=False)
                G_sampled = G.subgraph(nodes_to_sample).copy()
                print(f"Sampled {max_nodes} nodes from {n} total nodes")
            else:
                G_sampled = G
                print(f"Using all {n} nodes")

            # 计算图统计信息
            if G_sampled.number_of_nodes() > 0:
                avg_degree = sum(dict(G_sampled.degree()).values()) / G_sampled.number_of_nodes()
                density = nx.density(G_sampled)
            else:
                avg_degree = 0
                density = 0
                print("Warning: No nodes in graph")
                return

            # 创建可视化
            plt.figure(figsize=(18, 16))

            # 计算布局
            pos = nx.spring_layout(
                G_sampled,
                k=0.15,
                iterations=50,
                seed=42
            )

            # 根据度确定节点大小
            degrees = dict(G_sampled.degree())
            size_scale = max(1, 2000 / np.sqrt(len(G_sampled)))
            node_sizes = [1 + 0.8 * np.sqrt(degrees[n]) * size_scale for n in G_sampled.nodes()]

            # 根据边数调整透明度
            edge_alpha = max(0.05, min(0.2, 20000 / G_sampled.number_of_edges()))

            # 绘制边
            nx.draw_networkx_edges(
                G_sampled, pos,
                width=1.0,
                alpha=edge_alpha,
                edge_color="gray"
            )

            # 绘制节点
            nodes = nx.draw_networkx_nodes(
                G_sampled, pos,
                node_size=node_sizes,
                node_color=[degrees[n] for n in G_sampled.nodes()],
                cmap=plt.cm.viridis,
                alpha=0.7
            )

            # 添加颜色条
            plt.colorbar(nodes, label="Node Degree", shrink=0.6)

            # 添加标题
            plt.title(
                f"Similarity Network\n"
                f"Nodes: {G_sampled.number_of_nodes()}, Edges: {G_sampled.number_of_edges()}\n"
                f"Avg Degree: {avg_degree:.2f}, Density: {density:.4f}",
                fontsize=14
            )

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Graph visualization saved to {save_path}")

        except Exception as e:
            print(f"Graph visualization failed: {str(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CLIP Community Prompts Pool')

    # 数据集相关参数
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Dataset name: cifar10, cifar100, tiny_imagenet, etc.')
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='Proportion of training labels to use')

    # CLIP模型参数
    parser.add_argument('--clip_model_path', type=str, default="/home/ps/_jinwei/CLIP_L14",
                        help='Path to CLIP model')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Similarity threshold for building graph')

    # 处理参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for debugging)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='/home/ps/_jinwei/Happy-CGCD/prompts_pools',
                        help='Directory to save prompts pool')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for saving')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 获取数据集配置
    args = get_class_splits(args)

    # 设置实验名称
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.dataset_name}_thresh{args.similarity_threshold}_{timestamp}"

    save_dir = os.path.join(args.save_dir, args.exp_name)

    print(f"Starting prompts pool creation for {args.dataset_name}")
    print(f"Using {len(args.train_classes)} old classes: {args.train_classes}")
    print(f"Results will be saved to: {save_dir}")

    # 准备数据
    from data.augmentations import get_transform

    # 使用测试变换以获得干净的特征
    _, test_transform = get_transform('imagenet', image_size=224, args=args)

    # 构建配置字典（简化版，只用于离线数据）
    dataset_split_config_dict = {
        'continual_session_num': 1,  # 只使用离线数据
        'online_novel_unseen_num': 0,
        'online_old_seen_num': 0,
        'online_novel_seen_num': 0,
    }

    # 临时设置相关参数
    args.continual_session_num = 1
    args.online_novel_unseen_num = 0
    args.online_old_seen_num = 0
    args.online_novel_seen_num = 0
    args.shuffle_classes = False

    # 获取数据集
    offline_train_dataset, _, _, _, _, _, _ = get_datasets(
        args.dataset_name, test_transform, test_transform, args)

    # 创建数据加载器
    dataloader = DataLoader(
        offline_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Loaded dataset with {len(offline_train_dataset)} samples")

    # 创建prompts池
    creator = CLIPPromptsPoolCreator(
        clip_model_path=args.clip_model_path,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )

    # 执行创建过程
    prompts_pool, community_info = creator.create_prompts_pool(
        dataloader=dataloader,
        max_samples=args.max_samples,
        save_dir=save_dir
    )

    print(f"\nPrompts pool creation completed successfully!")
    print(f"Final prompts pool shape: {prompts_pool.shape}")
    print(f"Results saved to: {save_dir}")