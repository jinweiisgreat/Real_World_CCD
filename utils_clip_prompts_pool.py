#!/usr/bin/env python3
"""
独立创建CLIP社区Prompts池的脚本
使用CLIP ViT分支提取特征，通过Louvain社区发现算法构建prompts池
"""

import argparse
import os
import torch
import numpy as np
from networkx.algorithms.bipartite.covering import min_edge_cover
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
from transformers import CLIPModel, CLIPProcessor
from transformers import Dinov2Model, Dinov2Config
from collections import defaultdict
import pickle
from PIL import Image
import time
from datetime import datetime
import torchvision.transforms as transforms


# 添加项目路径以便导入现有模块
import sys

sys.path.append('models')

from data.get_datasets import get_class_splits, get_datasets
from data.augmentations import get_transform
from torch.utils.data import DataLoader




class VisualPromptsPoolCreator:

    def __init__(self, model_path, model_type='clip', similarity_threshold=0.7, device='cuda', dataset_name=None):
        """
        初始化

        Args:
            model_path: 模型路径
            model_type: 模型类型 ('clip' 或 'dino')
            similarity_threshold: 构建图时的相似度阈值
            device: 计算设备
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.model_type = model_type.lower()
        self.dataset_name = dataset_name

        print(f"Loading {model_type.upper()} model from {model_path}...")

        if self.model_type == 'clip':
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.feature_dim = self.model.config.vision_config.hidden_size
            print(f"CLIP feature dimension: {self.feature_dim}")
        elif self.model_type == 'dino':
            self.model = Dinov2Model.from_pretrained(model_path)
            # DINOv2没有专用的processor，使用简单的预处理
            self.processor = None
            self.feature_dim = self.model.config.hidden_size
            print(f"DINOv2 feature dimension: {self.feature_dim}")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}, 请选择 'clip' 或 'dino'")

        self.model.to(device)
        self.model.eval()
        print(f"{model_type.upper()} model loaded successfully!")

    def extract_features(self, dataloader, max_samples=None):
        """
        提取图像特征

        Args:
            dataloader: 数据加载器
            max_samples: 最大样本数量（用于调试）

        Returns:
            features: 提取的特征 [N, feature_dim]
            labels: 对应的标签 [N]
            indices: 样本索引 [N]
        """
        print(f"Extracting {self.model_type.upper()} features...")

        all_features = []
        all_labels = []
        all_indices = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {self.model_type.upper()} features")):
                # 处理不同格式的batch
                if len(batch) >= 3:
                    images, labels, indices = batch[:3]
                else:
                    images, labels = batch
                    indices = torch.arange(batch_idx * dataloader.batch_size,
                                           batch_idx * dataloader.batch_size + len(labels))

                images = images.to(self.device)

                # 根据不同模型类型提取特征
                if self.model_type == 'clip':
                    # 使用CLIP处理器
                    inputs = self.processor(images=images, return_tensors="pt", padding=True, do_rescale=False)
                    pixel_values = inputs['pixel_values'].to(self.device)

                    # 使用CLIP视觉编码器提取特征
                    vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                    # features = vision_outputs.pooler_output
                    features = vision_outputs.last_hidden_state[:, 0]
                elif self.model_type == 'dino':
                    # DINOv2直接接收归一化的图像张量
                    # 如果图像已经被处理为[0,1]范围，需要调整到[-1,1]或其他DINOv2期望的范围
                    if images.max() <= 1.0:
                        # 标准化到DINOv2期望的输入范围
                        if self.dataset_name == 'imagenet':
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                            images = (images - mean) / std
                        else:
                            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(self.device)
                            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(self.device)
                            print("Using default CIFAR normalization for DINOv2")
                            images = (images - mean) / std

                    # 使用DINOv2提取特征
                    outputs = self.model(pixel_values=images)
                    features = outputs.pooler_output
                    # features = outputs.last_hidden_state[:, 0]

                # L2归一化
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                all_features.append(features.cpu())
                all_labels.append(labels)
                all_indices.append(indices)

                sample_count += len(labels)

        if not all_features:
            raise RuntimeError(f"No features extracted! Check your data loader and {self.model_type.upper()} model.")

        # 合并所有特征和标签
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()

        print(f"Successfully extracted {len(features)} {self.model_type.upper()} features with shape {features.shape}")

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
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(np.int8)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
        print("Performing community detection...")
        G = nx.from_numpy_array(adjacency_matrix)

        # 检查图的连通性
        num_components = nx.number_connected_components(G)
        print(f"Graph has {num_components} connected components")

        return G, adjacency_matrix

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
        partition = community_louvain.best_partition(G, resolution=1.0)

        # Stats community sizes
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        print(f"Found {len(communities)} communities")

        # Filter out small communities
        min_community_size = 5
        valid_communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}
        print(f"Found {len(valid_communities)} valid communities")

        target_num_communities = 120 # num_claseses * 1.6
        if len(valid_communities) > target_num_communities:
            community_sizes = [(k, len(v)) for k, v in valid_communities.items()]
            community_sizes.sort(key=lambda x: x[1], reverse=True)
            valid_community_ids = [k for k, _ in community_sizes[:target_num_communities]]
            valid_communities = {k: valid_communities[k] for k in valid_community_ids}
            print(f"Found {len(valid_communities)} valid communities")

        # 显示社区大小分布
        min_size = min(len(v) for v in valid_communities.values())
        max_size = max(len(v) for v in valid_communities.values())
        print(f"Community size statistics:")
        print(f"  Min size: {min_size}")
        print(f"  Max size: {max_size}")

        return valid_communities

    def compute_community_centroids(self, features, labels, communities):
        """
        计算每个社区的中心点作为prompts

        Args:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            communities: 社区分配字典，格式为{community_id: [node_indices]}

        Returns:
            prompts_pool: prompts池 [num_communities, feature_dim]
            community_info: 社区信息列表
        """
        print("Computing community centroids as prompts...")

        prompts_pool = []
        community_info = []

        for community_id, nodes in tqdm(communities.items(), desc="Computing centroids"):
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

        print(f"Generated Prepare with {len(prompts_pool)} prompts")

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

        # 1. 提取特征
        features, labels, indices = self.extract_features(dataloader, max_samples)

        # 2. 可视化特征空间
        feature_vis_path = os.path.join(save_dir, "feature_space.png")
        self.visualize_feature_space(features, labels, feature_vis_path)

        # 2. 构建相似度图
        G, adj = self.build_similarity_graph(features)
        vis_path = os.path.join(save_dir, "prompts_pool.png")
        self.visualize_graph_network(adj, vis_path)

        # 3. 社区发现
        communities = self.discover_communities(G) # communities = valid_communities
        print("Communities discovered successfully!")

        # 4. 计算社区中心点作为prompts
        prompts_pool, community_info = self.compute_community_centroids(features, labels, communities)

        # 5. 保存结果
        if save_dir:
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

    def visualize_graph_network(self, adjacency_matrix, save_path, max_nodes=10000):
        """
        简化版的图网络可视化函数，只显示主网络图。

        Args:
            adjacency_matrix: 图的邻接矩阵
            save_path: 保存可视化图的路径
            max_nodes: 可视化的最大节点数
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置seaborn样式提高美观度
        sns.set(style="whitegrid", context="paper", font_scale=1.2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 如果图太大，采样节点
        n = adjacency_matrix.shape[0]
        if n > max_nodes:
            indices = np.random.choice(n, max_nodes, replace=False)
            sampled_adj_matrix = adjacency_matrix[indices][:, indices]

        else:
            sampled_adj_matrix = adjacency_matrix
            indices = np.arange(n)

        # 从邻接矩阵创建图
        G = nx.from_numpy_array(sampled_adj_matrix)

        # 过滤掉度数小的节点
        min_degree = 5
        degrees = dict(G.degree())
        nodes_to_keep = [node for node, degree in degrees.items() if degree >= min_degree]
        G = G.subgraph(nodes_to_keep).copy()

        # 计算图统计信息
        if G.number_of_nodes() > 0:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            density = nx.density(G)
        else:
            avg_degree = 0
            density = 0
            return

        # 设置主网络可视化
        plt.figure(figsize=(18, 16))

        # 计算节点位置 - 使用spring布局获得更好的分布
        pos = nx.spring_layout(
            G,
            k=0.15,  # 节点之间的最佳距离（越小越紧凑）
            iterations=50,  # 更多迭代次数以获得更好的布局
            seed=42
        )

        # 根据度确定节点大小
        degrees = dict(G.degree())

        # 根据节点数量自动缩放节点大小，避免过度拥挤
        size_scale = max(1, 2000 / np.sqrt(len(G)))
        node_sizes = [1 + 0.8 * np.sqrt(degrees[n]) * size_scale for n in G.nodes()]

        # 根据边数自动调整透明度
        edge_alpha = max(0.05, min(0.2, 20000 / G.number_of_edges()))

        # 绘制边
        edges = nx.draw_networkx_edges(
            G, pos,
            width=0.5,
            alpha=edge_alpha,
            # alpha=0.1,
            edge_color="gray"
        )

        # 使用基于度的颜色映射绘制节点
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            # node_size = 50,
            node_color=[degrees[n] for n in G.nodes()],
            # node_color='lightblue',
            cmap=plt.cm.viridis,
            alpha=0.5
        )

        # 添加颜色条显示节点度
        plt.colorbar(nodes, label="nodes degree", shrink=0.6)

        # 添加标题和图统计信息
        plt.title(
            f"Similarity Network\n"
            f"Node: {G.number_of_nodes()}, Edge: {G.number_of_edges()}\n"
            f"Avg degree: {avg_degree:.2f}, density: {density:.4f}",
            fontsize=14
        )

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_feature_space(self, features, labels, save_path):
        """
        用t-SNE可视化特征空间分布

        Args:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            print("Visualizing feature space with t-SNE...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
            features_2d = tsne.fit_transform(features)

            # 创建可视化
            plt.figure(figsize=(12, 10))

            # 按标签着色
            unique_labels = np.unique(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=[colors[i]], alpha=0.6, s=20, label=f'Class {label}')

            plt.title('Feature Space Distribution (t-SNE)', fontsize=14)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Feature space visualization saved to {save_path}")

        except Exception as e:
            print(f"Feature space visualization failed: {str(e)}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CLIP Community Prompts Pool')

    # 数据集相关参数
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Dataset name: cifar10, cifar100, tiny_imagenet, etc.')
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='Proportion of training labels to use')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='dino', choices=['clip', 'dino'],
                        help='Model type to use: clip or dino')
    # /home/ps/_jinwei/CLIP_L14
    # /home/ps/_jinwei/DINO_v2_base
    parser.add_argument('--model_path', type=str, default="/home/ps/_jinwei/DINO_v2_base",
                        help='Path to model')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Similarity threshold for building graph')

    # 处理参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for debugging)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./prompts_pools',
                        help='Directory to save Prepare')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for saving')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--num_old_classes', type=int, default=100,
                       help='Number of old classes')
    parser.add_argument('--use_ssb_splits', action='store_true', default=False,
                       help='Use SSB splits for FGVC datasets')

    parser.add_argument('--interpolation', type=int, default=3,
                        help='Interpolation method')
    parser.add_argument('--crop_pct', type=float, default=0.875,
                        help='Crop percentage')


    args = parser.parse_args()

    from get_dataset_prompts import get_full_dataset_for_prompts

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 获取数据集配置
    args = get_class_splits(args)
    args.exp_name

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.dataset_name}_thresh{args.similarity_threshold}_{timestamp}"

    save_dir = os.path.join(args.save_dir, args.exp_name)

    print(f"Starting Prepare creation for {args.dataset_name}")
    print(f"Using {len(args.train_classes)} old classes: {args.train_classes}")
    print(f"Results will be saved to: {save_dir}")

    # 准备数据
    from data.augmentations import get_transform

    # 构建配置字典（简化版，只用于离线数据）
    dataset_split_config_dict = {
        'continual_session_num': 5,  # 只使用离线数据
        'online_novel_unseen_num': 400,
        'online_old_seen_num': 25,
        'online_novel_seen_num': 25,
    }

    # 临时设置相关参数
    args.continual_session_num = 5
    args.online_novel_unseen_num = 400
    args.online_old_seen_num = 25
    args.online_novel_seen_num = 25
    args.shuffle_classes = True
    # 使用测试变换以获得干净的特征

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 获取数据集
    # offline_train_dataset, _, _, _, _, _, _ = get_datasets(
    #     args.dataset_name, test_transform, test_transform, args)

    full_dataset = get_full_dataset_for_prompts(args.dataset_name, test_transform)
    print(f"Successfully loaded {len(full_dataset)} samples")


    # 创建数据加载器
    dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Loaded dataset with {len(full_dataset)} samples")

    # 创建prompts池
    creator = VisualPromptsPoolCreator(
        model_path=args.model_path,
        model_type=args.model_type,
        similarity_threshold=args.similarity_threshold,
        device=args.device,
        dataset_name=args.dataset_name
    )
    # 执行创建过程
    prompts_pool, community_info = creator.create_prompts_pool(
        dataloader=dataloader,
        max_samples=args.max_samples,
        save_dir=save_dir
    )

    print(f"\nPrompts pool creation completed successfully!")
    print(f"Final Prepare shape: {prompts_pool.shape}")
    print(f"Results saved to: {save_dir}")