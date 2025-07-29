#!/usr/bin/env python3
"""
独立的Trainable Prompt Pool生成器
生成标准化的prompt pool，训练时再进行Key-Value分化
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import community as community_louvain
from collections import defaultdict
import pickle
import time
from datetime import datetime
import torch.nn.functional as F

# 导入现有模块
import sys

sys.path.append('models')
from data.get_datasets import get_class_splits, get_datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import vision_transformer as vits
from config import dino_pretrain_path

from transformers import Dinov2Model, Dinov2Config

class TrainablePromptPoolGenerator:
    """
    可训练Prompt Pool生成器
    生成统一的prompt向量，不进行Key-Value分化
    """

    def __init__(self, model_type='dino', model_path=None, similarity_threshold=0.65,
                 community_ratio=1.2, min_community_size=5, device='cuda'):
        """
        初始化生成器

        Args:
            Model_type: backbone类型 ('dino', 'clip', 'custom')
            model_path: 模型路径
            similarity_threshold: 相似度阈值
            community_ratio: 社区比例
            min_community_size: 最小社区大小
            device: 计算设备
        """
        self.model_type = model_type
        self.similarity_threshold = similarity_threshold
        self.community_ratio = community_ratio
        self.min_community_size = min_community_size
        self.device = device

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


    def extract_features(self, dataloader):
        """
        提取所有样本的特征

        Args:
            dataloader: 数据加载器

        Returns:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
        """
        print("Extracting features...")

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                # 处理不同格式的batch
                if len(batch) >= 3:
                    images, labels, _ = batch[:3]
                else:
                    images, labels = batch

                images = images.to(self.device)

                # 根据不同模型类型提取特征
                if self.model_type == 'clip':
                    # 使用CLIP处理器
                    inputs = self.processor(images=images, return_tensors="pt", padding=True, do_rescale=False)
                    pixel_values = inputs['pixel_values'].to(self.device)

                    # 使用CLIP视觉编码器提取特征
                    vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                    features = vision_outputs.pooler_output
                elif self.model_type == 'dino':
                    # DINOv2直接接收归一化的图像张量
                    # 如果图像已经被处理为[0,1]范围，需要调整到[-1,1]或其他DINOv2期望的范围
                    if images.max() <= 1.0:
                        # 标准化到DINOv2期望的输入范围
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                        images = (images - mean) / std

                    # 使用DINOv2提取特征
                    outputs = self.model(pixel_values=images)
                    features = outputs.pooler_output

                features = F.normalize(features, dim=1)

                all_features.append(features.cpu())
                all_labels.append(labels)

        if not all_features:
            raise RuntimeError("No features extracted! Check your data loader.")

        # 合并所有特征和标签
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        print(f"Successfully extracted {len(features)} features with shape {features.shape}")
        return features, labels

    def build_similarity_graph(self, features):
        """
        构建相似度图

        Args:
            features: 特征矩阵 [N, feature_dim]

        Returns:
            G: NetworkX图对象
        """
        print(f"Building similarity graph with threshold {self.similarity_threshold}...")

        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(features)
        n_samples = len(features)

        # 创建图
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))

        # 添加边
        edge_count = 0
        for i in tqdm(range(n_samples), desc="Building graph"):
            for j in range(i + 1, n_samples):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    edge_count += 1

        print(f"Graph built with {n_samples} nodes and {edge_count} edges")
        return G, similarity_matrix

    def discover_communities(self, G):
        """
        社区发现

        Args:
            G: NetworkX图对象

        Returns:
            communities: 社区分配字典
        """
        print("Discovering communities using Louvain algorithm...")

        if G.number_of_nodes() == 0:
            print("Empty graph detected")
            return {}

        # Louvain算法
        communities = community_louvain.best_partition(G, random_state=42)

        # 统计社区信息
        community_sizes = defaultdict(int)
        for node, community_id in communities.items():
            community_sizes[community_id] += 1

        num_communities = len(community_sizes)
        print(f"Discovered {num_communities} communities")

        return communities

    def compute_prompts_from_communities(self, features, labels, communities, num_classes):
        """
        从社区计算prompt向量

        Args:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            communities: 社区分配字典
            num_classes: 类别数量

        Returns:
            prompts: prompt向量 [num_prompts, feature_dim]
            community_info: 社区信息列表
        """
        print("Computing prompts from communities...")

        # 按社区分组
        community_groups = defaultdict(list)
        for node, community_id in communities.items():
            community_groups[community_id].append(node)

        # 筛选有效社区
        valid_communities = {k: v for k, v in community_groups.items()
                             if len(v) >= self.min_community_size}

        print(f"Found {len(valid_communities)} valid communities (size >= {self.min_community_size})")

        # 调整社区数量
        target_num_communities = int(num_classes * self.community_ratio)

        # 如果社区太少，降低最小大小要求
        while len(valid_communities) < target_num_communities and self.min_community_size > 2:
            self.min_community_size -= 1
            valid_communities = {k: v for k, v in community_groups.items()
                                 if len(v) >= self.min_community_size}
            print(f"Reduced min_community_size to {self.min_community_size}, "
                  f"now have {len(valid_communities)} valid communities")

        # 如果社区太多，保留最大的几个
        if len(valid_communities) > target_num_communities:
            community_sizes = [(k, len(v)) for k, v in valid_communities.items()]
            community_sizes.sort(key=lambda x: x[1], reverse=True)
            valid_community_ids = [k for k, _ in community_sizes[:target_num_communities]]
            valid_communities = {k: valid_communities[k] for k in valid_community_ids}
            print(f"Keeping top {target_num_communities} communities by size")

        # 生成prompt向量
        prompts = []
        community_info = []

        for community_id, node_indices in valid_communities.items():
            # 计算社区中心
            community_features = features[node_indices]
            community_prompt = np.mean(community_features, axis=0)
            community_prompt = community_prompt / np.linalg.norm(community_prompt)

            # 分析标签分布
            community_labels = labels[node_indices]
            unique_labels, counts = np.unique(community_labels, return_counts=True)
            label_distribution = dict(zip(unique_labels, counts))

            most_common_label = unique_labels[np.argmax(counts)]
            purity = np.max(counts) / len(node_indices)

            prompts.append(community_prompt)
            community_info.append({
                'community_id': community_id,
                'size': len(node_indices),
                'most_common_label': most_common_label,
                'purity': purity,
                'label_distribution': label_distribution
            })

        prompts = np.array(prompts)
        print(f"Generated {len(prompts)} prompt vectors")

        return prompts, community_info

    def visualize_graph_network(adjacency_matrix, save_path, max_nodes=5000):
        """
        简化版的图网络可视化函数，只显示主网络图。

        Args:
            adjacency_matrix: 图的邻接矩阵
            save_path: 保存可视化图的路径
            max_nodes: 可视化的最大节点数
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置seaborn样式提高美观度
            sns.set(style="whitegrid", context="paper", font_scale=1.2)

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
                width=1.0,
                alpha=edge_alpha,
                edge_color="gray"
            )

            # 使用基于度的颜色映射绘制节点
            nodes = nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=[degrees[n] for n in G.nodes()],
                cmap=plt.cm.viridis,
                alpha=0.7
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

        except Exception as e:
            pass

    '''
    def visualize_communities(self, G, communities, features, labels, save_dir):
        """
        可视化communities

        Args:
            G: NetworkX图对象
            communities: 社区分配字典
            features: 特征矩阵 [N, feature_dim]
            labels: 标签数组 [N]
            save_dir: 保存目录
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.manifold import TSNE

            print("Creating community visualizations...")

            # 设置样式
            sns.set(style="whitegrid", context="paper", font_scale=1.2)

            # 创建可视化目录
            vis_dir = os.path.join(save_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 1. 网络图可视化（显示communities）
            self._visualize_community_network(G, communities, vis_dir)

            # 2. t-SNE特征空间可视化（按community着色）
            self._visualize_community_feature_space(features, labels, communities, vis_dir)

            print(f"Community visualizations saved to {vis_dir}")

        except Exception as e:
            print(f"Community visualization failed: {str(e)}")
    
    def _visualize_community_network(self, G, communities, vis_dir, max_nodes=2000):
        """可视化网络图，按community着色"""
        import matplotlib.pyplot as plt
        import networkx as nx

        # 采样节点（如果图太大）
        n = G.number_of_nodes()
        if n > max_nodes:
            nodes_to_sample = np.random.choice(list(G.nodes()), max_nodes, replace=False)
            G_sampled = G.subgraph(nodes_to_sample).copy()
            communities_sampled = {node: communities[node] for node in nodes_to_sample if node in communities}
        else:
            G_sampled = G
            communities_sampled = communities

        if G_sampled.number_of_nodes() == 0:
            return

        # 计算布局
        pos = nx.spring_layout(G_sampled, k=0.15, iterations=50, seed=42)

        # 准备社区颜色 - 使用数值映射而不是离散颜色
        unique_communities = sorted(list(set(communities_sampled.values())))
        community_to_num = {comm_id: i for i, comm_id in enumerate(unique_communities)}

        # 创建节点颜色数组
        node_colors = [community_to_num[communities_sampled[node]] for node in G_sampled.nodes()]

        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 14))

        # 绘制边
        edge_alpha = max(0.05, min(0.1, 10000 / G_sampled.number_of_edges()))
        nx.draw_networkx_edges(G_sampled, pos, width=0.5, alpha=edge_alpha, edge_color="lightgray", ax=ax)

        # 绘制节点（使用连续颜色映射）
        nodes = nx.draw_networkx_nodes(
            G_sampled, pos,
            node_color=node_colors,
            node_size=25,
            alpha=0.8,
            cmap=plt.cm.tab20,
            ax=ax
        )

        # 添加color bar
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Community ID', rotation=270, labelpad=15)

        # 设置color bar的刻度
        if len(unique_communities) <= 20:
            cbar.set_ticks(range(len(unique_communities)))
            cbar.set_ticklabels([f'C{comm_id}' for comm_id in unique_communities])
        else:
            # 如果社区太多，只显示部分刻度
            step = max(1, len(unique_communities) // 10)
            tick_indices = range(0, len(unique_communities), step)
            cbar.set_ticks(tick_indices)
            cbar.set_ticklabels([f'C{unique_communities[i]}' for i in tick_indices])

        ax.set_title(f'Community Network Visualization\n'
                     f'Nodes: {G_sampled.number_of_nodes()}, Communities: {len(unique_communities)}',
                     fontsize=16)
        ax.axis('off')

        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        plt.savefig(os.path.join(vis_dir, 'community_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_community_feature_space(self, features, labels, communities, vis_dir):
        """t-SNE可视化特征空间，按community着色"""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        print("Computing t-SNE for feature space visualization...")

        # t-SNE降维（采样以加速）
        max_samples = 3000
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
            communities_sample = {i: communities.get(indices[i], -1) for i in range(len(indices))}
        else:
            features_sample = features
            labels_sample = labels
            communities_sample = {i: communities.get(i, -1) for i in range(len(features))}

        # 计算t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample) - 1))
        features_2d = tsne.fit_transform(features_sample)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 左图：按真实标签着色
        unique_labels = sorted(np.unique(labels_sample))
        label_to_num = {label: i for i, label in enumerate(unique_labels)}

        # 创建标签颜色数组
        label_colors = [label_to_num[label] for label in labels_sample]

        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                               c=label_colors, alpha=0.6, s=15, cmap=plt.cm.tab10)

        ax1.set_title('Feature Space by True Labels', fontsize=14)
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')

        # 为左图添加color bar
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Class ID', rotation=270, labelpad=15)
        if len(unique_labels) <= 20:
            cbar1.set_ticks(range(len(unique_labels)))
            cbar1.set_ticklabels([f'{label}' for label in unique_labels])

        # 右图：按community着色
        unique_communities = sorted([c for c in set(communities_sample.values()) if c != -1])
        community_to_num = {comm_id: i for i, comm_id in enumerate(unique_communities)}

        # 创建社区颜色数组（-1表示未分配，用特殊颜色）
        comm_colors = []
        for i in range(len(features_sample)):
            comm_id = communities_sample[i]
            if comm_id == -1:
                comm_colors.append(-1)  # 未分配的点
            else:
                comm_colors.append(community_to_num[comm_id])

        # 分别绘制已分配和未分配的点
        assigned_mask = np.array(comm_colors) != -1
        unassigned_mask = np.array(comm_colors) == -1

        # 绘制未分配的点（灰色）
        if unassigned_mask.any():
            ax2.scatter(features_2d[unassigned_mask, 0], features_2d[unassigned_mask, 1],
                        c='lightgray', alpha=0.3, s=10, label='Unassigned')

        # 绘制已分配的点
        if assigned_mask.any():
            assigned_colors = np.array(comm_colors)[assigned_mask]
            scatter2 = ax2.scatter(features_2d[assigned_mask, 0], features_2d[assigned_mask, 1],
                                   c=assigned_colors, alpha=0.6, s=15, cmap=plt.cm.tab20)

            # 为右图添加color bar
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
            cbar2.set_label('Community ID', rotation=270, labelpad=15)
            if len(unique_communities) <= 20:
                cbar2.set_ticks(range(len(unique_communities)))
                cbar2.set_ticklabels([f'C{comm_id}' for comm_id in unique_communities])

        ax2.set_title('Feature Space by Communities', fontsize=14)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')

        plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, wspace=0.3)
        plt.savefig(os.path.join(vis_dir, 'community_feature_space.png'), dpi=300, bbox_inches='tight')
        plt.close()
    '''
    def _fallback_kmeans_initialization(self, features, num_classes):
        """K-means备选方案"""
        print("Using K-means clustering as fallback...")

        n_clusters = min(num_classes * 2, 50, len(features) // 10)
        n_clusters = max(n_clusters, 5)

        if len(features) < n_clusters:
            n_clusters = max(1, len(features) // 2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_centers = kmeans.fit(features).cluster_centers_

        # 归一化
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

        print(f"Generated {len(cluster_centers)} prompts using K-means")

        return cluster_centers, []

    def create_prompt_pool(self, dataloader, num_classes, save_dir=None):
        """
        完整的prompt pool创建流程

        Args:
            dataloader: 数据加载器
            num_classes: 类别数量
            save_dir: 保存目录

        Returns:
            prompts: prompt向量 [num_prompts, feature_dim]
            community_info: 社区信息
        """
        print("=" * 50)
        print("Creating Trainable Prompt Pool")
        print("=" * 50)

        start_time = time.time()

        # 1. 提取特征
        features, labels = self.extract_features(dataloader)

        # 2. 构建图
        G, similarity_matrix = self.build_similarity_graph(features)

        # 3. 社区发现
        communities = self.discover_communities(G)

        # 4. 生成prompt
        if communities:
            prompts, community_info = self.compute_prompts_from_communities(
                features, labels, communities, num_classes)

            # 5. 可视化communities
            if save_dir:
                # self.visualize_communities(G, communities, features, labels, save_dir)
                self.visualize_graph_network(similarity_matrix,save_dir)
        else:
            print("No communities found, using K-means fallback")
            prompts, community_info = self._fallback_kmeans_initialization(features, num_classes)

        if save_dir:
            self.save_results(save_dir, prompts, community_info)

        elapsed_time = time.time() - start_time
        print("=" * 50)
        print(f"Prompt pool creation completed in {elapsed_time:.2f} seconds!")
        print(f"Created {len(prompts)} prompts with dimension {prompts.shape[1]}")
        print("=" * 50)

        return prompts, community_info

    def save_results(self, save_dir, prompts, community_info):
        """保存结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 主要结果（只基于offline数据生成的静态prompt pool）
        results = {
            'prompts': prompts,  # 统一的prompt向量，训练时进行Key-Value分化
            'community_info': community_info,
            'num_prompts': len(prompts),
            'feature_dim': prompts.shape[1],
            'model_type': self.model_type,
            'similarity_threshold': self.similarity_threshold,
            'creation_time': datetime.now().isoformat(),
            'generator_version': 'trainable_v1.0',
            'data_source': 'offline_only'  # 标明只使用offline数据
        }

        save_path = os.path.join(save_dir, 'trainable_prompt_pool.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Prompt pool saved to {save_path}")

        # 统计信息
        stats_path = os.path.join(save_dir, 'prompt_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("Trainable Prompt Pool Statistics (Offline Only)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of prompts: {len(prompts)}\n")
            f.write(f"Feature dimension: {prompts.shape[1]}\n")
            f.write(f"Backbone type: {self.model_type}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"Data source: Offline training data only\n")
            f.write(f"Creation time: {datetime.now().isoformat()}\n\n")

            if community_info:
                f.write("Community Details:\n")
                f.write("-" * 30 + "\n")
                for i, info in enumerate(community_info):
                    f.write(f"Community {i}:\n")
                    f.write(f"  Size: {info['size']}\n")
                    f.write(f"  Dominant label: {info['most_common_label']}\n")
                    f.write(f"  Purity: {info['purity']:.3f}\n")
                    f.write("\n")

        print(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Trainable Prompt Pool')

    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--prop_train_labels', type=float, default=0.8,
                        help='Proportion of training labels')
    parser.add_argument('--num_old_classes', type=int, default=50,
                        help='Number of old classes')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='dino',
                        choices=['dino', 'clip', 'custom'],
                        help='Model type')
    parser.add_argument('--model_path', type=str, default="/home/ps/_jinwei/DINO_v2_base",
                        help='Path to pretrained model')

    # 生成参数
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Similarity threshold for building graph')
    parser.add_argument('--community_ratio', type=float, default=1.2,
                        help='Community ratio relative to number of classes')
    parser.add_argument('--min_community_size', type=int, default=5,
                        help='Minimum community size')

    # 处理参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./prompt_pools',
                        help='Save directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_ssb_splits', action='store_true', default=False)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 获取数据集配置
    args = get_class_splits(args)

    # 设置实验名称
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.dataset_name}_{args.model_type}_thresh{args.similarity_threshold}_{timestamp}"

    save_dir = os.path.join(args.save_dir, args.exp_name)

    print(f"Starting offline prompt pool creation for {args.dataset_name}")
    print(f"Using {len(args.train_classes)} offline classes: {args.train_classes}")
    print(f"Backbone: {args.model_type}")
    print(f"Results will be saved to: {save_dir}")

    # 准备数据（只使用clean transform获取稳定特征）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 设置临时参数（只用于离线数据集加载）
    args.continual_session_num = 4
    args.online_novel_unseen_num = 400
    args.online_old_seen_num = 50
    args.online_novel_seen_num = 50
    args.shuffle_classes = False

    # 获取离线训练数据集（只用offline数据生成prompt pool）
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

    print(f"Loaded offline dataset with {len(offline_train_dataset)} samples")

    # 创建生成器（专门用于offline数据）
    generator = TrainablePromptPoolGenerator(
        model_type=args.model_type,
        model_path=args.model_path,
        similarity_threshold=args.similarity_threshold,
        community_ratio=args.community_ratio,
        min_community_size=args.min_community_size,
        device=args.device
    )

    # 生成prompt pool（基于offline数据）
    prompts, community_info = generator.create_prompt_pool(
        dataloader=dataloader,
        num_classes=len(args.train_classes),
        save_dir=save_dir
    )

    print(f"\nOffline prompt pool generation completed!")
    print(f"Final prompt pool shape: {prompts.shape}")
    print(f"Results saved to: {save_dir}")
    print(f"Note: This prompt pool is generated from offline data only.")