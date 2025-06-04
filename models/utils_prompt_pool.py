"""
Modify: 添加prompt pool
Date: 2025/05/14
Author: Wei Jin

update: 添加prompt pool update 机制
Date: 2025/05/15
Author: Wei Jin

update: Prompt pool Visualization
Date: 2025/05/16
Author: Wei Jin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import community as community_louvain
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import seaborn as sns


def visualize_graph_network(adjacency_matrix, save_path, max_nodes=5000, logger=None):
    """
    简化版的图网络可视化函数，只显示主网络图。

    Args:
        adjacency_matrix: 图的邻接矩阵
        save_path: 保存可视化图的路径
        max_nodes: 可视化的最大节点数
        logger: 日志记录器
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import seaborn as sns

        if logger:
            logger.info(f"Create Graph，show max {max_nodes} nodes...")

        # 设置seaborn样式提高美观度
        sns.set(style="whitegrid", context="paper", font_scale=1.2)

        # 如果图太大，采样节点
        n = adjacency_matrix.shape[0]
        if n > max_nodes:
            indices = np.random.choice(n, max_nodes, replace=False)
            sampled_adj_matrix = adjacency_matrix[indices][:, indices]
            if logger:
                logger.info(f"From {n} nodes to samping {max_nodes} nodes for visualization...")
        else:
            sampled_adj_matrix = adjacency_matrix
            indices = np.arange(n)
            if logger:
                logger.info(f"Using all {n} nodes for visualization...")

        # 从邻接矩阵创建图
        G = nx.from_numpy_array(sampled_adj_matrix)

        # 计算图统计信息
        if G.number_of_nodes() > 0:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            density = nx.density(G)
        else:
            avg_degree = 0
            density = 0
            if logger:
                logger.warning("There are no nodes in graph.")
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

        if logger:
            logger.info(f"Save to {save_path}")

    except Exception as e:
        if logger:
            logger.error(f"Create picture failed: {str(e)}")

class PromptPool:
    def __init__(self, feature_dim, min_community_size=5, similarity_threshold=0.6, community_ratio=1.4, device='cuda'):
        """
        Initialize the Prompt Pool.

        Args:
            feature_dim: Dimension of feature vectors
            min_community_size: Minimum size of a community to be considered
            similarity_threshold: Threshold for constructing adjacency matrix
            community_ratio: Ratio of number of communities to number of classes
            device: Device to run computations on
        """
        self.feature_dim = feature_dim
        self.min_community_size = min_community_size
        self.similarity_threshold = similarity_threshold
        self.community_ratio = community_ratio
        self.device = device
        self.prompts = None
        self.num_prompts = 0

    def create_prompt_pool(self, model, data_loader, num_classes, logger):
        """
        Create prompt pool from offline training data

        Args:
            model: The trained feature extractor
            data_loader: DataLoader for the dataset
            num_classes: Number of classes in the dataset
            logger: Logger to log progress
        """
        logger.info("Creating prompt pool using community detection...")
        model.eval()

        # Extract features for all samples
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="Extracting features")):
                images = images.to(self.device)
                features = model.backbone(images)  # Extract features from backbone
                features = F.normalize(features, dim=1)  # Normalize features
                all_features.append(features.cpu())
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        logger.info(f"Extracted features for {len(all_features)} samples")

        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(all_features)

        # Create adjacency matrix with threshold
        logger.info(f"Creating adjacency matrix with threshold {self.similarity_threshold}...")
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(np.int8)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

        # Drawing
        # log_dir = os.path.dirname(logger.handlers[0].baseFilename) if hasattr(logger,
        #                                                                       'handlers') and logger.handlers else "."
        # vis_path = os.path.join(log_dir, "graph_network_visualization.png")
        # visualize_graph_network(adjacency_matrix, vis_path, max_nodes=5000, logger=logger)

        # Convert to graph for community detection
        logger.info("Converting to graph and detecting communities...")
        G = nx.from_numpy_array(adjacency_matrix)

        # Use Louvain method for community detection (as an alternative to InfoMap)
        partition = community_louvain.best_partition(G, resolution=1.0)

        # Get communities
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        logger.info(f"Detected {len(communities)} communities")

        # Calculate target number of communities
        target_num_communities = int(num_classes * self.community_ratio)
        logger.info(f"Target number of communities: {target_num_communities}")

        # Filter communities by size
        valid_communities = {k: v for k, v in communities.items() if len(v) >= self.min_community_size}
        logger.info(f"Number of valid communities (size >= {self.min_community_size}): {len(valid_communities)}")

        # If we have too few communities, reduce the min_community_size
        if len(valid_communities) < target_num_communities:
            while len(valid_communities) < target_num_communities and self.min_community_size > 2:
                self.min_community_size -= 1
                valid_communities = {k: v for k, v in communities.items() if len(v) >= self.min_community_size}
                logger.info(
                    f"Reduced min_community_size to {self.min_community_size}, now have {len(valid_communities)} valid communities")

        # If we have too many communities, keep the largest ones
        community_sizes = [(k, len(v)) for k, v in valid_communities.items()]
        community_sizes.sort(key=lambda x: x[1], reverse=True)

        if len(valid_communities) > target_num_communities:
            valid_community_ids = [k for k, _ in community_sizes[:target_num_communities]]
            valid_communities = {k: valid_communities[k] for k in valid_community_ids}
            logger.info(f"Keeping top {target_num_communities} communities by size")

        # Create prompts from community means
        prompts = []
        community_labels = []
        community_info = []

        for community_id, node_indices in valid_communities.items():
            community_features = all_features[node_indices]
            community_prompt = np.mean(community_features, axis=0)
            community_prompt = community_prompt / np.linalg.norm(community_prompt)  # Normalize

            # Get most common label in this community
            community_label_counts = {}
            for idx in node_indices:
                label = all_labels[idx]
                if label not in community_label_counts:
                    community_label_counts[label] = 0
                community_label_counts[label] += 1

            most_common_label = max(community_label_counts.items(), key=lambda x: x[1])[0]
            purity = community_label_counts[most_common_label] / len(node_indices)

            prompts.append(community_prompt)
            community_labels.append(most_common_label)
            community_info.append({
                'size': len(node_indices),
                'most_common_label': most_common_label,
                'purity': purity
            })

        self.prompts = torch.tensor(np.array(prompts), dtype=torch.float32).to(self.device)
        self.num_prompts = len(self.prompts)

        # Log community statistics
        logger.info(f"Created prompt pool with {self.num_prompts} prompts")

        purities = [info['purity'] for info in community_info]
        avg_purity = sum(purities) / len(purities) if purities else 0
        logger.info(f"Community avage purity: {avg_purity:.4f}")

        # Create a visualization of community label distribution
        label_distribution = np.zeros((self.num_prompts, num_classes))
        for i, (community_id, node_indices) in enumerate(list(valid_communities.items())[:self.num_prompts]):
            for idx in node_indices:
                label = all_labels[idx]
                label_distribution[i, label] += 1
            # Normalize by community size
            label_distribution[i] /= len(node_indices)

        return {
            'num_prompts': self.num_prompts,
            'community_info': community_info,
            'label_distribution': label_distribution,
            'Communities avg purity': avg_purity,
            'adjacency_matrix': adjacency_matrix
        }

    def update_prompt_pool_incrementally(self, model, data_loader, similarity_threshold=0.8, ema_alpha=0.9,
                                         logger=None):
        """
        以增量方式更新prompt pool:
        1. 对于与现有prompts相似的样本，使用EMA更新对应的prompt
        2. 对于不与任何现有prompt相似的样本，收集起来进行社区发现，生成新的prompts

        Args:
            model: 当前训练好的模型 (PromptEnhancedModel)
            data_loader: 当前session的数据加载器
            similarity_threshold: 判断样本是否属于已知类的相似度阈值
            ema_alpha: EMA更新的动量系数
            logger: 日志记录器

        Returns:
            更新统计信息
        """
        if logger:
            logger.info("Starting incremental prompt pool update...")

        model.eval()

        # 1. 提取当前session所有样本的特征
        all_features = []

        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
                try:
                    # 处理不同格式的数据加载器返回的批次
                    if len(batch) >= 3:  # 至少包含图像、标签和唯一索引
                        if isinstance(batch[0], list):  # 如果images是一个list (对比学习中常见)
                            # 对比学习中，每个样本有多个视图，我们只取第一个视图
                            images = batch[0][0].to(self.device)  # 第一个视图
                            sample_count += len(images)
                        else:
                            # 常规单视图情况
                            images = batch[0].to(self.device)
                            sample_count += len(images)
                    else:
                        if logger:
                            logger.warning(f"Unexpected batch format at index {batch_idx}")
                        continue

                    # 从backbone提取特征
                    features = model.backbone(images)
                    features = F.normalize(features, dim=1)

                    all_features.append(features.cpu())
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

        if not all_features:
            if logger:
                logger.error("No features extracted. Cannot update prompt pool.")
            return {"error": "No features extracted"}

        all_features = torch.cat(all_features, dim=0)
        if logger:
            logger.info(f"Extracted features from {len(all_features)} samples (processed {sample_count} images)")

        # 确保提取的特征数量与处理的样本数量一致
        if len(all_features) != sample_count:
            if logger:
                logger.warning(f"Feature count ({len(all_features)}) doesn't match sample count ({sample_count})")

        # 2. 计算样本特征与现有prompts的相似度
        prompts = F.normalize(self.prompts, dim=1)
        similarities = all_features @ prompts.cpu().T  # [N_samples, N_prompts]

        # 3. 对每个样本，找到最相似的prompt
        max_similarities, most_similar_prompt_idxs = torch.max(similarities, dim=1)

        # 4. 分离属于已知类和新类的样本
        known_class_mask = max_similarities >= similarity_threshold
        unknown_class_mask = ~known_class_mask

        known_features = all_features[known_class_mask]
        known_prompt_idxs = most_similar_prompt_idxs[known_class_mask]
        unknown_features = all_features[unknown_class_mask]

        if logger:
            logger.info(f"Identified {known_features.shape[0]} samples from known classes")
            logger.info(f"Identified {unknown_features.shape[0]} samples from potentially new classes")

        # 5. 使用EMA更新已知类的prompts
        updated_prompt_count = 0
        prompt_updates = {}

        for prompt_idx in torch.unique(known_prompt_idxs):
            prompt_idx = prompt_idx.item()
            prompt_features = known_features[known_prompt_idxs == prompt_idx]
            if len(prompt_features) > 0:
                # 计算属于该prompt的样本的平均特征
                avg_feature = torch.mean(prompt_features, dim=0)
                avg_feature = F.normalize(avg_feature, dim=0)

                # 使用EMA更新prompt
                old_prompt = self.prompts[prompt_idx].cpu()
                updated_prompt = ema_alpha * old_prompt + (1 - ema_alpha) * avg_feature
                updated_prompt = F.normalize(updated_prompt, dim=0)

                # 更新prompt
                self.prompts[prompt_idx] = updated_prompt.to(self.device)
                updated_prompt_count += 1

                # 获取属于该prompt的样本的相似度
                prompt_similarities = max_similarities[known_class_mask][known_prompt_idxs == prompt_idx]

                prompt_updates[prompt_idx] = {
                    'num_samples': len(prompt_features),
                    'avg_similarity': prompt_similarities.mean().item(),
                    'similarity_std': prompt_similarities.std().item() if len(prompt_similarities) > 1 else 0.0
                }

        if logger:
            logger.info(f"Updated {updated_prompt_count} existing prompts using EMA")

        # 6. 对新类样本进行社区发现
        new_prompts = []
        detected_communities = 0
        community_stats = []

        if len(unknown_features) > 0:
            try:
                # 转换为numpy以使用社区检测算法
                unknown_features_np = unknown_features.numpy()

                # 计算相似度矩阵
                unknown_similarity_matrix = unknown_features @ unknown_features.T
                unknown_similarity_matrix = unknown_similarity_matrix.numpy()

                # 创建邻接矩阵
                adjacency_threshold = max(similarity_threshold * 0.9, 0.5)  # 略低于样本-prompt的阈值，但不低于0.5
                adjacency_matrix = (unknown_similarity_matrix > adjacency_threshold).astype(np.int8)
                np.fill_diagonal(adjacency_matrix, 0)  # 移除自环

                # 使用networkx和社区检测算法
                import networkx as nx
                import community as community_louvain

                # 创建图
                G = nx.from_numpy_array(adjacency_matrix)

                if G.number_of_nodes() > 0:
                    # 社区检测
                    partition = community_louvain.best_partition(G)

                    # 收集社区
                    communities = {}
                    for node, community_id in partition.items():
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node)

                    # 过滤太小的社区
                    min_community_size = max(self.min_community_size, 3)  # 至少3个样本
                    valid_communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}
                    detected_communities = len(valid_communities)

                    if logger:
                        logger.info(f"Detected {detected_communities} valid communities from new class samples")

                    # 7. 从每个有效社区创建新的prompts
                    for community_id, node_indices in valid_communities.items():
                        community_features = unknown_features[node_indices]
                        community_prompt = torch.mean(community_features, dim=0)
                        community_prompt = F.normalize(community_prompt, dim=0)
                        new_prompts.append(community_prompt)

                        # 计算社区内样本的平均相似度
                        community_sim_matrix = community_features @ community_features.T
                        community_sim = (community_sim_matrix.sum() - len(node_indices)) / max(
                            len(node_indices) * (len(node_indices) - 1), 1)  # 避免除零

                        community_stats.append({
                            'size': len(node_indices),
                            'avg_internal_similarity': community_sim.item()
                        })
                else:
                    if logger:
                        logger.warning("No nodes in graph for community detection")
            except Exception as e:
                if logger:
                    logger.error(f"Error during community detection: {str(e)}")
                import traceback
                if logger:
                    logger.error(traceback.format_exc())

        # 8. 将新prompts添加到现有prompts中
        num_new_prompts_added = 0
        if new_prompts:
            try:
                new_prompts_tensor = torch.stack(new_prompts)

                # 在添加前检查新prompts与现有prompts的相似度
                # 避免添加与现有prompts太相似的新prompts
                existing_prompts = F.normalize(self.prompts, dim=1)
                new_prompts_normalized = F.normalize(new_prompts_tensor, dim=1)

                cross_similarities = new_prompts_normalized @ existing_prompts.cpu().T
                max_cross_similarities, _ = torch.max(cross_similarities, dim=1)

                # 只添加与现有prompts相似度较低的新prompts
                unique_threshold = similarity_threshold * 0.95  # 略低于判定阈值
                unique_mask = max_cross_similarities < unique_threshold

                unique_new_prompts = new_prompts_tensor[unique_mask]
                num_new_prompts_added = len(unique_new_prompts)

                if num_new_prompts_added > 0:
                    self.prompts = torch.cat([self.prompts, unique_new_prompts.to(self.device)], dim=0)

                    if logger:
                        logger.info(f"Added {num_new_prompts_added} new prompts to the prompt pool")
                        if num_new_prompts_added < len(new_prompts):
                            logger.info(f"Filtered out {len(new_prompts) - num_new_prompts_added} redundant prompts")
                else:
                    if logger:
                        logger.info(f"All {len(new_prompts)} new prompts were too similar to existing ones, none added")
            except Exception as e:
                if logger:
                    logger.error(f"Error adding new prompts: {str(e)}")
                import traceback
                if logger:
                    logger.error(traceback.format_exc())

        # 9. 更新prompts池的大小信息
        self.num_prompts = len(self.prompts)
        logger.info(f"total prompts after update: {self.num_prompts}")

        # 10. 返回更新统计信息
        update_stats = {
            'num_samples': len(all_features),
            'num_known_samples': known_features.shape[0],
            'num_unknown_samples': unknown_features.shape[0],
            'num_updated_prompts': updated_prompt_count,
            'prompt_updates': prompt_updates,
            'detected_communities': detected_communities,
            'num_new_prompts_before_filtering': len(new_prompts) if new_prompts else 0,
            'num_new_prompts_added': num_new_prompts_added,
            'community_stats': community_stats,
            'total_prompts_after_update': len(self.prompts),
            'adjacency_matrix': adjacency_matrix
        }

        return update_stats

    def save_prompt_pool(self, save_path):
        """Save prompt pool to disk"""
        prompt_pool_dict = {
            'prompts': self.prompts.cpu(),
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'min_community_size': self.min_community_size,
            'similarity_threshold': self.similarity_threshold,
            'community_ratio': self.community_ratio
        }
        torch.save(prompt_pool_dict, save_path)

    def load_prompt_pool(self, load_path, device=None):
        """Load prompt pool from disk"""
        if device:
            self.device = device

        prompt_pool_dict = torch.load(load_path)
        self.prompts = prompt_pool_dict['prompts'].to(self.device)
        self.num_prompts = prompt_pool_dict['num_prompts']
        self.feature_dim = prompt_pool_dict['feature_dim']
        self.min_community_size = prompt_pool_dict['min_community_size']
        self.similarity_threshold = prompt_pool_dict['similarity_threshold']
        self.community_ratio = prompt_pool_dict['community_ratio']

    def enhance_features(self, features, top_k=5):
        """
        Enhance features using the prompt pool

        Args:
            features: Input features from the backbone [B, D]
            top_k: Number of top prompts to use

        Returns:
            Enhanced features [B, D]
        """
        if self.prompts is None or self.num_prompts == 0:
            return features

        # Compute similarity between features and prompts
        # features: [B, D], prompts: [P, D] -> similarity: [B, P]
        similarity = F.normalize(features, dim=1) @ F.normalize(self.prompts, dim=1).T

        # Get top-k prompts for each feature
        top_k_values, top_k_indices = torch.topk(similarity, min(top_k, self.num_prompts), dim=1)

        # Apply softmax to get attention weights
        top_k_weights = F.softmax(top_k_values / 0.1, dim=1)  # temperature=0.1

        # Get weighted prompts for each feature
        enhanced_features = features.clone()
        for i in range(features.shape[0]):
            prompt_contribution = torch.zeros_like(features[i])
            for j in range(top_k_weights.shape[1]):
                prompt_idx = top_k_indices[i, j]
                weight = top_k_weights[i, j]
                prompt_contribution += weight * self.prompts[prompt_idx]

            # Combine original feature with prompt contribution
            enhanced_features[i] = 0.8 * features[i] + 0.2 * prompt_contribution


        # Normalize the enhanced features
        enhanced_features = F.normalize(enhanced_features, dim=1)

        return enhanced_features