"""
Modify: 添加prompt pool
Date: 2025/05/14
Author: Wei Jin

update: 添加prompt pool update 机制
Date: 2025/05/15
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
import seaborn as sns

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
                features = model[0](images)  # Extract features from backbone
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
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(np.int8)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

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
            'label_distribution': label_distribution
        }

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
            enhanced_features[i] = features[i] + 0.5 * prompt_contribution

        # Normalize the enhanced features
        enhanced_features = F.normalize(enhanced_features, dim=1)

        return enhanced_features

    def visualize_prompt_pool(self, output_path, class_labels=None, label_names=None, figsize=(15, 15)):
        """
        可视化Prompt Pool中的社区划分情况。

        Args:
            output_path (str): 输出图像的保存路径
            class_labels (list, optional): 每个prompt对应的类别标签
            label_names (dict, optional): 类别ID到类别名称的映射
            figsize (tuple, optional): 图像大小
        """

        if self.prompts is None or len(self.prompts) == 0:
            print("Prompt pool is empty, nothing to visualize.")
            return

        # 设置颜色映射
        if class_labels is not None:
            num_classes = max(class_labels) + 1
            colors = sns.color_palette("husl", n_colors=num_classes)
            class_colors = {i: colors[i] for i in range(num_classes)}
        else:
            # 基于prompts之间的相似度进行社区检测
            prompts = self.prompts.cpu().numpy()
            similarity_matrix = prompts @ prompts.T
            np.fill_diagonal(similarity_matrix, 0)

            # 创建相似度图
            threshold = 0.5
            adjacency_matrix = (similarity_matrix > threshold).astype(np.int8)

            # 使用networkx进行社区检测
            import networkx as nx
            import community as community_louvain

            G = nx.from_numpy_array(adjacency_matrix)
            partition = community_louvain.best_partition(G)

            # 获取社区ID作为类别标签
            class_labels = [partition[i] for i in range(len(self.prompts))]
            num_communities = max(class_labels) + 1
            colors = sns.color_palette("husl", n_colors=num_communities)
            class_colors = {i: colors[i] for i in range(num_communities)}

        # 使用t-SNE降维以便可视化
        tsne = TSNE(n_components=2, perplexity=min(30, len(self.prompts) - 1), random_state=42)
        prompts_2d = tsne.fit_transform(self.prompts.cpu().numpy())

        # 创建图形
        plt.figure(figsize=figsize)

        # 绘制点
        for i, (x, y) in enumerate(prompts_2d):
            label = class_labels[i] if class_labels is not None else 0
            color = class_colors[label]
            plt.scatter(x, y, color=color, s=100, alpha=0.7)

            # 如果有标签名称，则显示标签
            if label_names is not None and label in label_names:
                plt.annotate(label_names[label], (x, y), fontsize=8,
                             ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            else:
                # 显示prompt索引
                plt.annotate(str(i), (x, y), fontsize=8,
                             ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        # 绘制高相似度连接
        if similarity_matrix is not None:
            for i in range(len(prompts_2d)):
                for j in range(i + 1, len(prompts_2d)):
                    if similarity_matrix[i, j] > threshold:
                        plt.plot([prompts_2d[i, 0], prompts_2d[j, 0]],
                                 [prompts_2d[i, 1], prompts_2d[j, 1]],
                                 'gray', alpha=0.2)

        # 添加图例
        if class_labels is not None and label_names is not None:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w',
                                      label=label_names[i] if i in label_names else f"Class {i}",
                                      markerfacecolor=class_colors[i], markersize=10)
                               for i in sorted(set(class_labels))]
            plt.legend(handles=legend_elements, loc='best')

        plt.title(f'Prompt Pool Visualization (t-SNE) - {len(self.prompts)} prompts')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_path}")

        # 额外创建社区网络图
        plt.figure(figsize=figsize)
        G = nx.Graph()

        # 添加节点
        for i in range(len(prompts_2d)):
            G.add_node(i, pos=(prompts_2d[i, 0], prompts_2d[i, 1]),
                       community=class_labels[i] if class_labels is not None else 0)

        # 添加边
        if similarity_matrix is not None:
            for i in range(len(prompts_2d)):
                for j in range(i + 1, len(prompts_2d)):
                    if similarity_matrix[i, j] > threshold:
                        G.add_edge(i, j, weight=similarity_matrix[i, j])

        # 获取布局
        pos = nx.get_node_attributes(G, 'pos')

        # 按社区给节点着色
        community_colors = [class_colors[G.nodes[n]['community']] for n in G.nodes]

        # 绘制图
        nx.draw(G, pos, node_color=community_colors, with_labels=True, node_size=100,
                font_size=8, font_color='black', alpha=0.7, width=0.5, edge_color='gray')

        plt.title(f'Prompt Pool Network - {len(self.prompts)} prompts')
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Network visualization saved to {output_path.replace('.png', '_network.png')}")

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

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
                # 适应不同的数据加载器格式
                if len(batch) == 2:
                    images, _ = batch
                elif len(batch) >= 3:
                    images = batch[0]

                images = images.to(self.device)
                features = model.backbone(images)
                features = F.normalize(features, dim=1)
                all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        if logger:
            logger.info(f"Extracted features from {len(all_features)} samples")

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

                prompt_updates[prompt_idx] = {
                    'num_samples': len(prompt_features),
                    'avg_similarity': max_similarities[known_prompt_idxs == prompt_idx].mean().item(),
                    'similarity_std': max_similarities[known_prompt_idxs == prompt_idx].std().item()
                }

        if logger:
            logger.info(f"Updated {updated_prompt_count} existing prompts using EMA")

        # 6. 对新类样本进行社区发现
        new_prompts = []
        detected_communities = 0

        if len(unknown_features) > 0:
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
            community_stats = []

            for community_id, node_indices in valid_communities.items():
                community_features = unknown_features[node_indices]
                community_prompt = torch.mean(community_features, dim=0)
                community_prompt = F.normalize(community_prompt, dim=0)
                new_prompts.append(community_prompt)

                # 计算社区内样本的平均相似度
                community_sim_matrix = community_features @ community_features.T
                community_sim = (community_sim_matrix.sum() - len(node_indices)) / (
                            len(node_indices) * (len(node_indices) - 1))

                community_stats.append({
                    'size': len(node_indices),
                    'avg_internal_similarity': community_sim.item()
                })

        # 8. 将新prompts添加到现有prompts中
        if new_prompts:
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

            if len(unique_new_prompts) > 0:
                self.prompts = torch.cat([self.prompts, unique_new_prompts.to(self.device)], dim=0)

                if logger:
                    logger.info(f"Added {len(unique_new_prompts)} new prompts to the prompt pool")
                    if len(unique_new_prompts) < len(new_prompts):
                        logger.info(f"Filtered out {len(new_prompts) - len(unique_new_prompts)} redundant prompts")
            else:
                if logger:
                    logger.info(f"All {len(new_prompts)} new prompts were too similar to existing ones, none added")

        # 9. 更新prompts池的大小信息
        self.num_prompts = len(self.prompts)

        # 10. 返回更新统计信息
        update_stats = {
            'num_samples': len(all_features),
            'num_known_samples': known_features.shape[0],
            'num_unknown_samples': unknown_features.shape[0],
            'num_updated_prompts': updated_prompt_count,
            'prompt_updates': prompt_updates,
            'detected_communities': detected_communities,
            'num_new_prompts_before_filtering': len(new_prompts) if new_prompts else 0,
            'num_new_prompts_added': len(unique_new_prompts) if 'unique_new_prompts' in locals() else 0,
            'community_stats': community_stats if 'community_stats' in locals() else [],
            'total_prompts_after_update': len(self.prompts)
        }

        return update_stats