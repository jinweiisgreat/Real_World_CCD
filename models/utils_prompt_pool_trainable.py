"""
Modify: 添加prompt pool
Date: 2025/05/14
Author: Wei Jin

update: 添加prompt pool update 机制 + 可学习prompts
Date: 2025/05/15
Author: Wei Jin

update: Prompt pool Visualization
Date: 2025/05/16
Author: Wei Jin

update: Prompt Training
Date: 2025/06/03
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


class LearnablePromptPool(nn.Module):
    def __init__(self, feature_dim, min_community_size=5, similarity_threshold=0.6,
                 community_ratio=1.4, device='cuda', max_prompts=200):
        """
        Initialize the Learnable Prompt Pool.

        Args:
            feature_dim: Dimension of feature vectors
            min_community_size: Minimum size of a community to be considered
            similarity_threshold: Threshold for constructing adjacency matrix
            community_ratio: Ratio of number of communities to number of classes
            device: Device to run computations on
            max_prompts: Maximum number of prompts to prevent memory explosion
        """
        super(LearnablePromptPool, self).__init__()

        self.feature_dim = feature_dim
        self.min_community_size = min_community_size
        self.similarity_threshold = similarity_threshold
        self.community_ratio = community_ratio
        self.device = device
        self.max_prompts = max_prompts
        self.num_prompts = 0

        # 可学习的prompt参数
        self.prompts = None
        self.prompt_keys = None

        # 训练相关参数
        self.prompt_learning_weight = 0.1  # prompt训练损失的权重
        self.diversity_weight = 0.05  # diversity损失权重
        self.alignment_weight = 0.1  # alignment损失权重

    def _initialize_learnable_prompts(self, initial_prompts):
        """从社区检测结果初始化可学习的prompts"""
        if initial_prompts is not None:
            self.num_prompts = min(len(initial_prompts), self.max_prompts)

            # 初始化可学习的prompt参数
            self.prompts = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )
            self.prompt_keys = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            print(f"Initialized {self.num_prompts} learnable prompts")

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

        # Convert to graph for community detection
        logger.info("Converting to graph and detecting communities...")
        G = nx.from_numpy_array(adjacency_matrix)

        # Use Louvain method for community detection
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

        initial_prompts = torch.tensor(np.array(prompts), dtype=torch.float32).to(self.device)

        # 初始化可学习的prompts
        self._initialize_learnable_prompts(initial_prompts)

        # Log community statistics
        logger.info(f"Created prompt pool with {self.num_prompts} learnable prompts")

        purities = [info['purity'] for info in community_info]
        avg_purity = sum(purities) / len(purities) if purities else 0
        logger.info(f"Community average purity: {avg_purity:.4f}")

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

    def update_prompt_pool_incrementally(self, model, data_loader, similarity_threshold=0.8,
                                         ema_alpha=0.9, logger=None):
        """
        以增量方式更新prompt pool - 现在包含learnable prompts的处理
        """
        if logger:
            logger.info("Starting incremental prompt pool update with learnable prompts...")

        model.eval()

        # 1. 提取当前session所有样本的特征
        all_features = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting features")):
                try:
                    if len(batch) >= 3:
                        if isinstance(batch[0], list):
                            images = batch[0][0].to(self.device)
                            sample_count += len(images)
                        else:
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
            logger.info(f"Extracted features from {len(all_features)} samples")

        # 2. 计算样本特征与现有prompts的相似度
        if self.prompts is not None:
            prompts = F.normalize(self.prompts, dim=1)
            similarities = all_features @ prompts.cpu().T  # [N_samples, N_prompts]

            # 3. 对每个样本，找到最相似的prompt
            max_similarities, most_similar_prompt_idxs = torch.max(similarities, dim=1)

            # 4. 分离属于已知类和新类的样本
            known_class_mask = max_similarities >= similarity_threshold
            unknown_class_mask = ~known_class_mask

            known_features = all_features[known_class_mask]
            unknown_features = all_features[unknown_class_mask]

            if logger:
                logger.info(f"Identified {known_features.shape[0]} samples from known classes")
                logger.info(f"Identified {unknown_features.shape[0]} samples from potentially new classes")

            # 5. 对于已知类样本，记录用于后续的EMA更新（在训练过程中进行）
            # 这里我们主要处理新类样本的prompt生成

        else:
            unknown_features = all_features
            if logger:
                logger.info("No existing prompts, treating all samples as unknown")

        # 6. 对新类样本进行社区发现并生成新prompts
        new_prompts = []
        detected_communities = 0

        if len(unknown_features) > 0:
            try:
                unknown_features_np = unknown_features.numpy()
                unknown_similarity_matrix = unknown_features @ unknown_features.T
                unknown_similarity_matrix = unknown_similarity_matrix.numpy()

                adjacency_threshold = max(similarity_threshold * 0.9, 0.5)
                adjacency_matrix = (unknown_similarity_matrix > adjacency_threshold).astype(np.int8)
                np.fill_diagonal(adjacency_matrix, 0)

                import networkx as nx
                import community as community_louvain

                G = nx.from_numpy_array(adjacency_matrix)

                if G.number_of_nodes() > 0:
                    partition = community_louvain.best_partition(G)
                    communities = {}
                    for node, community_id in partition.items():
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node)

                    min_community_size = max(self.min_community_size, 3)
                    valid_communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}
                    detected_communities = len(valid_communities)

                    if logger:
                        logger.info(f"Detected {detected_communities} valid communities from new class samples")

                    for community_id, node_indices in valid_communities.items():
                        community_features = unknown_features[node_indices]
                        community_prompt = torch.mean(community_features, dim=0)
                        community_prompt = F.normalize(community_prompt, dim=0)
                        new_prompts.append(community_prompt)

            except Exception as e:
                if logger:
                    logger.error(f"Error during community detection: {str(e)}")

        # 7. 将新prompts添加到现有prompts中（作为可学习参数）
        num_new_prompts_added = 0
        if new_prompts and len(new_prompts) > 0:
            try:
                new_prompts_tensor = torch.stack(new_prompts)

                # 检查是否超过最大prompt数量
                total_prompts_after_adding = self.num_prompts + len(new_prompts_tensor)
                if total_prompts_after_adding > self.max_prompts:
                    # 只添加到最大数量
                    max_new_prompts = self.max_prompts - self.num_prompts
                    if max_new_prompts > 0:
                        new_prompts_tensor = new_prompts_tensor[:max_new_prompts]
                    else:
                        new_prompts_tensor = torch.empty(0, self.feature_dim)

                if len(new_prompts_tensor) > 0:
                    # 检查与现有prompts的相似度，避免重复
                    if self.prompts is not None:
                        existing_prompts = F.normalize(self.prompts, dim=1)
                        new_prompts_normalized = F.normalize(new_prompts_tensor, dim=1)
                        cross_similarities = new_prompts_normalized @ existing_prompts.cpu().T
                        max_cross_similarities, _ = torch.max(cross_similarities, dim=1)
                        unique_threshold = similarity_threshold * 0.95
                        unique_mask = max_cross_similarities < unique_threshold
                        unique_new_prompts = new_prompts_tensor[unique_mask]
                    else:
                        unique_new_prompts = new_prompts_tensor

                    num_new_prompts_added = len(unique_new_prompts)

                    if num_new_prompts_added > 0:
                        # 重新创建可学习参数
                        if self.prompts is not None:
                            old_prompts = self.prompts.data
                            old_keys = self.prompt_keys.data

                            new_total_prompts = torch.cat([old_prompts, unique_new_prompts.to(self.device)], dim=0)
                            new_total_keys = torch.cat([old_keys, unique_new_prompts.to(self.device)], dim=0)
                        else:
                            new_total_prompts = unique_new_prompts.to(self.device)
                            new_total_keys = unique_new_prompts.to(self.device)

                        # 重新创建Parameter
                        self.prompts = nn.Parameter(new_total_prompts)
                        self.prompt_keys = nn.Parameter(new_total_keys)
                        self.num_prompts = len(self.prompts)

                        if logger:
                            logger.info(f"Added {num_new_prompts_added} new learnable prompts to the pool")

            except Exception as e:
                if logger:
                    logger.error(f"Error adding new prompts: {str(e)}")

        if logger:
            logger.info(f"Total prompts after update: {self.num_prompts}")

        return {
            'num_new_prompts_added': num_new_prompts_added,
            'detected_communities': detected_communities,
            'total_prompts_after_update': self.num_prompts,
        }

    def forward(self, features, top_k=5, return_attention=False):
        """
        Forward pass for learnable prompt pool

        Args:
            features: Input features [B, D]
            top_k: Number of top prompts to use
            return_attention: Whether to return attention weights

        Returns:
            enhanced_features: Enhanced features [B, D]
            attention_weights: (optional) Attention weights [B, top_k]
        """
        if self.prompts is None or self.num_prompts == 0:
            if return_attention:
                return features, None
            return features

        batch_size = features.shape[0]

        # 计算特征与prompt keys的相似度
        features_norm = F.normalize(features, dim=1)
        keys_norm = F.normalize(self.prompt_keys, dim=1)

        # [B, num_prompts]
        similarity = features_norm @ keys_norm.T

        # 选择top-k个最相似的prompts
        top_k = min(top_k, self.num_prompts)
        top_k_values, top_k_indices = torch.topk(similarity, top_k, dim=1)

        # 计算注意力权重
        attention_weights = F.softmax(top_k_values / 0.1, dim=1)  # temperature=0.1

        # 获取选中的prompts
        selected_prompts = self.prompts[top_k_indices]  # [B, top_k, D]

        # 计算prompt contribution
        # [B, top_k, 1] * [B, top_k, D] -> [B, top_k, D] -> [B, D]
        prompt_contribution = torch.sum(
            attention_weights.unsqueeze(-1) * selected_prompts,
            dim=1
        )

        # 自适应增强：基于最高相似度动态调整增强强度
        max_similarity = top_k_values.max(dim=1)[0]  # [B]
        enhancement_strength = torch.sigmoid(max_similarity * 2 - 1)  # 映射到(0,1)

        # 增强特征
        enhanced_features = features + enhancement_strength.unsqueeze(-1) * prompt_contribution
        enhanced_features = F.normalize(enhanced_features, dim=1)

        if return_attention:
            return enhanced_features, {
                'attention_weights': attention_weights,
                'selected_prompt_indices': top_k_indices,
                'enhancement_strength': enhancement_strength,
                'max_similarity': max_similarity
            }

        return enhanced_features

    def compute_prompt_losses(self, features, enhanced_features, attention_info, targets=None):
        """
        计算prompt相关的训练损失

        Args:
            features: 原始特征 [B, D]
            enhanced_features: 增强后特征 [B, D]
            attention_info: 注意力信息字典
            targets: 目标标签 [B] (可选)

        Returns:
            Dict of losses
        """
        losses = {}

        if self.prompts is None or self.num_prompts == 0:
            return {k: torch.tensor(0.0, device=features.device) for k in ['diversity', 'alignment', 'total']}

        # 1. Prompt diversity loss - 防止prompts收敛到相同值
        if self.num_prompts > 1:
            prompt_similarities = F.cosine_similarity(
                self.prompts.unsqueeze(1),
                self.prompts.unsqueeze(0),
                dim=2
            )
            # 移除对角线（自相似度）
            mask = ~torch.eye(self.num_prompts, dtype=torch.bool, device=self.prompts.device)
            off_diagonal_similarities = prompt_similarities[mask]
            # 我们希望prompts之间的相似度尽可能小
            diversity_loss = off_diagonal_similarities.mean()
        else:
            diversity_loss = torch.tensor(0.0, device=features.device)

        # 2. Feature alignment loss - 确保增强有助于特征区分
        # 计算增强前后特征的余弦相似度，希望它们相关但不完全相同
        feature_alignment = F.cosine_similarity(features, enhanced_features, dim=1)
        # 我们希望相似度在一个合理范围内（0.7-0.95）
        target_similarity = 0.85
        alignment_loss = F.mse_loss(feature_alignment,
                                    torch.full_like(feature_alignment, target_similarity))

        # 3. Attention concentration loss - 防止注意力过于分散
        if attention_info is not None and 'attention_weights' in attention_info:
            attention_weights = attention_info['attention_weights']  # [B, top_k]
            # 计算注意力的熵，希望不要太分散
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            concentration_loss = attention_entropy.mean()
        else:
            concentration_loss = torch.tensor(0.0, device=features.device)

        # 4. 如果有标签，计算prompt-target consistency loss
        consistency_loss = torch.tensor(0.0, device=features.device)
        if targets is not None and attention_info is not None:
            # 相同标签的样本应该倾向于选择相似的prompts
            selected_indices = attention_info['selected_prompt_indices']  # [B, top_k]

            # 计算样本对之间的标签相似度和prompt选择相似度
            batch_size = targets.shape[0]
            if batch_size > 1:
                label_similarity = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()

                # 计算prompt选择的相似度（基于选中的prompt indices）
                prompt_selection_similarity = torch.zeros_like(label_similarity)
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            # 计算两个样本选中的prompts的重叠度
                            intersection = len(set(selected_indices[i].cpu().numpy()) &
                                               set(selected_indices[j].cpu().numpy()))
                            union = len(set(selected_indices[i].cpu().numpy()) |
                                        set(selected_indices[j].cpu().numpy()))
                            prompt_selection_similarity[i, j] = intersection / union if union > 0 else 0

                # 希望标签相似的样本选择相似的prompts
                consistency_loss = F.mse_loss(prompt_selection_similarity, label_similarity)

        # 5. Feature-Prompt 对齐损失 - 拉近选中的prompts与对应特征的距离
        # feature_prompt_loss = torch.tensor(0.0, device=features.device)
        # if attention_info is not None and 'selected_prompt_indices' in attention_info:
        #     selected_indices = attention_info['selected_prompt_indices']  # [B, top_k]
        #     attention_weights = attention_info['attention_weights']  # [B, top_k]
        #
        #     batch_size = features.shape[0]
        #     selected_prompts = self.prompts[selected_indices]  # [B, top_k, D]
        #
        #     # 计算每个特征与其选中的prompts之间的余弦相似度
        #     features_expanded = features.unsqueeze(1)  # [B, 1, D]
        #     cosine_sim = F.cosine_similarity(features_expanded, selected_prompts, dim=2)  # [B, top_k]
        #
        #     # 使用注意力权重作为权重，计算加权的相似度损失
        #     # 我们希望最大化相似度，所以使用 1-similarity 作为损失
        #     feature_prompt_loss = torch.sum((1 - cosine_sim) * attention_weights) / batch_size


        # 组合所有损失
        losses['diversity'] = self.diversity_weight * diversity_loss
        losses['alignment'] = self.alignment_weight * alignment_loss
        losses['concentration'] = 0.02 * concentration_loss  # 较小的权重
        losses['consistency'] = 0.05 * consistency_loss  # 较小的权重
        # losses['feature_prompt'] = 0.1 * feature_prompt_loss  # 权重可调整

        losses['total'] = sum(losses.values())

        return losses

    def save_prompt_pool(self, save_path):
        """Save learnable prompt pool to disk"""
        prompt_pool_dict = {
            'prompts': self.prompts.cpu() if self.prompts is not None else None,
            'prompt_keys': self.prompt_keys.cpu() if self.prompt_keys is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'min_community_size': self.min_community_size,
            'similarity_threshold': self.similarity_threshold,
            'community_ratio': self.community_ratio,
            'max_prompts': self.max_prompts,
            'prompt_learning_weight': self.prompt_learning_weight,
            'diversity_weight': self.diversity_weight,
            'alignment_weight': self.alignment_weight
        }
        torch.save(prompt_pool_dict, save_path)

    def load_prompt_pool(self, load_path, device=None):
        """Load learnable prompt pool from disk"""
        if device:
            self.device = device

        prompt_pool_dict = torch.load(load_path, map_location=self.device)

        if prompt_pool_dict['prompts'] is not None:
            self.prompts = nn.Parameter(prompt_pool_dict['prompts'].to(self.device))
            self.prompt_keys = nn.Parameter(prompt_pool_dict['prompt_keys'].to(self.device))
        else:
            self.prompts = None
            self.prompt_keys = None

        self.num_prompts = prompt_pool_dict['num_prompts']
        self.feature_dim = prompt_pool_dict['feature_dim']
        self.min_community_size = prompt_pool_dict['min_community_size']
        self.similarity_threshold = prompt_pool_dict['similarity_threshold']
        self.community_ratio = prompt_pool_dict['community_ratio']

        # 加载训练相关参数（兼容旧版本）
        self.max_prompts = prompt_pool_dict.get('max_prompts', 200)
        self.prompt_learning_weight = prompt_pool_dict.get('prompt_learning_weight', 0.1)
        self.diversity_weight = prompt_pool_dict.get('diversity_weight', 0.05)
        self.alignment_weight = prompt_pool_dict.get('alignment_weight', 0.1)

    def enhance_features(self, features, top_k=5):
        """
        向后兼容的特征增强接口
        """
        return self.forward(features, top_k=top_k, return_attention=False)


# 为了兼容，保留原来的PromptPool类名
PromptPool = LearnablePromptPool