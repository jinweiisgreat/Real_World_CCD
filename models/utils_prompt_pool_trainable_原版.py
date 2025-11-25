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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm
import networkx as nx
import community as community_louvain
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
        super(LearnablePromptPool, self).__init__()

        self.feature_dim = feature_dim
        self.min_community_size = min_community_size
        self.similarity_threshold = similarity_threshold
        self.community_ratio = community_ratio
        self.device = device
        self.max_prompts = max_prompts
        self.num_prompts = 0

        # 可学习的prompt参数：分离Key和Value的设计
        self.prompt_keys = None  # 用于相似度计算和prompt选择
        self.prompt_values = None  # 用于实际的特征增强

        # 统计信息
        self.prompt_usage_stats = None
        self.community_info = []

        # 训练相关权重
        self.task_alignment_weight = 0.8  # 任务对齐损失权重
        self.usage_efficiency_weight = 0.2  # 使用效率损失权重
        self.key_value_consistency_weight = 0.1  # Key-Value一致性权重
        self.key_feature_similarity_weight = 0.15  # Key-Feature相似度损失权重

    def create_prompt_pool(self, model, data_loader, num_classes, logger):
        """
        使用社区发现算法创建初始的prompt pool

        Args:
            model: 训练后的模型，用于特征提取
            data_loader: 数据加载器
            num_classes: 类别数量
            logger: 日志记录器

        Returns:
            prompt_pool_stats: 创建统计信息
        """
        logger.info("Creating prompt pool using community detection...")
        model.eval()

        # 第一步：提取所有样本的特征
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="Extracting features")):
                images = images.to(self.device)
                # 使用已训练的backbone提取特征
                features = model.backbone(images)
                features = F.normalize(features, dim=1)
                all_features.append(features.cpu())
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        logger.info(f"Extracted features for {len(all_features)} samples")

        # 第二步：构建相似度矩阵和邻接矩阵
        logger.info("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(all_features)

        logger.info(f"Creating adjacency matrix with threshold {self.similarity_threshold}...")
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(np.int8)
        np.fill_diagonal(adjacency_matrix, 0)  # 移除自环

        # 第三步：使用Louvain算法进行社区检测
        logger.info("Performing community detection...")
        G = nx.from_numpy_array(adjacency_matrix)

        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, using K-means as fallback")
            return self._fallback_kmeans_initialization(all_features, num_classes, logger)

        # 使用Louvain方法进行社区检测
        partition = community_louvain.best_partition(G, resolution=1.0)

        # 第四步：处理社区检测结果
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        logger.info(f"Detected {len(communities)} communities")

        # 第五步：筛选有效社区并生成prompt
        valid_communities = {k: v for k, v in communities.items()
                             if len(v) >= self.min_community_size}

        logger.info(f"Found {len(valid_communities)} valid communities (size >= {self.min_community_size})")

        # 如果有效社区太少，降低最小社区大小要求
        target_num_communities = int(num_classes * self.community_ratio)
        while len(valid_communities) < target_num_communities and self.min_community_size > 2:
            self.min_community_size -= 1
            valid_communities = {k: v for k, v in communities.items()
                                 if len(v) >= self.min_community_size}
            logger.info(f"Reduced min_community_size to {self.min_community_size}, "
                        f"now have {len(valid_communities)} valid communities")

        # 如果社区太多，保留最大的几个
        if len(valid_communities) > target_num_communities:
            community_sizes = [(k, len(v)) for k, v in valid_communities.items()]
            community_sizes.sort(key=lambda x: x[1], reverse=True)
            valid_community_ids = [k for k, _ in community_sizes[:target_num_communities]]
            valid_communities = {k: valid_communities[k] for k in valid_community_ids}
            logger.info(f"Keeping top {target_num_communities} communities by size")

        # 第六步：从社区生成prompt
        prompts = []
        community_info = []

        for community_id, node_indices in valid_communities.items():
            # 计算社区中心作为prompt
            community_features = all_features[node_indices]
            community_prompt = np.mean(community_features, axis=0)
            community_prompt = community_prompt / np.linalg.norm(community_prompt)

            # 分析社区的标签分布
            community_labels = all_labels[node_indices]
            label_counts = {}
            for label in community_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
            purity = label_counts[most_common_label] / len(node_indices)

            prompts.append(community_prompt)
            community_info.append({
                'community_id': community_id,
                'size': len(node_indices),
                'most_common_label': most_common_label,
                'purity': purity,
                'label_distribution': label_counts
            })

        # 第七步：初始化可学习的prompt参数
        if prompts:
            initial_prompts = torch.tensor(np.array(prompts), dtype=torch.float32).to(self.device)
            self._initialize_learnable_prompts(initial_prompts)
            self.community_info = community_info

            logger.info(f"Successfully created {self.num_prompts} learnable prompts from communities")

            # 计算平均纯度
            avg_purity = np.mean([info['purity'] for info in community_info])
            logger.info(f"Average community purity: {avg_purity:.4f}")

        else:
            logger.warning("No valid communities found, using K-means fallback")
            return self._fallback_kmeans_initialization(all_features, num_classes, logger)

        return {
            'num_prompts': self.num_prompts,
            'community_info': community_info,
            'avg_purity': avg_purity,
            'adjacency_matrix': adjacency_matrix,
            'method': 'community_detection'
        }

    def _fallback_kmeans_initialization(self, all_features, num_classes, logger):
        """
        当社区检测失败时的备用K-means初始化方法
        """
        logger.info("Using K-means clustering as fallback initialization...")

        # 使用K-means聚类
        n_clusters = min(num_classes * 2, 50, len(all_features) // 10)
        n_clusters = max(n_clusters, 5)  # 至少5个cluster

        if len(all_features) < n_clusters:
            logger.warning(f"Too few samples ({len(all_features)}) for {n_clusters} clusters")
            n_clusters = max(1, len(all_features) // 2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_centers = kmeans.fit(all_features).cluster_centers_

        # 归一化聚类中心
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

        # 初始化prompt参数
        initial_prompts = torch.tensor(cluster_centers, dtype=torch.float32).to(self.device)
        self._initialize_learnable_prompts(initial_prompts)

        logger.info(f"Initialized {self.num_prompts} prompts using K-means clustering")

        return {
            'num_prompts': self.num_prompts,
            'community_info': [],
            'avg_purity': 0.0,
            'method': 'kmeans_fallback'
        }

    def _initialize_learnable_prompts(self, initial_prompts):
        """
        从初始prompts初始化可学习的参数

        Args:
            initial_prompts: 初始prompt张量 [num_prompts, feature_dim]
        """
        if initial_prompts is not None:
            self.num_prompts = min(len(initial_prompts), self.max_prompts)

            # 初始化Key和Value参数
            # Key：专门用于计算相似度，决定prompt选择
            self.prompt_keys = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            # Value：专门用于特征增强
            self.prompt_values = nn.Parameter(
                initial_prompts[:self.num_prompts].clone().detach()
            )

            # 初始化使用统计 - 确保使用正确的设备和数据类型
            self.prompt_usage_stats = torch.zeros(
                self.num_prompts,
                device=self.device,
                dtype=torch.float
            )

            print(f"Initialized {self.num_prompts} learnable prompts with separate Keys and Values")
            print(f"Created usage statistics tracker on device {self.device}")
        else:
            print("No initial prompts provided, skipping initialization")

    def forward(self, features, top_k=5, return_attention=False):
        """
        Forward pass for learnable prompt pool

        Args:
            features: Input features [B, D]
            top_k: Number of top prompts to use
            return_attention: Whether to return attention weights

        Returns:
            enhanced_features: Enhanced features [B, D]
            attention_info: (optional) Attention weights [B, top_k]
        """
        if self.prompt_keys is None or self.num_prompts == 0:
            if return_attention:
                return features, None
            return features

        batch_size = features.shape[0]

        # 计算特征与prompt keys的相似度
        features_norm = F.normalize(features, dim=1)
        keys_norm = F.normalize(self.prompt_keys, dim=1)

        # 相似度矩阵 [B, num_prompts]
        similarity = features_norm @ keys_norm.T

        # 选择top-k个最相似的prompts
        top_k = min(top_k, self.num_prompts)
        top_k_values, top_k_indices = torch.topk(similarity, top_k, dim=1)

        # 统计使用情况 - 修复的代码部分
        if hasattr(self, 'prompt_usage_stats'):
            # 确保usage_stats是在正确的设备上
            if self.prompt_usage_stats.device != features.device:
                self.prompt_usage_stats = self.prompt_usage_stats.to(features.device)

            # 创建一个用于计数的临时张量
            batch_usage = torch.zeros(self.num_prompts,
                                      device=features.device,
                                      dtype=torch.float)

            # 在每个batch中计数每个prompt的使用次数
            for indices in top_k_indices:
                for idx in indices:
                    batch_usage[idx] += 1

            # 更新总体使用统计 - 使用带有梯度的累加操作
            self.prompt_usage_stats = self.prompt_usage_stats + batch_usage

        # 计算注意力权重（使用较低的温度增强选择性）
        attention_weights = F.softmax(top_k_values / 0.05, dim=1)

        # 获取选中的prompt values用于特征增强
        selected_prompt_values = self.prompt_values[top_k_indices]  # [B, top_k, D]

        # 计算加权的prompt contribution
        prompt_contribution = torch.sum(
            attention_weights.unsqueeze(-1) * selected_prompt_values,
            dim=1
        )

        # 自适应增强策略：基于最高相似度动态调整增强强度
        max_similarity = top_k_values.max(dim=1)[0]  # [B]
        enhancement_strength = torch.sigmoid(max_similarity * 3 - 1.5)

        # 使用残差连接进行特征增强
        enhanced_features = features + enhancement_strength.unsqueeze(-1) * prompt_contribution
        enhanced_features = F.normalize(enhanced_features, dim=1)


        attention_info = {
            'attention_weights': attention_weights,
            'selected_prompt_indices': top_k_indices,
            'enhancement_strength': enhancement_strength,
            'max_similarity': max_similarity,
            'prompt_contribution': prompt_contribution
            }


        return enhanced_features, attention_info

    def compute_prompt_losses(self, features, enhanced_features, attention_info, targets=None, logits=None):
        """
        计算prompt相关的训练损失
        """
        losses = {}
        device = features.device

        if self.prompt_keys is None or self.num_prompts == 0:
            return {k: torch.tensor(0.0, device=device) for k in
                    ['task_alignment', 'usage_efficiency', 'key_value_consistency', 'total']}

        # 1. 任务对齐损失：确保prompt增强真正有助于分类任务
        task_alignment_loss = torch.tensor(0.0, device=device)

        if logits is not None and targets is not None:
            # 处理尺寸不匹配的问题
            batch_size = targets.size(0)
            n_views = logits.size(0) // batch_size  # 计算视图数量

            if n_views > 1:
                # 如果有多个视图，重新整形logits并只使用第一个视图进行评估
                reshaped_logits = logits.view(n_views, batch_size, -1)
                first_view_logits = reshaped_logits[0]  # 取第一个视图

                enhanced_predictions = F.softmax(first_view_logits, dim=1)
                predicted_classes = first_view_logits.argmax(dim=1)
            else:
                enhanced_predictions = F.softmax(logits, dim=1)
                predicted_classes = logits.argmax(dim=1)

            # 对于预测正确的样本，我们希望增强后的置信度更高
            correct_mask = (predicted_classes == targets)
            if correct_mask.any():
                correct_confidences = enhanced_predictions[correct_mask, targets[correct_mask]]
                # 使用负对数似然鼓励高置信度
                task_alignment_loss = -torch.log(correct_confidences + 1e-8).mean()
        else:
            # 备用方案：确保增强后的特征与原始特征保持合理的相关性
            feature_similarity = F.cosine_similarity(features, enhanced_features, dim=1)
            target_similarity = 0.85
            task_alignment_loss = F.mse_loss(feature_similarity,
                                             torch.full_like(feature_similarity, target_similarity))

        # 2. 使用效率损失：鼓励合理的注意力分布
        usage_efficiency_loss = torch.tensor(0.0, device=device)

        if attention_info is not None and 'attention_weights' in attention_info:
            attention_weights = attention_info['attention_weights']  # [B, top_k]

            # 计算注意力熵，避免过度集中或过度分散
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            # 目标熵：略低于uniform分布的熵，鼓励适度的专门化
            target_entropy = np.log(attention_weights.shape[1]) * 0.7
            usage_efficiency_loss = F.mse_loss(attention_entropy,
                                               torch.full_like(attention_entropy, target_entropy))

        # 3. Key-Value一致性损失：防止Key和Value过度分化
        key_value_consistency_loss = torch.tensor(0.0, device=device)

        if self.num_prompts > 1:
            # 计算Key和Value之间的相似度
            keys_norm = F.normalize(self.prompt_keys, dim=1)
            values_norm = F.normalize(self.prompt_values, dim=1)

            key_value_similarity = torch.sum(keys_norm * values_norm, dim=1)
            # 我们希望Key和Value保持一定的相关性，但不要完全相同
            target_similarity = 0.7
            key_value_consistency_loss = F.mse_loss(key_value_similarity,
                                                    torch.full_like(key_value_similarity, target_similarity))

        # 4. 新增：Key-Feature相似度损失 - 拉近选中的prompt keys与对应特征的距离
        key_feature_similarity_loss = torch.tensor(0.0, device=device)

        if attention_info is not None and 'selected_prompt_keys' in attention_info and 'features_norm' in attention_info:
            selected_prompt_keys = attention_info['selected_prompt_keys']  # [B, top_k, D]
            features_norm = attention_info['features_norm']  # [B, D]
            attention_weights = attention_info['attention_weights']  # [B, top_k]

            # 计算每个特征与其选中的prompt keys之间的余弦相似度
            features_expanded = features_norm.unsqueeze(1)  # [B, 1, D]
            cosine_similarities = F.cosine_similarity(features_expanded, selected_prompt_keys, dim=2)  # [B, top_k]

            # 使用注意力权重对相似度进行加权，重点关注最相关的prompt keys
            weighted_similarities = cosine_similarities * attention_weights  # [B, top_k]

            # 计算加权平均相似度
            avg_weighted_similarity = weighted_similarities.sum(dim=1) / (attention_weights.sum(dim=1) + 1e-8)  # [B]

            # 我们希望最大化相似度，所以使用 1 - similarity 作为损失
            # 也可以使用负相似度或者其他形式
            key_feature_similarity_loss = (1.0 - avg_weighted_similarity).mean()

            # 可选：添加额外的约束，鼓励最相似的prompt key与特征有更高的相似度
            # 取每个样本最相似的prompt key
            max_similarities = cosine_similarities.max(dim=1)[0]  # [B]
            # 鼓励最大相似度接近1
            max_similarity_loss = (1.0 - max_similarities).mean()

            # 将两个损失组合
            key_feature_similarity_loss = 0.7 * key_feature_similarity_loss + 0.3 * max_similarity_loss

        # 组合所有损失
        losses['task_alignment'] = self.task_alignment_weight * task_alignment_loss
        losses['usage_efficiency'] = self.usage_efficiency_weight * usage_efficiency_loss
        losses['key_value_consistency'] = self.key_value_consistency_weight * key_value_consistency_loss
        losses['key_feature_similarity'] = self.key_feature_similarity_weight * key_feature_similarity_loss
        losses['total'] = losses['task_alignment'] + losses['usage_efficiency'] + losses['key_value_consistency'] + losses['key_feature_similarity']

        return losses

    def update_prompt_pool_incrementally(self, model, data_loader, similarity_threshold=0.8,
                                         ema_alpha=0.9, logger=None):
        """
        增量更新prompt pool

        Args:
            model: 当前模型
            data_loader: 新数据的加载器
            similarity_threshold: 相似度阈值
            ema_alpha: EMA系数（暂未使用）
            logger: 日志记录器

        Returns:
            update_stats: 更新统计信息
        """
        if logger:
            logger.info("Starting incremental prompt pool update...")

        model.eval()
        all_features = []

        # 提取新数据的特征
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    if len(batch) >= 3:
                        if isinstance(batch[0], list):
                            images = batch[0][0].to(self.device)
                        else:
                            images = batch[0].to(self.device)
                    else:
                        continue

                    features = model.backbone(images)
                    features = F.normalize(features, dim=1)
                    all_features.append(features.cpu())

                except Exception as e:
                    if logger:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

        if not all_features:
            return {"error": "No features extracted"}

        all_features = torch.cat(all_features, dim=0)

        if logger:
            logger.info(f"Extracted features from {len(all_features)} samples")

        # 评估现有prompt pool的覆盖情况
        well_covered_ratio = 1.0

        if self.prompt_keys is not None:
            keys_norm = F.normalize(self.prompt_keys, dim=1)
            similarities = all_features @ keys_norm.cpu().T

            max_similarities, _ = torch.max(similarities, dim=1)
            well_covered_ratio = (max_similarities >= similarity_threshold).float().mean()

            if logger:
                logger.info(f"Current prompts cover {well_covered_ratio:.2%} of new samples")

        # 只有当覆盖率不足时才添加新prompt
        num_new_prompts_added = 0

        if well_covered_ratio < 0.8:  # 如果覆盖率低于80%
            # 找到覆盖不好的样本
            poorly_covered_mask = max_similarities < similarity_threshold
            poorly_covered_features = all_features[poorly_covered_mask]

            if len(poorly_covered_features) > 20:  # 确保有足够的样本
                # 对这些样本进行K-means聚类
                n_new_clusters = min(5, len(poorly_covered_features) // 15)
                if n_new_clusters > 0:
                    kmeans = KMeans(n_clusters=n_new_clusters, random_state=42)
                    cluster_centers = kmeans.fit(poorly_covered_features.numpy()).cluster_centers_

                    # 归一化新的prompt
                    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
                    new_prompts = torch.tensor(cluster_centers, dtype=torch.float32).to(self.device)

                    # 检查是否超过最大prompt数量
                    total_prompts_after_adding = self.num_prompts + len(new_prompts)
                    if total_prompts_after_adding <= self.max_prompts:
                        # 扩展现有的prompt参数
                        old_keys = self.prompt_keys.data
                        old_values = self.prompt_values.data

                        new_total_keys = torch.cat([old_keys, new_prompts], dim=0)
                        new_total_values = torch.cat([old_values, new_prompts], dim=0)

                        # 重新创建Parameter
                        self.prompt_keys = nn.Parameter(new_total_keys)
                        self.prompt_values = nn.Parameter(new_total_values)
                        self.num_prompts = len(self.prompt_keys)

                        # 扩展使用统计
                        old_stats = self.prompt_usage_stats
                        new_stats = torch.zeros(len(new_prompts), device=self.device)
                        self.prompt_usage_stats = torch.cat([old_stats, new_stats])

                        num_new_prompts_added = len(new_prompts)

                        if logger:
                            logger.info(f"Added {num_new_prompts_added} new prompts to the pool")

        return {
            'num_new_prompts_added': num_new_prompts_added,
            'total_prompts_after_update': self.num_prompts,
            'well_covered_ratio': well_covered_ratio.item()
        }

    def get_prompt_parameters(self):
        """获取所有prompt相关的参数"""
        params = []
        if self.prompt_keys is not None:
            params.append(self.prompt_keys)
        if self.prompt_values is not None:
            params.append(self.prompt_values)
        return params

    def save_prompt_pool(self, save_path):
        """保存prompt pool到磁盘"""
        prompt_pool_dict = {
            'prompt_keys': self.prompt_keys.cpu() if self.prompt_keys is not None else None,
            'prompt_values': self.prompt_values.cpu() if self.prompt_values is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'prompt_usage_stats': self.prompt_usage_stats.cpu() if self.prompt_usage_stats is not None else None,
            'community_info': self.community_info,
            'task_alignment_weight': self.task_alignment_weight,
            'usage_efficiency_weight': self.usage_efficiency_weight,
            'key_value_consistency_weight': self.key_value_consistency_weight,
            'key_feature_similarity_weight': self.key_feature_similarity_weight,
            'max_prompts': self.max_prompts,
            'similarity_threshold': self.similarity_threshold
        }
        torch.save(prompt_pool_dict, save_path)

    def load_prompt_pool(self, load_path, device=None):
        """从磁盘加载prompt pool"""
        if device:
            self.device = device

        prompt_pool_dict = torch.load(load_path, map_location=self.device)

        if prompt_pool_dict['prompt_keys'] is not None:
            self.prompt_keys = nn.Parameter(prompt_pool_dict['prompt_keys'].to(self.device))
            self.prompt_values = nn.Parameter(prompt_pool_dict['prompt_values'].to(self.device))
        else:
            self.prompt_keys = None
            self.prompt_values = None

        self.num_prompts = prompt_pool_dict['num_prompts']
        self.feature_dim = prompt_pool_dict['feature_dim']
        self.community_info = prompt_pool_dict.get('community_info', [])

        # 加载使用统计
        if 'prompt_usage_stats' in prompt_pool_dict and prompt_pool_dict['prompt_usage_stats'] is not None:
            self.prompt_usage_stats = prompt_pool_dict['prompt_usage_stats'].to(self.device)
        else:
            self.prompt_usage_stats = torch.zeros(self.num_prompts,
                                                  device=self.device) if self.num_prompts > 0 else None

        # 加载权重配置
        self.task_alignment_weight = prompt_pool_dict.get('task_alignment_weight', 0.8)
        self.usage_efficiency_weight = prompt_pool_dict.get('usage_efficiency_weight', 0.2)
        self.key_value_consistency_weight = prompt_pool_dict.get('key_value_consistency_weight', 0.1)
        self.key_feature_similarity_weight = prompt_pool_dict.get('key_feature_similarity_weight', 0.15)
        self.max_prompts = prompt_pool_dict.get('max_prompts', 200)
        self.similarity_threshold = prompt_pool_dict.get('similarity_threshold', 0.6)

    def enhance_features(self, features, top_k=5):
        """向后兼容的接口"""
        return self.forward(features, top_k=top_k, return_attention=False)

    def get_prompt_statistics(self):
        """获取prompt pool的统计信息"""
        stats = {
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'max_prompts': self.max_prompts
        }

        if self.prompt_usage_stats is not None:
            usage_stats = self.prompt_usage_stats
            stats.update({
                'most_used_prompt_usage': usage_stats.max().item(),
                'least_used_prompt_usage': usage_stats.min().item(),
                'average_usage': usage_stats.mean().item(),
                'unused_prompts': (usage_stats == 0).sum().item(),
                'usage_std': usage_stats.std().item()
            })

        if self.community_info:
            purities = [info['purity'] for info in self.community_info]
            sizes = [info['size'] for info in self.community_info]
            stats.update({
                'avg_community_purity': np.mean(purities),
                'avg_community_size': np.mean(sizes),
                'num_communities': len(self.community_info)
            })

        return stats

    def adaptive_load_state_dict(self, state_dict_subset):
        """加载state_dict中的prompt pool参数，处理大小不匹配的情况"""
        try:
            if 'prompt_keys' in state_dict_subset and 'prompt_values' in state_dict_subset:
                prev_keys = state_dict_subset['prompt_keys']
                prev_values = state_dict_subset['prompt_values']
                prev_size = prev_keys.size(0)
                current_size = self.prompt_keys.size(0) if self.prompt_keys is not None else 0

                print(f"Loading prompt pool: prev_size={prev_size}, current_size={current_size}")

                # 情况1: 加载的prompt pool比当前的小或相等
                if prev_size <= current_size:
                    self.prompt_keys.data[:prev_size].copy_(prev_keys.to(self.device))
                    self.prompt_values.data[:prev_size].copy_(prev_values.to(self.device))
                    print(f"Loaded {prev_size} prompts, keeping {current_size - prev_size} existing prompts")

                # 情况2: 加载的prompt pool比当前的大
                else:
                    if current_size > 0:
                        # 只加载能装下的部分
                        self.prompt_keys.data.copy_(prev_keys[:current_size].to(self.device))
                        self.prompt_values.data.copy_(prev_values[:current_size].to(self.device))
                        print(f"Truncated loading: loaded {current_size} out of {prev_size} prompts")
                    else:
                        # 当前没有prompts，需要重新创建
                        # 但这种情况下应该限制在max_prompts范围内
                        actual_size = min(prev_size, getattr(self, 'max_prompts', 200))

                        self.prompt_keys = nn.Parameter(
                            prev_keys[:actual_size].clone().detach().to(self.device)
                        )
                        self.prompt_values = nn.Parameter(
                            prev_values[:actual_size].clone().detach().to(self.device)
                        )
                        self.num_prompts = actual_size
                        print(f"Recreated prompt pool with {actual_size} prompts")

                # 更新使用统计
                if 'prompt_usage_stats' in state_dict_subset:
                    usage_stats = state_dict_subset['prompt_usage_stats']
                    current_stats_size = self.num_prompts

                    if hasattr(self, 'prompt_usage_stats') and self.prompt_usage_stats is not None:
                        if len(usage_stats) >= current_stats_size:
                            self.prompt_usage_stats.data.copy_(usage_stats[:current_stats_size].to(self.device))
                        else:
                            # 扩展使用统计
                            self.prompt_usage_stats.data.zero_()
                            self.prompt_usage_stats.data[:len(usage_stats)].copy_(usage_stats.to(self.device))
                    else:
                        # 创建新的使用统计
                        self.prompt_usage_stats = torch.zeros(current_stats_size, device=self.device)
                        if len(usage_stats) > 0:
                            copy_size = min(len(usage_stats), current_stats_size)
                            self.prompt_usage_stats[:copy_size].copy_(usage_stats[:copy_size].to(self.device))

                return True

        except Exception as e:
            print(f"Error in adaptive_load_state_dict: {str(e)}")
            return False

        return False

    def _update_usage_stats(self, usage_stats, device):
        """更新使用统计的辅助方法"""
        try:
            current_stats_size = self.num_prompts

            if hasattr(self, 'prompt_usage_stats') and self.prompt_usage_stats is not None:
                if len(usage_stats) >= current_stats_size:
                    self.prompt_usage_stats.data.copy_(usage_stats[:current_stats_size].to(device))
                else:
                    self.prompt_usage_stats.data.zero_()
                    self.prompt_usage_stats.data[:len(usage_stats)].copy_(usage_stats.to(device))
            else:
                self.prompt_usage_stats = torch.zeros(current_stats_size, device=device)
                if len(usage_stats) > 0:
                    copy_size = min(len(usage_stats), current_stats_size)
                    self.prompt_usage_stats[:copy_size].copy_(usage_stats[:copy_size].to(device))

            print(f"✓ Updated usage statistics for {current_stats_size} prompts")

        except Exception as e:
            print(f"Warning: Failed to update usage statistics: {str(e)}")
            self.prompt_usage_stats = torch.zeros(self.num_prompts, device=device)


# 为了兼容性，保留原来的类名
PromptPool = LearnablePromptPool