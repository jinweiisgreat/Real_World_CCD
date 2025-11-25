"""
Modified utils_prompt_pool_trainable.py
使用CLIP ViT分支进行无训练的特征提取来生成prompt pool
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
import warnings
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings('ignore')


class CLIPFeatureExtractor:
    """CLIP视觉特征提取器"""

    def __init__(self, device='cuda'):
        model_path = "/home/ps/_jinwei/CLIP_L14"
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded CLIP model: {model_path}")

    def extract_features_from_dataloader(self, data_loader, max_samples=None):
        """从数据加载器提取CLIP特征"""
        all_features = []
        all_labels = []
        sample_count = 0

        print("Extracting CLIP features...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting CLIP features")):
                # 处理不同格式的batch
                if len(batch) == 2:
                    images, labels = batch
                elif len(batch) >= 3:
                    images, labels, _ = batch[:3]
                else:
                    continue

                # 检查样本数量限制
                if max_samples and sample_count >= max_samples:
                    break

                images = images.to(self.device)

                # 使用CLIP提取特征
                features = self.model.encode_image(images)
                features = F.normalize(features, dim=-1)

                all_features.append(features.cpu())
                all_labels.append(labels)

                sample_count += len(images)

        if not all_features:
            raise ValueError("No features extracted! Check your data loader.")

        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        print(f"Extracted {len(all_features)} CLIP features of dimension {all_features.shape[1]}")
        return all_features, all_labels


class LearnablePromptPool(nn.Module):
    """可学习的Prompt Pool - 修改为使用CLIP特征"""

    def __init__(self, feature_dim, similarity_threshold=0.6, community_ratio=2.0,
                 device='cuda', max_prompts=200, num_heads=8, clip_model="ViT-B/32"):
        super().__init__()

        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.community_ratio = community_ratio
        self.device = device
        self.max_prompts = max_prompts
        self.clip_model_name = clip_model

        # 初始化CLIP特征提取器
        self.clip_extractor = None
        if clip is not None:
            try:
                self.clip_extractor = CLIPFeatureExtractor(clip_model, device)
                # 注意：CLIP特征维度可能与原始feature_dim不同
                self.clip_feature_dim = self.clip_extractor.feature_dim
                print(f"CLIP feature dim: {self.clip_feature_dim}, Target feature dim: {feature_dim}")

                # 如果维度不同，添加一个投影层
                if self.clip_feature_dim != feature_dim:
                    self.clip_projection = nn.Linear(self.clip_feature_dim, feature_dim).to(device)
                    print(f"Added projection layer: {self.clip_feature_dim} -> {feature_dim}")
                else:
                    self.clip_projection = None
            except Exception as e:
                print(f"Failed to initialize CLIP extractor: {e}")
                self.clip_extractor = None

        # 可学习参数
        self.prompt_keys = None
        self.prompt_values = None
        self.num_prompts = 0

        # 多头注意力参数
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        # 统计信息
        self.prompt_usage_stats = None
        self.community_info = []

        # 训练相关权重
        self.task_alignment_weight = 0.8
        self.usage_efficiency_weight = 0.2
        self.key_value_consistency_weight = 0.1
        self.key_feature_similarity_weight = 0.15
        self.cross_view_consistency_weight = 0.1

        self._initialize_attention_layers()

    def _initialize_attention_layers(self):
        """初始化注意力层参数"""
        self.query_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.key_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.output_proj = nn.Linear(self.feature_dim, self.feature_dim)

        # 层规范化
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)

    def create_prompt_pool_with_clip(self, data_loader, num_classes, logger, max_samples=None):
        """
        使用CLIP特征提取器创建prompt pool

        Args:
            data_loader: 数据加载器
            num_classes: 类别数量
            logger: 日志记录器
            max_samples: 最大样本数量限制

        Returns:
            prompt_pool_stats: 创建统计信息
        """
        if self.clip_extractor is None:
            logger.warning("CLIP extractor not available, falling back to K-means")
            return self._fallback_kmeans_initialization([], num_classes, logger)

        logger.info("Creating prompt pool using CLIP features and community detection...")

        try:
            # 第一步：使用CLIP提取特征
            all_features, all_labels = self.clip_extractor.extract_features_from_dataloader(
                data_loader, max_samples=max_samples
            )

            logger.info(f"Extracted CLIP features for {len(all_features)} samples")

            # 第二步：如果需要投影到目标维度
            if self.clip_projection is not None:
                logger.info("Projecting CLIP features to target dimension...")
                all_features_tensor = torch.tensor(all_features, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    projected_features = self.clip_projection(all_features_tensor)
                    projected_features = F.normalize(projected_features, dim=-1)
                all_features = projected_features.cpu().numpy()
                logger.info(f"Projected features to dimension {all_features.shape[1]}")

            # 第三步：社区检测
            logger.info("Computing similarity matrix...")
            similarity_matrix = cosine_similarity(all_features)

            # 构建邻接矩阵
            adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)  # 移除自连接

            logger.info(
                f"Adjacency matrix density: {np.sum(adjacency_matrix) / (adjacency_matrix.size - len(adjacency_matrix)):.4f}")

            # 第四步：使用NetworkX进行社区检测
            logger.info("Performing community detection...")
            G = nx.from_numpy_array(adjacency_matrix)

            # 移除孤立节点
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            logger.info(f"Removed {len(isolated_nodes)} isolated nodes")

            if len(G.nodes()) == 0:
                logger.warning("No connected nodes found, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

            # 第五步：社区检测
            communities = community_louvain.best_partition(G, resolution=self.community_ratio, random_state=42)

            # 第六步：分析社区质量并创建prompts
            community_groups = {}
            for node, comm_id in communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(node)

            # 过滤掉太小的社区
            min_community_size = max(5, len(all_features) // 100)
            valid_communities = {k: v for k, v in community_groups.items()
                                 if len(v) >= min_community_size}

            logger.info(f"Found {len(community_groups)} communities, {len(valid_communities)} valid ones")

            if not valid_communities:
                logger.warning("No valid communities found, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

            # 第七步：从社区创建prompts
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

            # 第八步：初始化可学习的prompt参数
            if prompts:
                initial_prompts = torch.tensor(np.array(prompts), dtype=torch.float32).to(self.device)
                self._initialize_learnable_prompts(initial_prompts)
                self.community_info = community_info

                logger.info(f"Successfully created {self.num_prompts} learnable prompts from CLIP-based communities")

                # 计算平均纯度
                avg_purity = np.mean([info['purity'] for info in community_info])
                logger.info(f"Average community purity: {avg_purity:.4f}")

                return {
                    'num_prompts': self.num_prompts,
                    'community_info': community_info,
                    'avg_purity': avg_purity,
                    'adjacency_matrix': adjacency_matrix,
                    'method': 'clip_community_detection',
                    'clip_model': self.clip_model_name,
                    'clip_feature_dim': self.clip_feature_dim,
                    'target_feature_dim': self.feature_dim
                }
            else:
                logger.warning("No prompts created from communities, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

        except Exception as e:
            logger.error(f"Error in CLIP-based prompt pool creation: {e}")
            logger.info("Falling back to K-means initialization")
            return self._fallback_kmeans_initialization(all_features if 'all_features' in locals() else [],
                                                        num_classes, logger)

    def create_prompt_pool(self, model, data_loader, num_classes, logger):
        """
        主要的prompt pool创建接口 - 现在默认使用CLIP特征

        Args:
            model: 训练后的模型（现在主要用于兼容性，实际使用CLIP）
            data_loader: 数据加载器
            num_classes: 类别数量
            logger: 日志记录器

        Returns:
            prompt_pool_stats: 创建统计信息
        """
        logger.info("Creating prompt pool with CLIP features (avoiding backbone contamination)...")

        # 使用CLIP进行特征提取
        return self.create_prompt_pool_with_clip(data_loader, num_classes, logger)

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

            # 初始化使用统计
            self.prompt_usage_stats = torch.zeros(
                self.num_prompts,
                device=self.device,
                dtype=torch.float
            )

            print(f"Initialized {self.num_prompts} learnable prompts with separate Keys and Values")
            print(f"Created usage statistics tracker on device {self.device}")
        else:
            print("No initial prompts provided, skipping initialization")

    def _fallback_kmeans_initialization(self, all_features, num_classes, logger):
        """
        当社区检测失败时的备用K-means初始化方法
        现在也可以使用CLIP特征
        """
        logger.info("Using K-means clustering as fallback initialization...")

        # 如果没有特征，尝试提取CLIP特征
        if len(all_features) == 0 and self.clip_extractor is not None:
            logger.warning("No features provided for K-means, cannot initialize prompt pool")
            return {
                'num_prompts': 0,
                'community_info': [],
                'avg_purity': 0.0,
                'method': 'failed_initialization'
            }

        # 使用K-means聚类
        n_clusters = min(num_classes * 2, 50, len(all_features) // 10)
        n_clusters = max(n_clusters, 5)

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
            'method': 'kmeans_fallback_with_clip' if self.clip_extractor else 'kmeans_fallback'
        }

    def forward(self, features, top_k=5, return_attention=False):
        """
        Forward pass for learnable prompt pool with dual-view support

        Args:
            features: Input features [B*n_views, D]
            top_k: Number of top prompts to use
            return_attention: Whether to return attention weights

        Returns:
            enhanced_features: Enhanced features [B*n_views, D]
            attention_info: (Optional) Attention information
        """
        if self.prompt_keys is None or self.num_prompts == 0:
            if return_attention:
                return features, None
            return features

        batch_size = features.shape[0]

        # 计算features与prompt_keys的相似度
        features_norm = F.normalize(features, dim=-1)
        keys_norm = F.normalize(self.prompt_keys, dim=-1)
        similarities = torch.mm(features_norm, keys_norm.t())  # [B, num_prompts]

        # 选择top-k个最相似的prompts
        top_k = min(top_k, self.num_prompts)
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=-1)  # [B, top_k]

        # 计算注意力权重
        attention_weights = F.softmax(top_similarities / 0.1, dim=-1)  # [B, top_k]

        # 选择对应的prompt values
        selected_values = self.prompt_values[top_indices]  # [B, top_k, D]

        # 加权组合prompt values
        weighted_prompts = torch.sum(
            attention_weights.unsqueeze(-1) * selected_values, dim=1
        )  # [B, D]

        # 特征增强：原始特征 + 加权prompts
        enhanced_features = features + weighted_prompts
        enhanced_features = F.normalize(enhanced_features, dim=-1)

        # 更新使用统计
        if self.training:
            with torch.no_grad():
                for i in range(batch_size):
                    for j in range(top_k):
                        prompt_idx = top_indices[i, j]
                        self.prompt_usage_stats[prompt_idx] += attention_weights[i, j]

        # 返回注意力信息
        attention_info = None
        if return_attention:
            attention_info = {
                'similarities': similarities,
                'top_similarities': top_similarities,
                'top_indices': top_indices,
                'attention_weights': attention_weights,
                'selected_values': selected_values,
                'weighted_prompts': weighted_prompts
            }

        if return_attention:
            return enhanced_features, attention_info
        return enhanced_features

    def compute_prompt_losses(self, features, enhanced_features, attention_info, targets=None, logits=None):
        """
        计算prompt相关的损失函数

        Args:
            features: 原始特征 [B, D]
            enhanced_features: 增强后特征 [B, D]
            attention_info: 注意力信息
            targets: 目标标签 [B] (可选)
            logits: 分类器输出 [B, C] (可选)

        Returns:
            losses: 损失字典
        """
        device = features.device
        losses = {
            'task_alignment': torch.tensor(0.0, device=device),
            'usage_efficiency': torch.tensor(0.0, device=device),
            'key_value_consistency': torch.tensor(0.0, device=device),
            'key_feature_similarity': torch.tensor(0.0, device=device),
            'total': torch.tensor(0.0, device=device)
        }

        if self.prompt_keys is None or attention_info is None:
            return losses

        # 1. 任务对齐损失：增强后的特征应该提高分类性能
        if targets is not None and logits is not None:
            # 计算分类损失的改进
            original_logits = logits  # 假设这是增强后的logits
            task_loss = F.cross_entropy(original_logits, targets)
            losses['task_alignment'] = task_loss * self.task_alignment_weight

        # 2. 使用效率损失：鼓励prompt的均衡使用
        if self.prompt_usage_stats is not None:
            usage_entropy = -torch.sum(
                F.softmax(self.prompt_usage_stats, dim=0) *
                F.log_softmax(self.prompt_usage_stats, dim=0)
            )
            max_entropy = torch.log(torch.tensor(float(self.num_prompts), device=device))
            usage_efficiency = usage_entropy / max_entropy
            losses['usage_efficiency'] = (1.0 - usage_efficiency) * self.usage_efficiency_weight

        # 3. Key-Value一致性损失
        key_value_similarity = F.cosine_similarity(
            self.prompt_keys, self.prompt_values, dim=1
        ).mean()
        losses['key_value_consistency'] = (1.0 - key_value_similarity) * self.key_value_consistency_weight

        # 4. Key-Feature相似度损失：鼓励keys与输入特征的相似性
        if 'top_similarities' in attention_info:
            avg_similarity = attention_info['top_similarities'].mean()
            losses['key_feature_similarity'] = (1.0 - avg_similarity) * self.key_feature_similarity_weight

        # 总损失
        losses['total'] = sum(loss for key, loss in losses.items() if key != 'total')

        return losses

    def get_prompt_parameters(self):
        """获取所有prompt相关的参数"""
        params = []
        if self.prompt_keys is not None:
            params.append(self.prompt_keys)
        if self.prompt_values is not None:
            params.append(self.prompt_values)
        if self.clip_projection is not None:
            params.extend(self.clip_projection.parameters())
        return params

    def save_prompt_pool(self, save_path):
        """保存prompt pool到磁盘"""
        prompt_pool_dict = {
            'prompt_keys': self.prompt_keys.cpu() if self.prompt_keys is not None else None,
            'prompt_values': self.prompt_values.cpu() if self.prompt_values is not None else None,
            'clip_projection': self.clip_projection.state_dict() if self.clip_projection is not None else None,
            'num_prompts': self.num_prompts,
            'feature_dim': self.feature_dim,
            'clip_feature_dim': getattr(self, 'clip_feature_dim', None),
            'clip_model_name': self.clip_model_name,
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

        # 加载CLIP投影层
        if prompt_pool_dict.get('clip_projection') is not None:
            clip_feature_dim = prompt_pool_dict.get('clip_feature_dim', 512)
            if clip_feature_dim != self.feature_dim:
                self.clip_projection = nn.Linear(clip_feature_dim, self.feature_dim).to(self.device)
                self.clip_projection.load_state_dict(prompt_pool_dict['clip_projection'])

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


def visualize_graph_network(adjacency_matrix, communities, labels, save_path=None):
    """可视化图网络和社区结构"""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        # 创建图
        G = nx.from_numpy_array(adjacency_matrix)

        # 设置社区颜色
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(set(communities.values()))))
        node_colors = [community_colors[communities.get(node, 0)] for node in G.nodes()]

        # 布局
        pos = nx.spring_layout(G, k=1, iterations=50)

        # 绘制
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=node_colors, node_size=20, alpha=0.8, edge_color='gray', width=0.1)
        plt.title("Community Detection Results (CLIP Features)")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Matplotlib or NetworkX not available, skipping visualization")
    except Exception as e:
        print(f"Error in visualization: {e}")


# 新增：CLIP特征缓存类，用于提高效率
class CLIPFeatureCache:
    """CLIP特征缓存，避免重复计算"""

    def __init__(self, cache_dir="./clip_feature_cache"):
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, dataset_name, clip_model, num_samples=None):
        """获取缓存文件路径"""
        import os
        cache_name = f"{dataset_name}_{clip_model.replace('/', '_')}"
        if num_samples:
            cache_name += f"_{num_samples}samples"
        cache_name += ".pt"
        return os.path.join(self.cache_dir, cache_name)

    def load_cached_features(self, dataset_name, clip_model, num_samples=None):
        """加载缓存的特征"""
        cache_path = self.get_cache_path(dataset_name, clip_model, num_samples)
        import os
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path)
                print(f"Loaded cached CLIP features from {cache_path}")
                return cached_data['features'], cached_data['labels']
            except Exception as e:
                print(f"Failed to load cached features: {e}")
        return None, None

    def save_features(self, features, labels, dataset_name, clip_model, num_samples=None):
        """保存特征到缓存"""
        cache_path = self.get_cache_path(dataset_name, clip_model, num_samples)
        try:
            torch.save({
                'features': features,
                'labels': labels,
                'dataset_name': dataset_name,
                'clip_model': clip_model,
                'num_samples': num_samples
            }, cache_path)
            print(f"Saved CLIP features to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save features to cache: {e}")


# 新增：增强的CLIP特征提取器，支持缓存和批处理优化
class EnhancedCLIPFeatureExtractor(CLIPFeatureExtractor):
    """增强版CLIP特征提取器，支持缓存和优化"""

    def __init__(self, model_name="ViT-B/32", device="cuda", use_cache=True, cache_dir="./clip_feature_cache"):
        super().__init__(model_name, device)
        self.use_cache = use_cache
        self.feature_cache = CLIPFeatureCache(cache_dir) if use_cache else None

    def extract_features_from_dataloader(self, data_loader, max_samples=None, dataset_name="unknown"):
        """增强版特征提取，支持缓存"""

        # 尝试从缓存加载
        if self.use_cache and self.feature_cache:
            cached_features, cached_labels = self.feature_cache.load_cached_features(
                dataset_name, self.model_name, max_samples
            )
            if cached_features is not None:
                return cached_features, cached_labels

        # 如果没有缓存，进行特征提取
        print(f"Extracting CLIP features for dataset: {dataset_name}")
        features, labels = super().extract_features_from_dataloader(data_loader, max_samples)

        # 保存到缓存
        if self.use_cache and self.feature_cache:
            self.feature_cache.save_features(features, labels, dataset_name, self.model_name, max_samples)

        return features, labels

    def extract_features_with_augmentation(self, data_loader, num_augments=1, max_samples=None, dataset_name="unknown"):
        """提取多个增强版本的特征并平均"""
        all_features_list = []

        for aug_idx in range(num_augments):
            print(f"Extracting features with augmentation {aug_idx + 1}/{num_augments}")
            features, labels = self.extract_features_from_dataloader(
                data_loader, max_samples, f"{dataset_name}_aug{aug_idx}"
            )
            all_features_list.append(features)

        # 平均多个增强版本的特征
        if num_augments > 1:
            averaged_features = np.mean(all_features_list, axis=0)
            print(f"Averaged {num_augments} augmented feature versions")
            return averaged_features, labels
        else:
            return all_features_list[0], labels


# 修改LearnablePromptPool以使用增强版提取器
class LearnablePromptPoolV2(LearnablePromptPool):
    """增强版可学习Prompt Pool"""

    def __init__(self, feature_dim, similarity_threshold=0.6, community_ratio=2.0,
                 device='cuda', max_prompts=200, num_heads=8, clip_model="ViT-B/32",
                 use_cache=True, cache_dir="./clip_feature_cache"):
        super().__init__(feature_dim, similarity_threshold, community_ratio, device,
                         max_prompts, num_heads, clip_model)

        # 使用增强版CLIP提取器
        if clip is not None:
            try:
                self.clip_extractor = EnhancedCLIPFeatureExtractor(
                    clip_model, device, use_cache, cache_dir
                )
                self.clip_feature_dim = self.clip_extractor.feature_dim

                if self.clip_feature_dim != feature_dim:
                    self.clip_projection = nn.Linear(self.clip_feature_dim, feature_dim).to(device)
                    print(f"Added projection layer: {self.clip_feature_dim} -> {feature_dim}")
                else:
                    self.clip_projection = None
            except Exception as e:
                print(f"Failed to initialize enhanced CLIP extractor: {e}")
                self.clip_extractor = None

    def create_prompt_pool_with_clip_v2(self, data_loader, num_classes, logger,
                                        max_samples=None, dataset_name="unknown",
                                        num_augments=1, quality_threshold=0.5):
        """
        增强版CLIP prompt pool创建方法

        Args:
            data_loader: 数据加载器
            num_classes: 类别数量
            logger: 日志记录器
            max_samples: 最大样本数量限制
            dataset_name: 数据集名称（用于缓存）
            num_augments: 数据增强次数
            quality_threshold: 社区质量阈值
        """
        if self.clip_extractor is None:
            logger.warning("Enhanced CLIP extractor not available, falling back to K-means")
            return self._fallback_kmeans_initialization([], num_classes, logger)

        logger.info(f"Creating prompt pool using enhanced CLIP features for {dataset_name}...")

        try:
            # 使用增强版特征提取
            if num_augments > 1:
                all_features, all_labels = self.clip_extractor.extract_features_with_augmentation(
                    data_loader, num_augments, max_samples, dataset_name
                )
            else:
                all_features, all_labels = self.clip_extractor.extract_features_from_dataloader(
                    data_loader, max_samples, dataset_name
                )

            logger.info(f"Extracted CLIP features for {len(all_features)} samples")

            # 维度投影
            if self.clip_projection is not None:
                logger.info("Projecting CLIP features to target dimension...")
                all_features_tensor = torch.tensor(all_features, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    projected_features = self.clip_projection(all_features_tensor)
                    projected_features = F.normalize(projected_features, dim=-1)
                all_features = projected_features.cpu().numpy()

            # 自适应相似度阈值
            similarity_matrix = cosine_similarity(all_features)
            mean_similarity = np.mean(similarity_matrix[similarity_matrix > 0])
            adaptive_threshold = max(self.similarity_threshold, mean_similarity - np.std(similarity_matrix))
            logger.info(f"Using adaptive similarity threshold: {adaptive_threshold:.4f}")

            # 构建邻接矩阵
            adjacency_matrix = (similarity_matrix > adaptive_threshold).astype(int)
            np.fill_diagonal(adjacency_matrix, 0)

            # 社区检测
            G = nx.from_numpy_array(adjacency_matrix)
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)

            if len(G.nodes()) == 0:
                logger.warning("No connected nodes found after filtering, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

            # 多分辨率社区检测
            best_communities = None
            best_modularity = -1

            for resolution in [0.5, 1.0, 1.5, 2.0]:
                communities = community_louvain.best_partition(G, resolution=resolution, random_state=42)
                modularity = community_louvain.modularity(communities, G)

                if modularity > best_modularity:
                    best_modularity = modularity
                    best_communities = communities

            logger.info(f"Best modularity: {best_modularity:.4f}")

            # 处理社区
            community_groups = {}
            for node, comm_id in best_communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(node)

            # 社区质量过滤
            min_community_size = max(5, len(all_features) // 100)
            valid_communities = {}

            for comm_id, nodes in community_groups.items():
                if len(nodes) >= min_community_size:
                    # 计算社区内的标签纯度
                    community_labels = all_labels[nodes]
                    unique_labels, counts = np.unique(community_labels, return_counts=True)
                    purity = np.max(counts) / len(community_labels)

                    if purity >= quality_threshold:
                        valid_communities[comm_id] = nodes

            logger.info(f"Found {len(community_groups)} communities, {len(valid_communities)} high-quality ones")

            if not valid_communities:
                logger.warning("No high-quality communities found, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

            # 创建prompts（按质量排序）
            prompts = []
            community_info = []

            # 计算每个社区的质量分数
            community_scores = []
            for comm_id, node_indices in valid_communities.items():
                community_features = all_features[node_indices]
                community_labels = all_labels[node_indices]

                # 标签纯度
                unique_labels, counts = np.unique(community_labels, return_counts=True)
                purity = np.max(counts) / len(community_labels)

                # 内部紧密度
                intra_similarities = cosine_similarity(community_features)
                cohesion = np.mean(intra_similarities[np.triu_indices_from(intra_similarities, k=1)])

                # 综合分数
                quality_score = 0.7 * purity + 0.3 * cohesion
                community_scores.append((comm_id, quality_score, node_indices))

            # 按质量排序，选择最好的社区
            community_scores.sort(key=lambda x: x[1], reverse=True)
            selected_communities = community_scores[:min(len(community_scores), self.max_prompts)]

            for comm_id, quality_score, node_indices in selected_communities:
                community_features = all_features[node_indices]
                community_labels = all_labels[node_indices]

                # 计算社区中心
                community_prompt = np.mean(community_features, axis=0)
                community_prompt = community_prompt / np.linalg.norm(community_prompt)

                # 标签分布统计
                label_counts = {}
                for label in community_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                purity = label_counts[most_common_label] / len(node_indices)

                prompts.append(community_prompt)
                community_info.append({
                    'community_id': comm_id,
                    'size': len(node_indices),
                    'most_common_label': most_common_label,
                    'purity': purity,
                    'quality_score': quality_score,
                    'label_distribution': label_counts
                })

            # 初始化可学习参数
            if prompts:
                initial_prompts = torch.tensor(np.array(prompts), dtype=torch.float32).to(self.device)
                self._initialize_learnable_prompts(initial_prompts)
                self.community_info = community_info

                avg_purity = np.mean([info['purity'] for info in community_info])
                avg_quality = np.mean([info['quality_score'] for info in community_info])

                logger.info(f"Created {self.num_prompts} high-quality prompts from CLIP-based communities")
                logger.info(f"Average community purity: {avg_purity:.4f}")
                logger.info(f"Average quality score: {avg_quality:.4f}")

                return {
                    'num_prompts': self.num_prompts,
                    'community_info': community_info,
                    'avg_purity': avg_purity,
                    'avg_quality_score': avg_quality,
                    'best_modularity': best_modularity,
                    'adjacency_matrix': adjacency_matrix,
                    'method': 'enhanced_clip_community_detection',
                    'clip_model': self.clip_model_name,
                    'adaptive_threshold': adaptive_threshold,
                    'num_augments': num_augments
                }
            else:
                logger.warning("No prompts created, using K-means fallback")
                return self._fallback_kmeans_initialization(all_features, num_classes, logger)

        except Exception as e:
            logger.error(f"Error in enhanced CLIP-based prompt pool creation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._fallback_kmeans_initialization([], num_classes, logger)


# 导出主要类和函数
__all__ = [
    'LearnablePromptPool',
    'LearnablePromptPoolV2',
    'CLIPFeatureExtractor',
    'EnhancedCLIPFeatureExtractor',
    'CLIPFeatureCache',
    'visualize_graph_network'
]