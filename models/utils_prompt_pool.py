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