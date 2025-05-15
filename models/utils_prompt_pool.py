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
import os


def visualize_graph_network(adjacency_matrix, save_path, max_nodes=5000, logger=None,
                            similarity_threshold=None):
    """
    Visualize the graph network created from the adjacency matrix before community detection
    using seaborn for better aesthetics, with options for displaying more nodes and edges.

    Args:
        adjacency_matrix: The adjacency matrix of the graph
        save_path: Path to save the visualization
        max_nodes: Maximum number of nodes to visualize (to avoid overcrowding)
        logger: Logger to log progress
        similarity_threshold: Current similarity threshold (for reference)
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import seaborn as sns

        if logger:
            logger.info(f"Creating enhanced graph network visualization with up to {max_nodes} nodes...")

        # Set seaborn style for better aesthetics
        sns.set(style="whitegrid", context="paper", font_scale=1.2)

        # If the graph is too large, sample nodes
        n = adjacency_matrix.shape[0]
        if n > max_nodes:
            # Sample more nodes to show larger network structure
            indices = np.random.choice(n, max_nodes, replace=False)
            sampled_adj_matrix = adjacency_matrix[indices][:, indices]
            logger.info(f"Sampling {max_nodes} nodes from total {n} nodes for visualization")
        else:
            sampled_adj_matrix = adjacency_matrix
            indices = np.arange(n)
            logger.info(f"Using all {n} nodes for visualization")

        # Create graph from adjacency matrix
        G = nx.from_numpy_array(sampled_adj_matrix)

        # Get initial statistics
        initial_nodes = G.number_of_nodes()
        initial_edges = G.number_of_edges()

        # Get statistics for connected components
        components = list(nx.connected_components(G))

        if logger:
            logger.info(f"Initial graph: {initial_nodes} nodes, {initial_edges} edges")
            logger.info(f"Number of connected components: {len(components)}")
            if len(components) > 0:
                sizes = [len(c) for c in components]
                logger.info(f"Largest component size: {max(sizes)}")
                logger.info(f"Average component size: {sum(sizes) / len(sizes):.2f}")

        # Calculate graph metrics
        if G.number_of_nodes() > 0:  # Ensure the graph is not empty
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            density = nx.density(G)
        else:
            avg_degree = 0
            density = 0
            if logger:
                logger.warning("Graph has no connected nodes")
            return

        # Create a degree histogram visualization
        plt.figure(figsize=(10, 6))
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        degreeCount = {i: degree_sequence.count(i) for i in set(degree_sequence)}

        # Plot degree histogram
        plt.bar(degreeCount.keys(), degreeCount.values(), color='skyblue', alpha=0.8)
        plt.title(f"Node Degree Distribution (similarity threshold: {similarity_threshold})")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.yscale('log')  # Log scale often works better for degree distributions
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        degree_hist_path = os.path.splitext(save_path)[0] + "_degree_hist.png"
        plt.savefig(degree_hist_path, dpi=300)
        plt.close()

        if logger:
            logger.info(f"Degree histogram saved to {degree_hist_path}")

        # Setup main network visualization
        plt.figure(figsize=(16, 14))

        # If network is very large, we can visualize a simplified version
        if initial_nodes > 2000:
            # Identify largest connected components
            largest_components = sorted(components, key=len, reverse=True)

            # Take the top 3 largest components for better visualization
            top_components = largest_components[:3]
            nodes_in_top = sum(len(c) for c in top_components)

            if logger:
                logger.info(f"Visualizing top 3 components with {nodes_in_top} nodes total")

            # Create subgraph of these components
            nodes_to_keep = set().union(*top_components)
            H = G.subgraph(nodes_to_keep)

            # Compute positions - using sfdp layout for better distribution
            pos = nx.spring_layout(
                H,
                k=0.2,  # Optimal distance between nodes (smaller = tighter)
                iterations=50,  # More iterations for better layout
                seed=42
            )

            # Determine node sizes based on degree
            degrees = dict(H.degree())

            # Scale node size inversely with node count to avoid overcrowding
            size_scale = max(1, 1000 / np.sqrt(len(H)))
            node_sizes = [3 + 1.5 * np.sqrt(degrees[n]) * size_scale for n in H.nodes()]

            # Determine edge alpha based on number of edges
            edge_alpha = max(0.05, min(0.3, 50000 / H.number_of_edges()))

            # Draw the network with improved aesthetics
            edges = nx.draw_networkx_edges(
                H, pos,
                width=0.2,
                alpha=edge_alpha,
                edge_color="lightblue"
            )

            # Use a colormap based on degree for nodes
            nodes = nx.draw_networkx_nodes(
                H, pos,
                node_size=node_sizes,
                node_color=[degrees[n] for n in H.nodes()],
                cmap=plt.cm.viridis,
                alpha=0.7
            )

            # Add colorbar for node degree
            plt.colorbar(nodes, label="Node Degree", shrink=0.6)

            # Add title with graph metrics
            plt.title(
                f"Feature Similarity Network (Top 3 Components)\n"
                f"Nodes: {H.number_of_nodes()} (of {initial_nodes}), "
                f"Edges: {H.number_of_edges()} (of {initial_edges})\n"
                f"Avg. Degree: {avg_degree:.2f}, Density: {density:.4f}, "
                f"Similarity Threshold: {similarity_threshold}",
                fontsize=14
            )

        else:
            # For smaller networks, we can visualize everything
            # Compute positions - using spring layout with optimized parameters
            pos = nx.spring_layout(
                G,
                k=0.2,  # Optimal distance between nodes
                iterations=50,  # More iterations for better layout
                seed=42
            )

            # Determine node sizes based on degree
            degrees = dict(G.degree())

            # Scale node size inversely with node count to avoid overcrowding
            size_scale = max(1, 1000 / np.sqrt(len(G)))
            node_sizes = [2 + np.sqrt(degrees[n]) * size_scale for n in G.nodes()]

            # Determine edge alpha based on number of edges
            edge_alpha = max(0.01, min(0.2, 20000 / G.number_of_edges()))

            # Draw the network with improved aesthetics
            edges = nx.draw_networkx_edges(
                G, pos,
                width=0.1,
                alpha=edge_alpha,
                edge_color="lightblue"
            )

            # Use a colormap based on degree for nodes
            nodes = nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=[degrees[n] for n in G.nodes()],
                cmap=plt.cm.viridis,
                alpha=0.7
            )

            # Add colorbar for node degree
            plt.colorbar(nodes, label="Node Degree", shrink=0.6)

            # Add title with graph metrics
            plt.title(
                f"Feature Similarity Network\n"
                f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n"
                f"Avg. Degree: {avg_degree:.2f}, Density: {density:.4f}, "
                f"Similarity Threshold: {similarity_threshold}",
                fontsize=14
            )

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if logger:
            logger.info(f"Enhanced graph network visualization saved to {save_path}")

        # Create a 2D embedding visualization of the feature space (shows clusters without edges)
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            plt.figure(figsize=(14, 12))

            # Step 1: Use PCA to reduce dimensionality to 50D for faster processing
            if all_features.shape[1] > 50:
                pca = PCA(n_components=50)
                reduced_features = pca.fit_transform(all_features[indices])
            else:
                reduced_features = all_features[indices]

            # Step 2: Use t-SNE for final 2D embedding
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            embedding = tsne.fit_transform(reduced_features)

            # Step 3: Color points by their connected component
            colors = []
            component_map = {}

            # Map each node to its component ID
            for i, comp in enumerate(components):
                for node in comp:
                    component_map[node] = i

            # Get top 15 components by size (for coloring)
            top_components = sorted(components, key=len, reverse=True)[:15]
            top_comp_sets = [set(c) for c in top_components]

            # Create a color for each node based on component
            for i in range(len(embedding)):
                node_id = i

                # Check if node is in one of the top components
                assigned = False
                for j, comp_set in enumerate(top_comp_sets):
                    if node_id in comp_set:
                        colors.append(j)
                        assigned = True
                        break

                # If not in a top component, assign a default color
                if not assigned:
                    colors.append(len(top_comp_sets))

            # Plot the embedding with component-based coloring
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                cmap='tab20',
                alpha=0.7,
                s=3  # Small point size to show more points clearly
            )

            plt.title(f"t-SNE Visualization of Feature Space\nColored by Connected Component")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")

            # Save t-SNE visualization
            tsne_path = os.path.splitext(save_path)[0] + "_tsne.png"
            plt.tight_layout()
            plt.savefig(tsne_path, dpi=300)
            plt.close()

            if logger:
                logger.info(f"t-SNE visualization saved to {tsne_path}")

        except Exception as e:
            if logger:
                logger.error(f"Failed to create t-SNE visualization: {str(e)}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create graph visualization: {str(e)}")


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

        # Try different similarity thresholds to find a good balance
        thresholds = [self.similarity_threshold]

        # Optionally add more thresholds to test
        if len(all_features) > 1000:
            additional_thresholds = [
                self.similarity_threshold - 0.1,  # Lower threshold for more connections
                self.similarity_threshold + 0.1,  # Higher threshold for fewer connections
            ]
            thresholds.extend([t for t in additional_thresholds if 0.3 <= t <= 0.9])

        threshold_results = {}
        best_threshold = self.similarity_threshold
        best_components = 0

        for threshold in thresholds:
            # Create adjacency matrix with current threshold
            adjacency_matrix = (similarity_matrix > threshold).astype(np.int8)
            np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

            # Create graph and check component structure
            G = nx.from_numpy_array(adjacency_matrix)
            components = list(nx.connected_components(G))

            # Store results
            threshold_results[threshold] = {
                "num_edges": G.number_of_edges(),
                "num_components": len(components),
                "largest_component": max(len(c) for c in components) if components else 0,
            }

            # Log results
            logger.info(f"Threshold {threshold:.2f}: "
                        f"{G.number_of_edges()} edges, "
                        f"{len(components)} components, "
                        f"largest component: {max(len(c) for c in components) if components else 0} nodes")

            # Select threshold with most components (balance between too connected and too disconnected)
            if len(components) > best_components:
                best_threshold = threshold
                best_components = len(components)

        # Update to best threshold if different
        if best_threshold != self.similarity_threshold:
            logger.info(f"Updating similarity threshold from {self.similarity_threshold} to {best_threshold}")
            self.similarity_threshold = best_threshold

        # Create final adjacency matrix with selected threshold
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(np.int8)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

        # Generate graph visualization before community detection
        log_dir = os.path.dirname(logger.handlers[0].baseFilename) if hasattr(logger,
                                                                              'handlers') and logger.handlers else "."
        vis_path = os.path.join(log_dir, "graph_network_visualization.png")
        visualize_graph_network(adjacency_matrix, vis_path, max_nodes=5000,
                                logger=logger, similarity_threshold=self.similarity_threshold)

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

        # Generate community visualization
        community_vis_path = os.path.join(log_dir, "community_visualization.png")
        try:
            # Set seaborn style for better visualization
            sns.set(style="white", context="paper", palette="viridis", font_scale=1.2)

            # Create a new figure for community visualization
            plt.figure(figsize=(12, 10))

            # Sort nodes by community for better visualization
            community_order = []
            for community_id, nodes in valid_communities.items():
                community_order.extend(nodes)

            if len(community_order) > 0:
                # Reorder the adjacency matrix based on communities
                reordered_adj = adjacency_matrix[community_order, :][:, community_order]

                # Create a visualization with a reasonable size limit
                max_vis_size = min(2000, len(reordered_adj))
                if len(reordered_adj) > max_vis_size:
                    reordered_adj = reordered_adj[:max_vis_size, :max_vis_size]

                # Use seaborn heatmap for better visualization
                ax = sns.heatmap(
                    reordered_adj[:max_vis_size, :max_vis_size],
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=False,
                    cbar_kws={"label": "Connection"}
                )

                plt.title("Community Structure in Adjacency Matrix", fontsize=14)
                plt.xlabel("Node Index (reordered by community)")
                plt.ylabel("Node Index (reordered by community)")

                # Add annotations for community boundaries
                current_idx = 0
                prev_community = None
                community_boundaries = []

                for node_idx in community_order[:max_vis_size]:
                    community_id = partition[node_idx]
                    if community_id != prev_community and prev_community is not None:
                        community_boundaries.append(current_idx)
                    prev_community = community_id
                    current_idx += 1

                # Draw community boundary lines
                for boundary in community_boundaries:
                    if boundary < max_vis_size:
                        plt.axhline(y=boundary, color='r', linestyle='-', alpha=0.3)
                        plt.axvline(x=boundary, color='r', linestyle='-', alpha=0.3)

                plt.tight_layout()
                plt.savefig(community_vis_path, dpi=300)
                plt.close()

                # Generate a summary of community distribution
                plt.figure(figsize=(10, 8))

                # Create a heatmap of label distribution per community
                ax = sns.heatmap(
                    label_distribution,
                    cmap="YlGnBu",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Proportion of Class Samples"}
                )

                plt.title("Class Distribution Across Communities", fontsize=14)
                plt.xlabel("Class ID")
                plt.ylabel("Community ID")

                # Save class distribution visualization
                class_dist_path = os.path.join(log_dir, "community_class_distribution.png")
                plt.tight_layout()
                plt.savefig(class_dist_path, dpi=300)
                plt.close()

                logger.info(f"Community visualization saved to {community_vis_path}")
                logger.info(f"Class distribution visualization saved to {class_dist_path}")
            else:
                logger.warning("No communities to visualize")

        except Exception as e:
            logger.error(f"Failed to create community visualization: {str(e)}")

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