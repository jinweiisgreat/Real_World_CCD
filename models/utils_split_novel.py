"""
Energy-based Novel Class Detection and Enhanced Distillation Loss
Author: Wei Jin
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os


def compute_energy_scores(model, images, temperature=1.0):
    """
    Compute energy scores for input images using the given model.

    Energy(x) = -T * log(sum(exp(f(x)/T)))
    Lower energy indicates higher confidence (seen class)
    Higher energy indicates lower confidence (unseen class)

    Args:
        model: Pre-trained model from previous session
        images: Input images tensor [B, C, H, W]
        temperature: Temperature scaling parameter

    Returns:
        energy_scores: Energy scores for each image [B]
    """
    model.eval()
    with torch.no_grad():
        # Get logits from model
        if hasattr(model, 'projector'):
            # For PromptEnhancedModel or similar structure
            _, logits = model(images)
        else:
            # For basic model
            logits = model(images)

        # Compute energy scores
        # Energy = -T * log(sum(exp(logits/T)))
        scaled_logits = logits / temperature
        energy_scores = -temperature * torch.logsumexp(scaled_logits, dim=1)

    return energy_scores.cpu().numpy()

'''
def visualize_energy_distribution(all_energy_scores, cluster_labels, separation_stats, save_path, logger=None):
    """
    Visualize the energy distribution of seen vs unseen samples after GMM clustering

    Args:
        all_energy_scores: Array of energy scores for all samples
        cluster_labels: GMM cluster assignments (0 or 1)
        separation_stats: Statistics from energy separation
        save_path: Path to save the visualization
        logger: Logger for info messages
    """
    try:
        # Determine which cluster is seen vs unseen
        cluster_0_mean = all_energy_scores[cluster_labels == 0].mean()
        cluster_1_mean = all_energy_scores[cluster_labels == 1].mean()

        if cluster_0_mean < cluster_1_mean:
            seen_cluster, unseen_cluster = 0, 1
        else:
            seen_cluster, unseen_cluster = 1, 0

        # Separate energy scores by cluster
        seen_energies = all_energy_scores[cluster_labels == seen_cluster]
        unseen_energies = all_energy_scores[cluster_labels == unseen_cluster]

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot histograms
        plt.hist(seen_energies, bins=50, alpha=0.7, color='blue', label=f'Seen Samples (n={len(seen_energies)})',
                 density=True)
        plt.hist(unseen_energies, bins=50, alpha=0.7, color='red', label=f'Unseen Samples (n={len(unseen_energies)})',
                 density=True)

        # Add vertical lines for means
        plt.axvline(seen_energies.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Seen Mean: {seen_energies.mean():.3f}')
        plt.axvline(unseen_energies.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Unseen Mean: {unseen_energies.mean():.3f}')

        # Add statistics text box
        stats_text = f"""Separation Statistics:
Seen: μ={separation_stats['seen_energy_mean']:.3f}, σ={separation_stats['seen_energy_std']:.3f}
Unseen: μ={separation_stats['unseen_energy_mean']:.3f}, σ={separation_stats['unseen_energy_std']:.3f}
Separation Score: {separation_stats['energy_separation_score']:.3f}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Formatting
        plt.xlabel('Energy Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Energy Distribution: Seen vs Unseen Samples after GMM Clustering', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if logger:
            logger.info(f"Energy distribution visualization saved to: {save_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create energy distribution visualization: {str(e)}")
        else:
            print(f"Error creating visualization: {str(e)}")
'''


def visualize_energy_distribution(all_energy_scores, cluster_labels, separation_stats, save_path, logger=None):
    """
    Visualize the energy distribution of seen vs unseen samples after GMM clustering using density plots

    Args:
        all_energy_scores: Array of energy scores for all samples
        cluster_labels: GMM cluster assignments (0 or 1)
        separation_stats: Statistics from energy separation
        save_path: Path to save the visualization
        logger: Logger for info messages
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # Determine which cluster is seen vs unseen
        cluster_0_mean = all_energy_scores[cluster_labels == 0].mean()
        cluster_1_mean = all_energy_scores[cluster_labels == 1].mean()

        if cluster_0_mean < cluster_1_mean:
            seen_cluster, unseen_cluster = 0, 1
        else:
            seen_cluster, unseen_cluster = 1, 0

        # Separate energy scores by cluster
        seen_energies = all_energy_scores[cluster_labels == seen_cluster]
        unseen_energies = all_energy_scores[cluster_labels == unseen_cluster]

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create density estimation
        # Generate x range for smooth curves
        x_min = min(all_energy_scores.min(), seen_energies.min(), unseen_energies.min())
        x_max = max(all_energy_scores.max(), seen_energies.max(), unseen_energies.max())
        x_range = np.linspace(x_min, x_max, 300)

        # Calculate KDE for seen samples
        if len(seen_energies) > 1:
            seen_kde = gaussian_kde(seen_energies)
            seen_density = seen_kde(x_range)
        else:
            seen_density = np.zeros_like(x_range)

        # Calculate KDE for unseen samples
        if len(unseen_energies) > 1:
            unseen_kde = gaussian_kde(unseen_energies)
            unseen_density = unseen_kde(x_range)
        else:
            unseen_density = np.zeros_like(x_range)

        # Find the boundary between seen and unseen clusters
        # Use the midpoint between the two means as the boundary
        boundary = (seen_energies.mean() + unseen_energies.mean()) / 2

        # Create masks for truncating density curves at the boundary
        if seen_energies.mean() < unseen_energies.mean():
            # Seen cluster has lower energy, truncate seen curve at boundary (right side)
            # and unseen curve at boundary (left side)
            seen_mask = x_range <= boundary
            unseen_mask = x_range >= boundary
        else:
            # Seen cluster has higher energy, truncate seen curve at boundary (left side)
            # and unseen curve at boundary (right side)
            seen_mask = x_range >= boundary
            unseen_mask = x_range <= boundary

        # Apply masks to density curves and x_range
        seen_x = x_range[seen_mask]
        seen_density_truncated = seen_density[seen_mask]
        unseen_x = x_range[unseen_mask]
        unseen_density_truncated = unseen_density[unseen_mask]

        # Plot truncated density curves with fill
        plt.fill_between(seen_x, seen_density_truncated, alpha=0.6, color='blue',
                         label=f'Seen Samples (n={len(seen_energies)})')
        plt.fill_between(unseen_x, unseen_density_truncated, alpha=0.6, color='red',
                         label=f'Unseen Samples (n={len(unseen_energies)})')

        # Plot truncated density curves as lines for better visibility
        plt.plot(seen_x, seen_density_truncated, color='blue', linewidth=2, alpha=0.9)
        plt.plot(unseen_x, unseen_density_truncated, color='red', linewidth=2, alpha=0.9)

        # Add boundary line
        plt.axvline(boundary, color='black', linestyle=':', linewidth=2, alpha=0.8,
                    label=f'Cluster Boundary: {boundary:.3f}')

        # Add vertical lines for means
        plt.axvline(seen_energies.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Seen Mean: {seen_energies.mean():.3f}', alpha=0.8)
        plt.axvline(unseen_energies.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Unseen Mean: {unseen_energies.mean():.3f}', alpha=0.8)

        # Add statistics text box
        stats_text = f"""Separation Statistics:
Seen: μ={separation_stats['seen_energy_mean']:.3f}, σ={separation_stats['seen_energy_std']:.3f}
Unseen: μ={separation_stats['unseen_energy_mean']:.3f}, σ={separation_stats['unseen_energy_std']:.3f}
Separation Score: {separation_stats['energy_separation_score']:.3f}
Cluster Boundary: {boundary:.3f}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Formatting
        plt.xlabel('Energy Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Energy Distribution: Seen vs Unseen Samples after GMM Clustering', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if logger:
            logger.info(f"Energy distribution visualization saved to: {save_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to create energy distribution visualization: {str(e)}")
        else:
            print(f"Error creating visualization: {str(e)}")

def energy_based_sample_separation(model_prev, dataloader, args):
    """
    Separate samples into seen and unseen classes using energy-based method + GMM

    Args:
        model_prev: Model from previous session
        dataloader: Current session dataloader
        args: Arguments containing configuration

    Returns:
        seen_indices: Set of indices for seen samples
        unseen_indices: Set of indices for unseen samples
        separation_stats: Statistics about the separation
    """
    args.logger.info("Starting energy-based sample separation...")
    args.logger.info("Expected data distribution:")
    args.logger.info(f"  - Seen classes: {args.num_seen_classes} classes × 25 samples ≈ {args.num_seen_classes * 25} samples")
    args.logger.info(f"  - Novel classes: {args.num_novel_class_per_session} classes × 400 samples = {args.num_novel_class_per_session * 400} samples")

    model_prev.eval()

    # Step 1: Compute energy scores for all samples
    all_energy_scores = []
    all_uq_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing energy scores")):
            images, labels, uq_idxs, _ = batch
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            # 确保 uq_idxs 和 labels 的长度与 images 一致
            uq_idxs = torch.cat([uq_idxs, uq_idxs], dim=0)
            labels = torch.cat([labels, labels], dim=0)

            # 计算能量分数
            energy_scores = compute_energy_scores(model_prev, images, temperature=0.1) # Lower temp for better separation

            all_energy_scores.extend(energy_scores)
            all_uq_indices.extend(uq_idxs.numpy())
            all_labels.extend(labels.numpy())

    all_energy_scores = np.array(all_energy_scores)
    all_uq_indices = np.array(all_uq_indices)
    all_labels = np.array(all_labels)

    args.logger.info(f"Computed energy scores for {len(all_energy_scores)} samples")
    args.logger.info(f"Energy score range: [{all_energy_scores.min():.4f}, {all_energy_scores.max():.4f}]")

    # Step 2: Calculate expected ratios
    expected_seen_count = args.num_seen_classes * 25
    expected_unseen_count = args.num_novel_class_per_session * 400
    total_expected = expected_seen_count + expected_unseen_count
    expected_seen_ratio = expected_seen_count / total_expected

    args.logger.info(
        f"Expected ratios: Seen={expected_seen_ratio:.3f} ({expected_seen_count}), Unseen={1 - expected_seen_ratio:.3f} ({expected_unseen_count})")

    # Step 3: Use threshold-based separation instead of pure GMM
    # Calculate threshold based on expected ratio
    energy_threshold = np.percentile(all_energy_scores, expected_seen_ratio * 100)

    # Apply threshold
    threshold_seen_mask = all_energy_scores <= energy_threshold
    threshold_unseen_mask = all_energy_scores > energy_threshold

    threshold_seen_indices = set(all_uq_indices[threshold_seen_mask])
    threshold_unseen_indices = set(all_uq_indices[threshold_unseen_mask])

    args.logger.info(f"Threshold-based separation (threshold={energy_threshold:.4f}):")
    args.logger.info(f"  - Seen: {len(threshold_seen_indices)} samples")
    args.logger.info(f"  - Unseen: {len(threshold_unseen_indices)} samples")

    # Step 4: Fit GMM on energy scores
    args.logger.info("Fitting GMM on energy scores...")

    # Reshape for sklearn
    energy_features = all_energy_scores.reshape(-1, 1)

    # Fit GMM with 2 components (seen vs unseen)
    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='diag', max_iter=200, tol=1e-6)
    gmm.fit(energy_features)

    # Predict cluster assignments
    cluster_labels = gmm.predict(energy_features)
    cluster_probs = gmm.predict_proba(energy_features)

    # Step 3: Determine which cluster corresponds to seen vs unseen
    # Lower energy cluster should be "seen", higher energy cluster should be "unseen"
    cluster_0_mean_energy = all_energy_scores[cluster_labels == 0].mean()
    cluster_1_mean_energy = all_energy_scores[cluster_labels == 1].mean()

    if cluster_0_mean_energy < cluster_1_mean_energy:
        seen_cluster = 0
        unseen_cluster = 1
    else:
        seen_cluster = 1
        unseen_cluster = 0

    gmm_boundary = (gmm.means_[0, 0] + gmm.means_[1, 0]) / 2

    # Step 5: Adaptive threshold adjustment
    # If we want more novel classes, shift threshold left (lower values)
    # Base threshold on expected ratio, but allow adaptive adjustment

    # Strategy 1: Use percentile directly (most aggressive for getting more novel classes)
    adaptive_threshold = energy_threshold

    # Strategy 2: If you want to be more conservative, blend with GMM boundary
    # adaptive_threshold = 0.7 * energy_threshold + 0.3 * gmm_boundary

    # Strategy 3: For maximum novel detection, shift threshold further left
    threshold_shift = -0.004  # Shift left by 0.002 to get more novel classes
    adaptive_threshold = energy_threshold + threshold_shift

    args.logger.info(f"Threshold adjustment:")
    args.logger.info(f"  - Original percentile threshold: {energy_threshold:.4f}")
    args.logger.info(f"  - GMM boundary: {gmm_boundary:.4f}")
    args.logger.info(f"  - Adaptive threshold (shifted): {adaptive_threshold:.4f}")

    # Apply adaptive threshold
    final_seen_mask = all_energy_scores <= adaptive_threshold
    final_unseen_mask = all_energy_scores > adaptive_threshold

    seen_indices = set(all_uq_indices[final_seen_mask])
    unseen_indices = set(all_uq_indices[final_unseen_mask])

    # Step 6: Compute separation statistics
    seen_energy_mean = all_energy_scores[final_seen_mask].mean()
    seen_energy_std = all_energy_scores[final_seen_mask].std()
    unseen_energy_mean = all_energy_scores[final_unseen_mask].mean()
    unseen_energy_std = all_energy_scores[final_unseen_mask].std()

    # # Step 4: Get indices for seen and unseen samples
    # seen_mask = cluster_labels == seen_cluster
    # unseen_mask = cluster_labels == unseen_cluster
    # seen_indices = set(all_uq_indices[seen_mask])
    # unseen_indices = set(all_uq_indices[unseen_mask])

    # # Step 5: Compute separation statistics
    # seen_energy_mean = all_energy_scores[seen_mask].mean()
    # seen_energy_std = all_energy_scores[seen_mask].std()
    # unseen_energy_mean = all_energy_scores[unseen_mask].mean()
    # unseen_energy_std = all_energy_scores[unseen_mask].std()

    # Compute separation quality (larger is better)
    energy_separation = abs(seen_energy_mean - unseen_energy_mean) / (seen_energy_std + unseen_energy_std)

    # Check if separation aligns with expected distribution
    actual_seen_ratio = len(seen_indices) / len(all_energy_scores)
    actual_unseen_ratio = len(unseen_indices) / len(all_energy_scores)
    ratio_difference = abs(actual_seen_ratio - expected_seen_ratio)

    separation_stats = {
        'num_seen': len(seen_indices),
        'num_unseen': len(unseen_indices),
        'seen_energy_mean': seen_energy_mean,
        'seen_energy_std': seen_energy_std,
        'unseen_energy_mean': unseen_energy_mean,
        'unseen_energy_std': unseen_energy_std,
        'energy_separation_score': energy_separation,
        'expected_seen_ratio': expected_seen_ratio,
        'actual_seen_ratio': actual_seen_ratio,
        'actual_unseen_ratio': actual_unseen_ratio,
        'ratio_difference': ratio_difference,
        'original_threshold': energy_threshold,
        'gmm_boundary': gmm_boundary,
        'adaptive_threshold': adaptive_threshold,
        'threshold_shift': threshold_shift,
        'gmm_means': gmm.means_.flatten(),
        'gmm_covariances': gmm.covariances_.flatten()
    }

    # Step 6: Create visualization
    if hasattr(args, 'log_dir'):
        # Try to determine current session number for naming
        current_session = 1
        if hasattr(args, 'num_seen_classes') and hasattr(args, 'num_labeled_classes') and hasattr(args,
                                                                                                  'num_novel_class_per_session'):
            if args.num_seen_classes > args.num_labeled_classes:
                completed_sessions = (
                                                 args.num_seen_classes - args.num_labeled_classes) // args.num_novel_class_per_session
                current_session = completed_sessions + 1

        # Create visualizations directory
        vis_dir = os.path.join(args.log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Save energy distribution plot
        energy_plot_path = os.path.join(vis_dir, f'energy_distribution_session_{current_session}.png')
        visualize_energy_distribution(all_energy_scores, cluster_labels, separation_stats, energy_plot_path, args.logger)

    # Log statistics
    args.logger.info(f"Sample separation completed:")
    args.logger.info(f"  Seen samples: {len(seen_indices)} (energy: {seen_energy_mean:.4f} ± {seen_energy_std:.4f})")
    args.logger.info(f"  Unseen samples: {len(unseen_indices)} (energy: {unseen_energy_mean:.4f} ± {unseen_energy_std:.4f})")
    args.logger.info(f"  Energy separation score: {energy_separation:.4f}")

    # Success metrics
    unseen_retrieval_rate = len(unseen_indices) / expected_unseen_count if expected_unseen_count > 0 else 0
    args.logger.info(f"  Novel class retrieval rate: {unseen_retrieval_rate:.1%} ({len(unseen_indices)}/{expected_unseen_count})")

    # Validate separation quality
    if ratio_difference > 0.10:  # If difference > 10%
        args.logger.warning(f"Energy separation may still need adjustment! Ratio difference: {ratio_difference:.3f}")
    else:
        args.logger.info("✓ Energy separation ratio looks good!")

    if unseen_retrieval_rate > 0.95:  # If we got >95% of expected novel samples
        args.logger.info("✓ Excellent novel class retrieval!")
    elif unseen_retrieval_rate > 0.85:
        args.logger.info("✓ Good novel class retrieval!")
    else:
        args.logger.warning(f"Low novel class retrieval rate: {unseen_retrieval_rate:.1%}")

    return seen_indices, unseen_indices, separation_stats


class EnhancedDistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1, sinkhorn=0.2):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.sinkhorn = sinkhorn

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        sk = SinkhornKnopp()
        alpha = self.sinkhorn
        logits_sk = sk(teacher_output / temp)
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = (1 - alpha) * teacher_out + alpha * logits_sk
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.iter = 0

    @torch.no_grad()
    def forward(self, logits):
        # Q = torch.exp(logits / self.epsilon).t()
        Q = torch.exp(logits).t()
        B = Q.shape[0]
        K = Q.shape[1]  # how many prototypes
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


def compute_sample_type_loss(student_proj, student_out, teacher_out, sample_types,
                             criterion_seen, criterion_unseen, epoch):
    """
    Compute loss based on sample types (seen vs unseen)

    Args:
        student_proj: Student projection features
        student_out: Student output logits
        teacher_out: Teacher output logits  
        sample_types: List of sample types ('seen' or 'unseen')
        criterion_seen: DistillLoss for seen samples
        criterion_unseen: EnhancedDistillLoss for unseen samples
        epoch: Current epoch

    Returns:
        total_loss: Combined loss
        loss_info: Dictionary with loss components
    """
    device = student_out.device

    # Convert sample_types to masks
    seen_mask = torch.tensor([t == 'seen' for t in sample_types], device=device)
    unseen_mask = torch.tensor([t == 'unseen' for t in sample_types], device=device)

    total_loss = 0
    loss_info = {}

    # Process seen samples
    if seen_mask.any():
        # Get seen samples (both views)
        seen_indices = torch.where(seen_mask)[0]
        seen_student_out = student_out.chunk(2)
        seen_teacher_out = teacher_out.chunk(2)

        seen_student_combined = torch.cat([chunk[seen_indices] for chunk in seen_student_out], dim=0)
        seen_teacher_combined = torch.cat([chunk[seen_indices] for chunk in seen_teacher_out], dim=0)

        if len(seen_student_combined) > 0:
            seen_loss = criterion_seen(seen_student_combined, seen_teacher_combined, epoch)
            total_loss += seen_loss
            loss_info['seen_loss'] = seen_loss.item()

    # Process unseen samples  
    if unseen_mask.any():
        # Get unseen samples (both views)
        unseen_indices = torch.where(unseen_mask)[0]
        unseen_student_out = student_out.chunk(2)
        unseen_teacher_out = teacher_out.chunk(2)

        unseen_student_combined = torch.cat([chunk[unseen_indices] for chunk in unseen_student_out], dim=0)
        unseen_teacher_combined = torch.cat([chunk[unseen_indices] for chunk in unseen_teacher_out], dim=0)

        if len(unseen_student_combined) > 0:
            unseen_loss = criterion_unseen(unseen_student_combined, unseen_teacher_combined, epoch)
            total_loss += unseen_loss
            loss_info['unseen_loss'] = unseen_loss.item()

    return total_loss, loss_info