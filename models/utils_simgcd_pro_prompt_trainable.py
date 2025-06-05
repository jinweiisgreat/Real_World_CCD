from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
from models.prompt_enhanced_model_trainable import PromptEnhancedModel

'''
get_kmeans_centroid_for_new_head(): use model_pre to obtain new class head init

compute_prior_old_new_ratio(): use model_cur to predict old and new ratio

both use online session training data of current stage

Updated for compatibility with learnable prompt pool
'''


def get_kmeans_centroid_for_new_head(model, online_session_train_loader, args, device):
    # 确认使用的是PromptEnhancedModel
    assert isinstance(model, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(model)}"

    model.to(device)
    model.eval()

    # 在特征提取时暂时禁用prompt增强，确保使用原始特征进行K-means
    original_prompt_training = model.enable_prompt_training
    model.disable_prompt_learning()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')

    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            # 直接使用backbone提取原始特征，不使用prompt增强
            feats = model.backbone(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    # 恢复原始的prompt训练设置
    if original_prompt_training:
        model.enable_prompt_learning()

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_cur_novel_classes, random_state=0).fit(all_feats)
    centroids_np = kmeans.cluster_centers_
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    with torch.no_grad():
        # 使用projector计算logits
        _, logits = model.projector(centroids)
        max_logits, _ = torch.max(logits, dim=-1)
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)
        new_head = centroids[proto_idx]

    return new_head


def compute_prior_old_new_ratio(model, online_session_train_loader, args, device):
    # 确认使用的是PromptEnhancedModel
    assert isinstance(model, PromptEnhancedModel), f"Expected PromptEnhancedModel but got {type(model)}"

    model.to(device)
    model.eval()

    all_preds_list = []
    args.logger.info('Using model_cur (initialized new heads) to predicting labels...')

    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            # 使用完整的模型进行预测（包括prompt增强）
            _, logits = model(images)
            all_preds_list.append(logits.argmax(1))

    all_preds = torch.cat(all_preds_list, dim=0)
    args.logger.info('Computing prior old and new ratio...')
    pred_prior_old_ratio = len(all_preds[all_preds < args.num_seen_classes]) / len(all_preds)
    pred_prior_new_ratio = len(all_preds[all_preds >= args.num_seen_classes]) / len(all_preds)
    args.logger.info(
        f'Pred prior old ratio: {pred_prior_old_ratio:.4f} | Pred prior new ratio: {pred_prior_new_ratio:.4f}')

    pred_prior_ratio_dict = {
        'pred_prior_old_ratio': pred_prior_old_ratio,
        'pred_prior_new_ratio': pred_prior_new_ratio,
    }

    return pred_prior_ratio_dict


def analyze_prompt_usage(model, data_loader, args, device, logger=None):
    """
    分析prompt使用情况的工具函数

    Args:
        model: PromptEnhancedModel
        data_loader: 数据加载器
        args: 参数
        device: 设备
        logger: 日志记录器

    Returns:
        Dict containing prompt usage statistics
    """
    if not isinstance(model, PromptEnhancedModel):
        return {"error": "Model is not PromptEnhancedModel"}

    if model.prompt_pool is None or model.prompt_pool.num_prompts == 0:
        return {"error": "No prompt pool available"}

    model.eval()

    prompt_usage_counts = torch.zeros(model.prompt_pool.num_prompts)
    attention_weights_sum = torch.zeros(model.prompt_pool.num_prompts)
    enhancement_strengths = []
    max_similarities = []

    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Analyzing prompt usage")):
            try:
                if len(batch) >= 3:
                    if isinstance(batch[0], list):
                        images = batch[0][0].to(device)
                    else:
                        images = batch[0].to(device)
                else:
                    continue

                # 获取prompt相关信息
                _, _, prompt_info = model.forward(images, return_prompt_info=True)

                if prompt_info['enhancement_applied'] and prompt_info['attention_info'] is not None:
                    attention_info = prompt_info['attention_info']

                    # 统计prompt使用情况
                    selected_indices = attention_info['selected_prompt_indices']  # [B, top_k]
                    attention_weights = attention_info['attention_weights']  # [B, top_k]

                    # 更新使用计数
                    for b in range(selected_indices.shape[0]):
                        for k in range(selected_indices.shape[1]):
                            prompt_idx = selected_indices[b, k].item()
                            weight = attention_weights[b, k].item()

                            prompt_usage_counts[prompt_idx] += 1
                            attention_weights_sum[prompt_idx] += weight

                    # 记录增强强度和相似度
                    enhancement_strengths.extend(attention_info['enhancement_strength'].cpu().numpy())
                    max_similarities.extend(attention_info['max_similarity'].cpu().numpy())

                    total_samples += images.shape[0]

            except Exception as e:
                if logger:
                    logger.warning(f"Error in batch {batch_idx}: {str(e)}")
                continue

    # 计算统计信息
    avg_attention_weights = attention_weights_sum / (prompt_usage_counts + 1e-8)

    stats = {
        'total_samples': total_samples,
        'prompt_usage_counts': prompt_usage_counts.numpy(),
        'prompt_usage_rates': (prompt_usage_counts / total_samples).numpy(),
        'avg_attention_weights': avg_attention_weights.numpy(),
        'avg_enhancement_strength': np.mean(enhancement_strengths) if enhancement_strengths else 0,
        'avg_max_similarity': np.mean(max_similarities) if max_similarities else 0,
        'enhancement_strength_std': np.std(enhancement_strengths) if enhancement_strengths else 0,
        'max_similarity_std': np.std(max_similarities) if max_similarities else 0,
        'num_unused_prompts': int((prompt_usage_counts == 0).sum()),
        'most_used_prompt_idx': int(prompt_usage_counts.argmax()),
        'least_used_prompt_idx': int(prompt_usage_counts.argmin()),
    }

    if logger:
        logger.info(f"Prompt Usage Analysis:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Average enhancement strength: {stats['avg_enhancement_strength']:.4f}")
        logger.info(f"  Average max similarity: {stats['avg_max_similarity']:.4f}")
        logger.info(f"  Unused prompts: {stats['num_unused_prompts']}/{model.prompt_pool.num_prompts}")
        logger.info(
            f"  Most used prompt (idx {stats['most_used_prompt_idx']}): {stats['prompt_usage_counts'][stats['most_used_prompt_idx']]} times")

    return stats


def visualize_prompt_attention_patterns(model: object, data_loader: object, args: object, device: object, save_path: object, logger: object = None) -> None:
    """
    可视化prompt注意力模式
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        stats = analyze_prompt_usage(model, data_loader, args, device, logger)

        if 'error' in stats:
            if logger:
                logger.warning(f"Cannot visualize prompt patterns: {stats['error']}")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Prompt使用频率
        axes[0, 0].bar(range(len(stats['prompt_usage_counts'])), stats['prompt_usage_counts'])
        axes[0, 0].set_title('Prompt Usage Frequency')
        axes[0, 0].set_xlabel('Prompt Index')
        axes[0, 0].set_ylabel('Usage Count')

        # 2. 平均注意力权重
        axes[0, 1].bar(range(len(stats['avg_attention_weights'])), stats['avg_attention_weights'])
        axes[0, 1].set_title('Average Attention Weights')
        axes[0, 1].set_xlabel('Prompt Index')
        axes[0, 1].set_ylabel('Average Weight')

        # 3. 使用率分布
        axes[1, 0].hist(stats['prompt_usage_rates'], bins=20, alpha=0.7)
        axes[1, 0].set_title('Prompt Usage Rate Distribution')
        axes[1, 0].set_xlabel('Usage Rate')
        axes[1, 0].set_ylabel('Number of Prompts')

        # 4. 统计信息文本
        axes[1, 1].axis('off')
        stats_text = f"""
        Prompt Statistics:

        Total Prompts: {len(stats['prompt_usage_counts'])}
        Unused Prompts: {stats['num_unused_prompts']}

        Enhancement Strength:
        Mean: {stats['avg_enhancement_strength']:.4f}
        Std: {stats['enhancement_strength_std']:.4f}

        Max Similarity:
        Mean: {stats['avg_max_similarity']:.4f}
        Std: {stats['max_similarity_std']:.4f}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if logger:
            logger.info(f"Prompt attention patterns saved to {save_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to visualize prompt patterns: {str(e)}")


def evaluate_prompt_effectiveness(model_with_prompts, model_without_prompts, data_loader, args, device, logger=None):
    """
    评估prompt的有效性

    Args:
        model_with_prompts: 启用prompt的模型
        model_without_prompts: 禁用prompt的模型（或同样的模型但禁用prompt）
        data_loader: 测试数据加载器
        args: 参数
        device: 设备
        logger: 日志记录器

    Returns:
        Dict containing effectiveness metrics
    """

    def get_predictions(model, enable_prompts=True):
        model.eval()
        if hasattr(model, 'enable_prompt_learning') and hasattr(model, 'disable_prompt_learning'):
            if enable_prompts:
                model.enable_prompt_learning()
            else:
                model.disable_prompt_learning()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {'with' if enable_prompts else 'without'} prompts"):
                if len(batch) >= 2:
                    if isinstance(batch[0], list):
                        images = batch[0][0].to(device)  # 处理多视图情况
                    else:
                        images = batch[0].to(device)
                    targets = batch[1]
                else:
                    continue

                _, logits = model(images)
                preds = logits.argmax(1).cpu()

                all_preds.append(preds)
                all_targets.append(targets)

        return torch.cat(all_preds), torch.cat(all_targets)

    # 获取有prompt和无prompt的预测结果
    preds_with_prompts, targets = get_predictions(model_with_prompts, enable_prompts=True)
    preds_without_prompts, _ = get_predictions(model_without_prompts, enable_prompts=False)

    # 计算准确率
    acc_with_prompts = (preds_with_prompts == targets).float().mean().item()
    acc_without_prompts = (preds_without_prompts == targets).float().mean().item()

    # 计算改进
    improvement = acc_with_prompts - acc_without_prompts
    relative_improvement = improvement / acc_without_prompts if acc_without_prompts > 0 else 0

    # 分析一致性（两个模型预测相同的样本比例）
    consistency = (preds_with_prompts == preds_without_prompts).float().mean().item()

    # 分析prompt帮助的样本（有prompt正确，无prompt错误）
    prompt_helped = ((preds_with_prompts == targets) & (preds_without_prompts != targets)).sum().item()

    # 分析prompt伤害的样本（有prompt错误，无prompt正确）
    prompt_hurt = ((preds_with_prompts != targets) & (preds_without_prompts == targets)).sum().item()

    effectiveness = {
        'accuracy_with_prompts': acc_with_prompts,
        'accuracy_without_prompts': acc_without_prompts,
        'absolute_improvement': improvement,
        'relative_improvement': relative_improvement,
        'consistency': consistency,
        'samples_helped_by_prompts': prompt_helped,
        'samples_hurt_by_prompts': prompt_hurt,
        'net_samples_helped': prompt_helped - prompt_hurt,
        'total_samples': len(targets)
    }

    if logger:
        logger.info(f"Prompt Effectiveness Analysis:")
        logger.info(f"  Accuracy with prompts: {acc_with_prompts:.4f}")
        logger.info(f"  Accuracy without prompts: {acc_without_prompts:.4f}")
        logger.info(f"  Absolute improvement: {improvement:.4f}")
        logger.info(f"  Relative improvement: {relative_improvement:.2%}")
        logger.info(f"  Prediction consistency: {consistency:.4f}")
        logger.info(f"  Samples helped by prompts: {prompt_helped}")
        logger.info(f"  Samples hurt by prompts: {prompt_hurt}")
        logger.info(f"  Net improvement: {prompt_helped - prompt_hurt} samples")

    return effectiveness