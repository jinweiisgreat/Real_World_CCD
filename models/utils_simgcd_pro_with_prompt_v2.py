from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm


def get_kmeans_centroid_for_new_head(model, prompts_enhancer, online_session_train_loader, args, device):
    """
    使用增强的模型获取K-means质心用于新分类头初始化

    Args:
        model: 模型 [backbone, projector]
        prompts_enhancer: PromptsEnhancer实例（可能为None）
        online_session_train_loader: 在线会话训练数据加载器
        args: 参数
        device: 设备

    Returns:
        new_head: 新类别的分类头初始化权重
    """
    model.to(device)
    model.eval()

    if prompts_enhancer is not None:
        prompts_enhancer.eval()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')

    # 提取所有特征
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)  # shape: (batch_size*2, 3, 224, 224)

            # 获取backbone特征
            if hasattr(model[0], 'base_vit'):
                # 如果是增强的backbone，直接使用基础ViT获取特征（不使用prompts）
                # 这样可以避免prompts对聚类的影响，获得更纯净的特征用于初始化
                feats = model[0].base_vit(images)  # 使用基础backbone
            else:
                # 如果是普通backbone
                feats = model[0](images)  # backbone

            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    # K-MEANS聚类
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_cur_novel_classes, random_state=0).fit(all_feats)
    centroids_np = kmeans.cluster_centers_  # (60, 768)
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)  # torch.Size([60, 768])

    with torch.no_grad():
        """
        使用旧模型的分类头来预测质心，找到最不确定的质心作为新类别初始化
        这利用了旧模型的"无知"来识别新类！
        """
        if hasattr(model[0], 'base_vit'):
            # 对于增强的backbone，我们需要手动构建特征序列
            # 创建临时的CLS token序列用于projector
            batch_size = centroids.shape[0]
            cls_tokens = model[0].base_vit.cls_token.expand(batch_size, -1, -1)
            centroids_seq = torch.cat([cls_tokens, centroids.unsqueeze(1)], dim=1)

            # 通过projector获取logits
            _, logits = model[1](centroids)  # 直接使用归一化的质心特征
        else:
            # 普通backbone的处理方式
            _, logits = model[1](centroids)  # torch.Size([60, 50])

        max_logits, _ = torch.max(logits, dim=-1)  # torch.Size([60])
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)  # torch.Size([10])
        new_head = centroids[proto_idx]  # torch.Size([10, 768])

    return new_head


def compute_prior_old_new_ratio(model, prompts_enhancer, online_session_train_loader, args, device):
    """
    计算先验的新旧类别比例

    Args:
        model: 当前模型
        prompts_enhancer: PromptsEnhancer实例（可能为None）
        online_session_train_loader: 在线会话训练数据加载器
        args: 参数
        device: 设备

    Returns:
        pred_prior_ratio_dict: 预测的先验比例字典
    """
    model.to(device)
    model.eval()

    if prompts_enhancer is not None:
        prompts_enhancer.eval()

    all_preds_list = []
    args.logger.info('Using enhanced model_cur (initialized new heads) to predict labels...')

    # 提取所有预测
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)

            # 使用完整的增强模型进行预测
            _, logits = model(images)
            all_preds_list.append(logits.argmax(1))

    all_preds = torch.cat(all_preds_list, dim=0)  # NOTE!!!
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