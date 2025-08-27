from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm

'''
get_kmeans_centroid_for_new_head(): use model_pre to obtain new class head init

compute_prior_old_new_ratio(): use model_cur to predict old and new ratio

both use online session training data of current stage
'''

def get_kmeans_centroid_for_new_head(model, prompts_enhancer, online_session_train_loader, args, device):

    model.to(device)
    model.eval()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True) # shape: (batch_size*2, 3, 224, 224)
            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model[0](images)   # backbone
            feats = torch.nn.functional.normalize(feats, dim=-1)

            if prompts_enhancer is not None:
                enhanced_features, _ = prompts_enhancer(feats)
                all_feats.append(enhanced_features.cpu().numpy())
            else:
                all_feats.append(feats.cpu().numpy())

            all_feats.append(feats.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    if prompts_enhancer is not None:
        print('Fitting prompt K-Means...')
    else:
        print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes+args.num_cur_novel_classes, random_state=0).fit(all_feats)
    #preds = kmeans.labels_
    centroids_np = kmeans.cluster_centers_   # (60, 768)
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)   # torch.Size([60, 768])
    #centroids = centroids.float()
    with torch.no_grad():
        """
        model[1]内部的last_layer计算质心向量与已知类别原型间的余弦相似度
        这产生logits张量，表示每个质心属于已知类别的可能性
        """
        _, logits = model[1](centroids)   # torch.Size([60, 50]) 从50个已知类分类头去预测60个类（新来了10个）它利用了旧模型的"无知"来识别新类！
        max_logits, _ = torch.max(logits, dim=-1)   # torch.Size([60])
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)   # torch.Size([10]) 当largest=False时，返回最小的k个元素
        new_head = centroids[proto_idx]   # torch.Size([10, 768])

    return new_head

def get_kmeans_centroid_for_new_head_clip(clip_model, projector_pre, attention_fusion, online_session_train_loader, args,
                                          device):
    """
    为CLIP特征定制的K-means新头初始化函数

    Args:
        clip_model: CLIP模型
        projector_pre: 前一个会话的投影头
        attention_fusion: 特征融合模块
        online_session_train_loader: 在线会话训练数据加载器
        args: 参数
        device: 设备

    Returns:
        new_head: 新类别的初始化头部 [num_novel_classes, feat_dim]
    """

    # 设置模型为eval模式
    clip_model.eval()
    projector_pre.eval()
    # attention_fusion.eval()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating CLIP features...')

    # 提取所有特征
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)  # shape: (batch_size, 3, 224, 224)

            # 通过CLIP模型获取融合特征
            all_img_feats, all_txt_feats = clip_model(images)

            # 确保数据类型一致
            all_img_feats = all_img_feats.float()
            all_txt_feats = all_txt_feats.float()

            # 通过weighted_gamma融合图像和文本特征
            fusion_feats = attention_fusion(all_img_feats, all_txt_feats)

            # 归一化特征
            fusion_feats = torch.nn.functional.normalize(fusion_feats, dim=-1)

            # 收集特征
            all_feats.append(fusion_feats.cpu().numpy())

    # -----------------------
    # K-MEANS聚类
    # -----------------------
    args.logger.info('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)  # shape: (total_samples, feat_dim)

    # 聚类数量 = 已有类别数 + 当前会话新类别数
    n_clusters = args.num_labeled_classes + args.num_cur_novel_classes
    args.logger.info(f'K-Means clustering into {n_clusters} clusters...')

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_feats)
    centroids_np = kmeans.cluster_centers_  # shape: (n_clusters, feat_dim)
    args.logger.info('K-Means clustering done!')

    # 转换为tensor并归一化
    centroids = torch.from_numpy(centroids_np).to(device).float()
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    # -----------------------
    # 选择新类别的质心
    # -----------------------
    args.logger.info('Selecting centroids for new classes...')

    with torch.no_grad():
        # 使用前一个会话的投影头来评估质心
        # 这里利用了"已知类别分类器对新类别的无知"来识别新类别
        _, logits = projector_pre(centroids)  # shape: (n_clusters, num_seen_classes)

        # 计算每个质心在已知类别上的最大预测概率
        max_logits, _ = torch.max(logits, dim=-1)  # shape: (n_clusters,)

        # 选择预测概率最小的质心作为新类别的代表
        # 这些质心对应于模型最"不确定"的区域，很可能是新类别
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)

        # 提取新类别的质心
        new_head = centroids[proto_idx]  # shape: (num_novel_classes, feat_dim)

    args.logger.info(f'Selected {len(new_head)} centroids for new class head initialization')
    args.logger.info(f'New head shape: {new_head.shape}')

    return new_head

def compute_prior_old_new_ratio(model, online_session_train_loader, args, device):
    model.to(device)
    model.eval()

    all_preds_list = [] = []
    args.logger.info('Using model_cur (initialized new heads) to predicting labels...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            _, logits = model(images)
            all_preds_list.append(logits.argmax(1))

    all_preds = torch.cat(all_preds_list, dim=0)   # NOTE!!!
    args.logger.info('Computing prior old and new ratio...')
    pred_prior_old_ratio = len(all_preds[all_preds<args.num_seen_classes]) / len(all_preds)
    pred_prior_new_ratio = len(all_preds[all_preds>=args.num_seen_classes]) / len(all_preds)
    args.logger.info(f'Pred prior old ratio: {pred_prior_old_ratio:.4f} | Pred prior new ratio: {pred_prior_new_ratio:.4f}')


    pred_prior_ratio_dict = {
        'pred_prior_old_ratio': pred_prior_old_ratio,
        'pred_prior_new_ratio': pred_prior_new_ratio,
    }

    return pred_prior_ratio_dict
