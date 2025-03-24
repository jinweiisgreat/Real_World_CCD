#!/bin/bash

# 遇到错误立即退出
set -e

# 打印执行的命令，方便调试
set -x

# 运行训练脚本
CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --dataset_name 'clear10' \
    --batch_size 128 \
    --transform 'imagenet' \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 10 \
    --lr 0.01 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1 \
    --radius_scale 1.0 \
    --hardness_temp 0.1 \
    --eval_funcs 'v2' \
    --num_old_classes 7 \
    --prop_train_labels 1.0 \
    --train_session online \
    --epochs_online_per_session 30 \
    --continual_session_num 3 \
    --online_novel_unseen_num 600 \
    --online_old_seen_num 40 \
    --online_novel_seen_num 50 \
    --init_new_head \
    --load_offline_id Old7_Ratio1.0_20250323-183709 \
    --shuffle_classes \
    --seed 0