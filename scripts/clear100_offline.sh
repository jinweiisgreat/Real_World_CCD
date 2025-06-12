#!/bin/bash

# 遇到错误立即退出
set -e

# 打印执行的命令，方便调试
set -x

# 运行训练脚本
CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --dataset_name 'clear100' \
    --batch_size 128 \
    --transform 'imagenet' \
    --lr 0.1 \
    --memax_weight 1 \
    --eval_funcs 'v2' \
    --num_old_classes 7 \
    --prop_train_labels 1.0 \
    --train_session offline \
    --epochs_offline 100 \
    --continual_session_num 3 \
    --online_novel_unseen_num 400 \
    --online_old_seen_num 25 \
    --online_novel_seen_num 25