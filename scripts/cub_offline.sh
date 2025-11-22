#!/bin/bash

# cub training script
set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --transform 'imagenet' \
    --lr 0.1 \
    --memax_weight 2 \
    --eval_funcs 'v2' \
    --num_old_classes 100 \
    --prop_train_labels 0.8 \
    --train_session offline \
    --epochs_offline 100 \
    --continual_session_num 10 \
    --online_novel_unseen_num 25 \
    --online_old_seen_num 5 \
    --online_novel_seen_num 5
#    --prompt_pool \
#    --enable_prompt_training




