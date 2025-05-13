#!/bin/bash

set -e
set -x

# Run training script with parameters
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --transform 'imagenet' \
    --lr 0.1 \
    --memax_weight 1 \
    --eval_funcs 'v2' \
    --num_old_classes 50 \
    --prop_train_labels 0.8 \
    --train_session offline \
    --epochs_offline 100 \
    --continual_session_num 5 \
    --online_novel_unseen_num 400 \
    --online_old_seen_num 25 \
    --online_novel_seen_num 25 \
    --prompt_pool