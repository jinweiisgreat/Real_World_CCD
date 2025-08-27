#!/bin/bash

set -e
set -x

# Run training script with parameters
CUDA_VISIBLE_DEVICES=0 python train_happy.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --transform 'imagenet' \
    --lr 0.1 \
    --memax_weight 10 \
    --eval_funcs 'v2' \
    --num_old_classes 100 \
    --prop_train_labels 0.8 \
    --train_session offline \
    --warmup_teacher_temp_epochs 1 \
    --epochs_offline 50 \
    --continual_session_num 5 \
    --online_novel_unseen_num 25 \
    --online_old_seen_num 5 \
    --online_novel_seen_num 5
#    --prompt_pool \
#    --enable_prompt_training