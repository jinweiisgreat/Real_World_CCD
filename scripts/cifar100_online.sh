#!/bin/bash

# CIFAR-100 training script
set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train_clip.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --transform 'imagenet' \
    --warmup_teacher_temp 0.05 \
    --teacher_temp 0.05 \
    --warmup_teacher_temp_epochs 10 \
    --lr 0.2 \
    --memax_old_new_weight 1 \
    --memax_old_in_weight 1 \
    --memax_new_in_weight 1 \
    --proto_aug_weight 1 \
    --feat_distill_weight 1 \
    --radius_scale 1.0 \
    --hardness_temp 0.1 \
    --eval_funcs 'v2' \
    --num_old_classes 50 \
    --prop_train_labels 0.8 \
    --train_session online \
    --epochs_online_per_session 50 \
    --continual_session_num 5 \
    --online_novel_unseen_num 400 \
    --online_old_seen_num 25 \
    --online_novel_seen_num 25 \
    --init_new_head \
    --load_offline_id Old50_Ratio0.8_20250527-153553 \
    --shuffle_classes \
    --seed 0
#    --prompt_pool\
#    --enable_prompt_training