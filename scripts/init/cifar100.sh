#!/bin/bash
ulimit -n 65535  # or even higher like 131072
set -e
GPU_ID=$1

ipc=10
gpc=80
lr=0.01

res=32
total_params=$((ipc*res*res*3))
total_points=$((total_params*2/9))
max_num_points=$((total_points/gpc))
CUDA_VISIBLE_DEVICES=$GPU_ID python init_gs.py --config-path "configs/init/" --config-name "cifar100_bf16" ipc=$ipc gpc=$gpc batch_size=$((gpc*100)) main.max_tasks_per_gpu=1 main.allow_mixed_classes=true init_num_points=$max_num_points max_num_points=$max_num_points version="fix_alias" lr=$lr
