#!/bin/bash

# nums of gpus
NUM_PROC=$1
shift

# Check if PORT environment variable is set, default use port 29501
MASTER_PORT=${PORT:-29501}

python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$NUM_PROC \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    main.py \
    --data-path dataset/imagenet \
    --output output \
    "$@"

# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/litevla_m.yaml --epoch 100 --use-checkpoint --no-wandb
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/litevla_m.yaml --epoch 300 --use-checkpoint --no-wandb
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m.yaml --epoch 100 --no-wandb
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m.yaml --epoch 100 --no-wandb --resume output/aha_m/20250811003000/ckpt_epoch_33.pth
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m.yaml --epoch 300 --no-wandb