#!/bin/bash

# nums of gpus
NUM_PROC=$1
shift

# Check if PORT environment variable is set, default use port 29501
MASTER_PORT=${PORT:-29501}

while lsof -i:"$MASTER_PORT" >/dev/null 2>&1; do
  echo "Port $MASTER_PORT is in use, trying next..."
  MASTER_PORT=$((MASTER_PORT+1))
done
echo "Using port $MASTER_PORT"

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

# now we use --no-log
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m.yaml --epoch 300 --no-log
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m.yaml --epoch 100 --log-method swanlab
# CUDA_VISIBLE_DEVICES=5,4,3,2 bash main.sh 4 -b 256 --cfg configs/aha/aha_m_an0.yaml --epoch 100 --log-method swanlab


# git remote set-url origin https://kkgithub.com/sc-hua/LiteVLA.git
# git remote set-url origin https://github.com/sc-hua/LiteVLA.git
# git pull origin main --no-rebase