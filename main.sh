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