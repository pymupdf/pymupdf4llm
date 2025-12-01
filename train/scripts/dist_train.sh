#!/bin/bash

# Default values (can be overridden by arguments)
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"
WORLD_SIZE="2"
RANK="0"
NCCL_SOCKET_IFNAME="eth0"
CUDA_VISIBLE_DEVICES="0"
CONFIG="train/cfgs/config.yaml"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --master_addr=*)
      MASTER_ADDR="${1#*=}"
      shift
      ;;
    --master_port=*)
      MASTER_PORT="${1#*=}"
      shift
      ;;
    --world_size=*)
      WORLD_SIZE="${1#*=}"
      shift
      ;;
    --rank=*)
      RANK="${1#*=}"
      shift
      ;;
    --nccl_socket_ifname=*)
      NCCL_SOCKET_IFNAME="${1#*=}"
      shift
      ;;
    --cuda_visible_devices=*)
      CUDA_VISIBLE_DEVICES="${1#*=}"
      shift
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: ./dist_train.sh --master_addr=<ADDR> --master_port=<PORT> --world_size=<NODES> --rank=<RANK> --nccl_socket_ifname=<IFNAME> --cuda_visible_devices=<DEVICES> --config=<CONFIG_PATH>"
      echo "Example: ./dist_train.sh --master_addr=192.168.0.191 --master_port=29500 --world_size=2 --rank=0 --nccl_socket_ifname=eno1 --cuda_visible_devices=0 --config=train/cfgs/3_config.yaml"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

# NCCL configuration
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# PyTorch distributed training configuration
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=${WORLD_SIZE}
export RANK=${RANK}

# CUDA configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Python path
export PYTHONPATH=.

# Run torch distributed training
torchrun \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --nproc_per_node=1 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/tools/train_gnn.py ${CONFIG}