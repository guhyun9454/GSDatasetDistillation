#!/bin/bash

set -e

export OMP_NUM_THREADS=8       # OpenMP
export MKL_NUM_THREADS=8       # Intel MKL
export OPENBLAS_NUM_THREADS=8  # OpenBLAS
export NUMEXPR_NUM_THREADS=8   # NumExpr
export VECLIB_MAXIMUM_THREADS=8 # macOS
export PYTORCH_NUM_THREADS=8   # PyTorch (native threads)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_IDS=$1
if [ -z "${GPU_IDS}" ]; then
  echo "Usage: $0 <gpu_ids> <config_file> [args...]"
  echo "Example: $0 0,1,2,3 config.yaml --arg1 value1"
  exit 1
fi
shift 1

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export NCCL_IB_DISABLE=1
export PYTHONHASHSEED=0

# GPU exclusive-mode locking removed: on Slurm clusters `--gres=gpu:1` already
# isolates the allocated GPU, and `sudo nvidia-smi` is not permitted for users.


# note batch_train=128 and batch_real=256, could be scaled up if more GPU memory is available
buffer_path="./store_local/buffer"
if [ ! -d "${buffer_path}" ]; then
  buffer_path="./store/buffer"
fi
python DM/main_DM.py --config-name imagenette128 "$@" buffer_path=${buffer_path}

