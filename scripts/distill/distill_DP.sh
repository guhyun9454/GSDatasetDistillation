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
CFG=$2
if [ -z "${CFG}" ]; then
  echo "Error: Configuration file is required."
  exit 1
fi
shift 2

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export NCCL_IB_DISABLE=1
export PYTHONHASHSEED=0

# SCRIPT=distill_gs.py
# MASTER_PORT=29501
# NUM_GPUS=$(echo "${GPU_IDS}" | awk -F, '{print NF}')

# GPU exclusive-mode locking removed: on Slurm clusters `--gres=gpu:1` already
# isolates the allocated GPU, and `sudo nvidia-smi` is not permitted for users.

# # total evaluation rounds = len(GPU) * num_eval
# torchrun \
#   --standalone \
#   --nproc_per_node="${NUM_GPUS}" \
#   --master_port=${MASTER_PORT} \
#   ${SCRIPT} \
#   --config-name "${CFG}" \
#   "$@"
python distill_gs_dp.py --config-name "${CFG}" "$@"

