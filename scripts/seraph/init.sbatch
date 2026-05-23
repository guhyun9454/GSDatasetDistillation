#!/bin/bash
# GSDD Gaussian init for one subset (paper-faithful gpc=200 -> num_points=54).
# Init is single-device and the SAME init serves both DC and DM (both use
# gpc=200, np=54, res=128). Run this once per subset before its DC/DM distill.
#
# Submit (do NOT submit until ceph paths are final — see SUBMIT.md):
#   sbatch --export=ALL,SUBSET=imagewoof scripts/seraph/init.sbatch
#
# Resource knobs verified 2026-05-23 on ariel (post-migration layout):
#   account=ugrad -> partitions batch_ugrad (infinite) / debug_ugrad (4h).
#   24GB A5000 nodes = ariel-g1..g5 (gpu:8). Do NOT use ariel-k* (high_perf, QOS-blocked).
#   Personal QOS = qos_guhyun9454_2026+ ; re-verify the GPU cap at submit time:
#     sacctmgr -n show qos qos_guhyun9454_2026 format=Name,Flags,MaxTRESPU,GrpTRES
#
#SBATCH --job-name=gi
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00          # batch_ugrad is infinite; explicit 1d cap for init
# Cluster policy requires excluding high_perf nodes (or pinning v/g with -w).
# Excluding k/m/n leaves the 24GB A5000 v-nodes (ariel-v[6-12]) in batch_ugrad.
# Do NOT pin ariel-g* — grad-partition only, not in batch_ugrad.
#SBATCH --exclude=ariel-m1,ariel-m2,ariel-k1,ariel-k2,ariel-n1

set -e
ulimit -n 65535                    # init_gs.py shares many fds; avoids Errno 24

# Source conda with `.` (POSIX) so this works under any shell.
# conda lives on /data (no conda on ceph); the env works regardless of cwd.
. /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate gsdd

# Canonical repo post-migration is on ceph.
cd /ceph_data/$USER/g/GSDatasetDistillation

SUBSET=${SUBSET:-imagenette}
GPC=${GPC:-200}                    # paper DC/DM ImageNet-subset IPC=1 (Tables 12/10)
IPC=${IPC:-1}
RES=128
LR=${LR:-0.01}

# bf16 budget formula (matches scripts/init/subset.sh):
#   total_points = ipc * res^2 * 3 * 2 / 9 ; max_num_points = total_points / gpc
TOTAL_PARAMS=$((IPC*RES*RES*3))
TOTAL_POINTS=$((TOTAL_PARAMS*2/9))
MNP=$((TOTAL_POINTS/GPC))          # gpc=200 -> 54
echo "[init] subset=$SUBSET gpc=$GPC ipc=$IPC -> max_num_points=$MNP"

# NOTE: drop the buggy top-level `subset=` override; the init config keys it as
# `dataset.subset`. batch_size=gpc*10 fits all synthetic images at once; at
# gpc=200 (2000 imgs) this may need tuning / a smaller batch if the init OOMs
# on 24GB — see SUBMIT.md caveat.
python init_gs.py --config-path configs/init/ --config-name imagenette_quant \
  ipc=$IPC gpc=$GPC dataset.subset=$SUBSET \
  batch_size=$((GPC*10)) main.max_tasks_per_gpu=1 main.allow_mixed_classes=true \
  init_num_points=$MNP max_num_points=$MNP version=fix_alias lr=$LR

echo "[init] done -> store/init/quant/imagenet_${SUBSET}_${RES}/ipc${IPC}/GaussianImage_Cholesky_50000_${MNP}_${GPC}_fix_alias_bf16"
