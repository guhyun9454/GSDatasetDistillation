#!/bin/bash
# GSDD paper-faithful distill for one (objective, subset) at IPC=1.
# Parameterized by OBJ in {DC, DM} and SUBSET in the 6 benchmark subsets.
#
# Paper hyperparameters baked in (docs/gsdd/paper_numbers.md):
#   DC = Table 12: gpc=200, num_points=54, batch_real=720, batch_syn=2000, Iter=5000
#   DM = Table 10: gpc=200, num_points=54, batch_real=1024, batch_syn=2000, Iter=20000
# batch_syn=2000 with gpc=200 == "use all 200 syn/class" == batch_syn=0 (the full
# path), so we pass batch_syn=0 to use the fork's exact per-class grad-accum path.
#
# Submit (do NOT submit until ceph paths are final + Phase 2 lands for DC — see SUBMIT.md):
#   sbatch --export=ALL,OBJ=DM,SUBSET=imagewoof scripts/seraph/distill.sbatch
#
# Resource knobs verified 2026-05-23 on ariel (post-migration):
#   account=ugrad -> batch_ugrad (infinite). 24GB A5000 = ariel-g1..g5 (gpu:8).
#   Personal QOS qos_guhyun9454_2026+ ; re-verify GPU cap before requesting gpu:4:
#     sacctmgr -n show qos qos_guhyun9454_2026 format=Name,Flags,MaxTRESPU,GrpTRES
#
#SBATCH --job-name=gd
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:4               # DEFAULT for DM; OVERRIDE per objective at submit:
                                   #   DC -> sbatch --gres=gpu:1 (single-GPU, micro-batched)
                                   #   DM -> sbatch --gres=gpu:4 (DataParallel)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                  # REQUIRED: cluster DefMemPerCPU=100M → only 1.6G
                                   # default, but build_tensor_dataset caches the full
                                   # real set into a CPU tensor (OOM-kills at default).
#SBATCH --time=3-00:00:00          # batch_ugrad is infinite; explicit 3d cap
# Cluster submit policy REQUIRES pinning a v/g node via `-w` at submit time
# (the `-x` high_perf-exclude path is rejected). Pass a free A5000 v-node:
#   sbatch -w ariel-v8 --export=ALL,OBJ=DM,SUBSET=... distill.sbatch
# v-nodes (gpu:8 each) in batch_ugrad = ariel-v[6-12]. Do NOT pin ariel-g*.

set -e
ulimit -n 65535

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 PYTORCH_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1,2,3   # use all 4 allocated GPUs

# conda lives on /data (no conda on ceph); the env works regardless of cwd.
. /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate gsdd
# Canonical repo post-migration is on ceph.
cd /ceph_data/$USER/g/GSDatasetDistillation

OBJ=${OBJ:?set OBJ=DC or OBJ=DM}
SUBSET=${SUBSET:-imagenette}
GPC=${GPC:-200}
NP=${NP:-54}
IPC=${IPC:-1}
RES=128
SEED=${SEED:-0}
BSZ=$((GPC*10))                    # gaussian.batch_size must match the init chunking
                                   # (init wrote mixed/batch_0-$((BSZ-1))); gpc=200 -> 2000

# Init produced by init.sbatch (quant path; the distill configs default to init/dm).
GS_DIR="store/init/quant/imagenet_${SUBSET}_${RES}/ipc${IPC}/GaussianImage_Cholesky_50000_${NP}_${GPC}_fix_alias_bf16"
if [ ! -d "$GS_DIR" ]; then
  echo "ERROR: init dir not found: $GS_DIR  (run init.sbatch for SUBSET=$SUBSET first)"; exit 1
fi

# ---------------------------------------------------------------------------
# MULTI-GPU status (Phase 2 — RESOLVED, see docs/gsdd/phase2_multigpu_design.md):
#   DC  -> SINGLE GPU. Real forward micro-batched (batch_real_micro=128); peak
#          14.84 GiB at the paper config, gradient exact (fp64-verified). Submit
#          with --gres=gpu:1 so 4 DC subsets run concurrently under QOS-4.
#   DM  -> main_DM.py wraps the embed net in nn.DataParallel; submit --gres=gpu:4
#          (batch_real=1024 sharded). READY.
# ---------------------------------------------------------------------------

if [ "$OBJ" = "DC" ]; then
  echo "[distill] DC subset=$SUBSET gpc=$GPC np=$NP batch_real=720 micro=128 Iter=5000"
  python DC/main_DC.py --config-name imagenette \
    subset=$SUBSET gpc=$GPC num_points=$NP \
    batch_real=720 batch_real_micro=128 batch_syn=0 Iteration=5000 eval_it=500 \
    ipc=$IPC res=$RES model=ConvNetD5 zca=false seed=$SEED batch_size=$BSZ \
    ${WANDB:+wandb=true wandb_project=GSDD-DC} \
    gs_dir=$GS_DIR
elif [ "$OBJ" = "DM" ]; then
  echo "[distill] DM subset=$SUBSET gpc=$GPC np=$NP batch_real=1024 Iter=20000"
  python DM/main_DM.py --config-name imagenette128 \
    subset=$SUBSET gpc=$GPC num_points=$NP \
    batch_real=1024 batch_syn=0 Iteration=20000 eval_it=2000 \
    ipc=$IPC res=$RES model=ConvNetD5 zca=false seed=$SEED batch_size=$BSZ \
    ${WANDB:+wandb=true wandb_project=GSDD-DM} \
    gs_dir=$GS_DIR
else
  echo "ERROR: OBJ must be DC or DM (got '$OBJ')"; exit 1
fi
