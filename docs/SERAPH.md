# Running this fork on a Slurm GPU cluster (Seraph)

This fork (`guhyun9454/GSDatasetDistillation`, branch `seraph`) differs from
upstream `j-cyoung/GSDatasetDistillation` by **one change**: the GPU
exclusive-mode locking was removed from the three distillation launchers.

## What changed and why

Upstream launchers ran `sudo nvidia-smi -i <id> -c EXCLUSIVE_PROCESS` (via
`lock_gpu`/`unlock_gpu` + a `trap`) to reserve a GPU on a personal workstation.
On a Slurm cluster this is both **impossible** (regular users have no `sudo`) and
**unnecessary** (`--gres=gpu:1` already gives the job exclusive access to its
allocated GPU). The `lock_gpu`/`unlock_gpu` functions, the `trap`, and the
`lock_gpu` call were deleted from:

- `scripts/distill/distill_DP.sh` (TM)
- `DC/scripts/distill_subset128.sh` (DC)
- `DM/scripts/distill_subset128.sh` (DM)

`set -e` and everything else (env exports, arg parsing, `CUDA_VISIBLE_DEVICES`,
the final `python ... main.py`) are unchanged. The vendored CUDA-toolkit
installer `GaussianImage/gsplat/workflows/cuda/Linux.sh` still contains `sudo`
but is not part of the run path and is left untouched.

## Bring-up on a Seraph account (per account, once)

Each account needs its own independent clone (no scp/symlink between accounts).

```bash
# 1. Clone this fork into the account filesystem
#    ariel/guhyun9454 -> /data/$USER/g/      moana/* -> /ceph_data/$USER/projects/
cd /data/$USER/g/        # adjust per account
git clone -b seraph https://github.com/guhyun9454/GSDatasetDistillation.git
cd GSDatasetDistillation

# 2. Create / activate the conda env and install Python deps
conda create -n gsdd python=3.10 -y && conda activate gsdd
pip install -r requirements.txt

# 3. Build the CUDA rasterizer ONCE, on a GPU node (compiles .cu kernels).
#    Do NOT build on the login node. Use an interactive debug allocation:
srun --partition=debug_ugrad --nodelist=ariel-v6 --gres=gpu:1 --pty bash
#    (moana: --partition=debug_ce_ugrad --nodelist=moana-y1)
conda activate gsdd
cd GaussianImage/gsplat && bash build_fast.sh   # A5000/3090 are sm_86, auto-detected
exit
```

## Launching a distillation under sbatch

The launchers take the GPU id list as `$1` and set `CUDA_VISIBLE_DEVICES` from
it. Under Slurm the allocated GPU is index `0` from the job's view, so pass `0`:

```bash
cd /data/$USER/g/GSDatasetDistillation && mkdir -p logs
sbatch --job-name=gsdd-dc --nodelist=ariel-v6 --gres=gpu:1 \
       --cpus-per-gpu=8 --mem-per-gpu=32G --time=6-0 \
       --partition=batch_ugrad -o logs/%j.out \
       --wrap="source /data/\$USER/anaconda3/etc/profile.d/conda.sh && conda activate gsdd && bash DC/scripts/distill_subset128.sh 0 imagenette"
```

See the project's `seraph` operational guide for node/partition/QOS details.
