# GSDD paper-faithful IPC=1 run — submit reference (PLAN ONLY)

Scaffold for reproducing GSDD's ImageNet-subset IPC=1 numbers (DC + DM) across
all 6 subsets under paper-faithful settings. **Do NOT submit any job** until the
user confirms the ceph data paths are final and (for DC) the Phase 2 multi-GPU
change has landed. Target numbers: `docs/gsdd/paper_numbers.md`.

All paths/QOS verified on `guhyun9454@ariel.khu.ac.kr -p 30080`. Post-migration
canonical repo is on **ceph**: `/ceph_data/guhyun9454/g/GSDatasetDistillation`,
branch `seraph`, origin = fork. `git pull` there to get these scaffolds. conda
lives on `/data` (`/data/guhyun9454/anaconda3`, env `gsdd`); the env works from
any cwd. QOS = `qos_guhyun9454_2026_1` (GPU cap: re-verify before requesting gpu:4).

## 0. Tarball → subset mapping

Tarballs live at `/nas2/data/guhyun9454/ImageNet/*.tar.gz`. Each archive root is
`ImageNet_<name>/{train,val}/` and **carries the full 1000 sorted ImageNet class
dirs** (verified), with only that subset's 10 classes populated. So
`sorted(listdir(train))[index]` resolves correctly (confirmed end-to-end on woof).

| Benchmark | GSDD config name | Tarball |
|---|---|---|
| Nette   | imagenette  | ImageNet_nette.tar.gz   |
| Woof    | imagewoof   | ImageNet_woof.tar.gz    |
| Fruit   | imagefruit  | ImageNet_fruits.tar.gz  |
| Yellow  | imageyellow | ImageNet_yellow.tar.gz  |
| Meow    | imagemeow   | ImageNet_cats.tar.gz    |
| Squawk  | imagesquawk | ImageNet_birds.tar.gz   |

## 1. Data prep (Phase 1)

Extract each tarball, then build GSDD's clean 10-class layout. Stage on ceph
(`/ceph_data/guhyun9454`, 192T) post-migration; adjust paths as final.

```bash
cd /ceph_data/$USER/g/GSDatasetDistillation
EX=/ceph_data/$USER/gsdd_extract        # extraction scratch on ceph (192T free)
mkdir -p "$EX"
for pair in nette:imagenette woof:imagewoof fruits:imagefruit \
            yellow:imageyellow cats:imagemeow birds:imagesquawk; do
  tar=${pair%%:*}; sub=${pair##*:}
  tar xzf /nas2/data/$USER/ImageNet/ImageNet_${tar}.tar.gz -C "$EX"
  python scripts/seraph/prepare_subsets.py \
      --imagenet "$EX/ImageNet_${tar}" --out store/dataset --subsets "$sub" --mode symlink
done
```

The script enforces the **class-membership gate**: it resolves the 10 synsets by
index, cross-checks them against the bundled canonical `imagenet1k_sorted_wnids.txt`,
verifies each is non-empty in train+val, and hard-fails otherwise. Eyeball the
printed WNIDs once before trusting the runs. (Nette may already exist at
`/data/guhyun9454/datasets/imagenette2-320` symlinked as `store/dataset/imagenette`.)

## 2. Init (6 jobs — one per subset, shared by DC & DM)

Same Gaussian init (gpc=200 → num_points=54, res=128) serves both objectives.

```bash
for sub in imagenette imagewoof imagefruit imageyellow imagemeow imagesquawk; do
  sbatch --export=ALL,SUBSET=$sub scripts/seraph/init.sbatch
done
```

## 3. Distill (12 jobs = 6 subsets × {DC, DM})

Chain each distill after its subset's init with `--dependency=afterok:<initjob>`,
or just submit after all inits report COMPLETED.

```bash
for sub in imagenette imagewoof imagefruit imageyellow imagemeow imagesquawk; do
  sbatch --export=ALL,OBJ=DM,SUBSET=$sub scripts/seraph/distill.sbatch   # DM: multi-GPU ready
  sbatch --export=ALL,OBJ=DC,SUBSET=$sub scripts/seraph/distill.sbatch   # DC: needs Phase 2 (see caveats)
done
```

The 12 logical scripts are the parameterized `distill.sbatch` invoked with the 12
`(OBJ, SUBSET)` combinations above — kept as one template instead of 12 copies to
avoid drift; the paper Table 12 (DC) / Table 10 (DM) settings are baked into it.

## 4. Resource verification (do at submit time)

The migration renamed nodes; re-verify before submitting:

```bash
sinfo -o '%P %l %D %G'                                  # partitions/time/gres
sinfo -N -o '%N %G %m %t' | grep ariel-g                # 24GB A5000 nodes g1..g5
sacctmgr -n show qos qos_guhyun9454_2026 \
   format=Name,Flags,MaxTRESPU,GrpTRES                  # GPU cap (was 4 concurrent)
squeue -u $USER -o '%.10i %.12j %.8T %.10M %R'          # current queue
```

- Account `ugrad` → partition `batch_ugrad` (TIMELIMIT infinite). `debug_ugrad`
  (4h) for quick smoke tests.
- A5000 24GB = `ariel-g1..g5` (gpu:8 each, so a 4-GPU job fits on one node). Do
  **not** target `ariel-k*` (high_perf, QOS-blocked). Update `--nodelist` to a free node.
- If the QOS GPU cap is still 4, the 12 distill jobs drain a few at a time.

## 5. Caveats (carry into `docs/gsdd/` + the eventual results)

- **DC multi-GPU is NOT done.** `main_DC.py` matching is second-order; at paper
  batch (real=720, gpc=200) it OOMs on one 24GB GPU and does not yet use >1 GPU.
  DC runs are blocked on Phase 2 (`docs/gsdd/phase2_multigpu_design.md`). **DM is
  ready** (it already wraps the embed net in `nn.DataParallel`).
- **Init memory at gpc=200.** init batch_size = gpc×10 = 2000 images; the prior
  gpc=64 init used 640. If init OOMs on 24GB, lower `batch_size` (it only affects
  init fitting speed, not the distill budget) — note the change.
- **batch_syn=2000 (paper) == batch_syn=0 (full)** here, since gpc=200 caps the
  per-class sample at 200. We pass 0 to use the fork's exact per-class grad-accum.
- **Residual batch-vs-paper gap:** record the actual `batch_real` used vs paper
  (720 DC / 1024 DM) if multi-GPU forces a reduction; document as a caveat.
- **Validity gate:** reproduction is paper-faithful iff the 6-subset average top-1
  is within **±2pp** of the `paper_numbers.md` average, per objective. Log results
  to wandb `TGwithIU` + `docs/results.md`; regenerate `degradation_nette_dc.png`.
