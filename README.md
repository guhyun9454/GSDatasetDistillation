# Beyond Pixels: Efficient Dataset Distillation via Sparse Gaussian Representation

This repository provides the official implementation of **Gaussian Splatting Dataset Distillation (GSDD)**,  
a novel sparse parameterization framework for efficient dataset distillation.

Unlike conventional pixel-level dense representations, GSDD encodes each distilled image as a compact set of 2D Gaussian primitives. This sparse representation significantly reduces storage redundancy, improves dataset diversity under the same budget, and enhances generalization to challenging samples.  

To ensure both efficiency and scalability, GSDD introduces a CUDA-based differentiable rasterizer that enables batch-parallel rendering, together with analytic pre-filtering and supersampling anti-aliasing strategies for artifact-free synthesis.  

Extensive experiments demonstrate that GSDD achieves state-of-the-art or highly competitive results on CIFAR-10, CIFAR-100, and multiple ImageNet subsets, while offering substantial improvements in inference/training speed and memory efficiency compared to prior approaches such as DDiF.


## Highlights

- Batch-parallel Gaussian rasterization with differentiable forward/backward passes.
- Anti-aliasing via analytic filters and SSAA for artifact-free synthesis.
- Mixed-precision execution (including bf16) and optional quantization to reduce memory and storage.
- Strategy framework for pruning Gaussians by opacity, spatial footprint, or gradient statistics.
- Turnkey initialization and training pipelines for TM, DC, and DM distillation variants.

## Repository Layout

- `configs/` — configuration files for initialization and distillation runs.
- `DC/`, `DM/` — dataset condensation and distribution matching baselines integrated with GSDD.
- `GaussianImage/gsplat/` — CUDA rasterizer adapted for batch-parallel Gaussian splatting.
- `lib/`
  - `gaussian/` — `GaussianImage_Cholesky_Batch` and supporting utilities for rendering distilled images.
  - other modules — auxiliary Gaussian processing utilities.
- `lib_ddif/` — components reused from DDiF for trajectory matching.
- `scripts/` — shell scripts for initialization (`scripts/init/`) and distillation (`scripts/distill/`).
- `utils/` — helper tools such as `python utils/generate_subset.py` for constructing ImageNet subsets.
- `buffer.py` — buffer creation helpers for trajectory matching.
- `distill_gs_dp.py` — entry point for TM-based Gaussian distillation.
- `init_gs.py` — distilled image initialization routine.

## Status & TODO

- [x] Release core GSDD codebase.
- [ ] Provide code for video dataset distillation.

## Installation

1. Create and activate a Python environment of your choice.
2. Install Python dependencies:

   ```bash
   # you should install torch first
   pip install -r requirements.txt
   ```

3. Build the optimized rasterization backend:

   ```bash
   cd GaussianImage/gsplat
   bash build_fast.sh
   ```

## Usage

All files are stored in the `./store` directory.  
You can either create the directory directly or set up a symbolic link:

```bash
mkdir -p ./store
# ln -s ${your_path} ./store
```


### 1. Prepare buffers

If you follow the trajectory matching pipeline, first create buffers as described in the
[MTT repository](https://github.com/georgecazenavette/mtt-distillation).

### 2. Initialize distilled Gaussians

Edit the scripts to specify dataset-specific parameters (`ipc`, `gpc`, `subset`, etc.), then run, for example:

```bash
bash scripts/init/cifar10.sh ${GPU_ID}
bash scripts/init/subset.sh ${GPU_ID}
```

### 3. Run distillation

Choose the configuration and available GPUs, then launch distillation:

```bash
# GPU_IDS: comma-separated list of devices, e.g., 0 or 0,1,2,3
# CFG: dataset config such as cifar10, cifar100, or imagenette_dp
bash scripts/distill/distill_DP.sh ${GPU_IDS} ${CFG}

# Override hyperparameters after ${CFG}, e.g.:
bash scripts/distill/distill_DP.sh "0,1,2,3" imagenette_dp subset="imageyellow" ipc=10
```

> The provided script contains helper functions (e.g., `lock_gpu`) that may require elevated
> privileges on shared machines. Remove or adapt them if you do not need GPU locking.

## Acknowledgements

This codebase builds upon and extends prior work. We gratefully acknowledge:

- **Rasterization operator**
  - [GaussianImage](https://github.com/Xinjie-Q/GaussianImage)
  - [gsplat](https://github.com/nerfstudio-project/gsplat)

- **Core implementations and frameworks**
  - [GLAD](https://github.com/GeorgeCazenavette/glad)
  - [DDiF](https://github.com/aailab-kaist/DDiF)

- **Dataset distillation methods**
  - *Dataset Condensation with Gradient Matching* — [paper](https://arxiv.org/abs/2006.05929), [code](https://github.com/VICO-UoE/DatasetCondensation)
  - *Dataset Condensation with Distribution Matching* — [paper](https://arxiv.org/abs/2110.04181), [code](https://github.com/VICO-UoE/DatasetCondensation)
  - *Dataset Distillation by Matching Training Trajectories* — [paper](https://arxiv.org/abs/2203.11932), [code](https://github.com/georgecazenavette/mtt-distillation)

## License

- Original code in this repository (outside directories that include their own license files) is released under the MIT License. See the root `LICENSE` file for full text and scoped details.
- Code in `GaussianImage/` remains covered by the upstream MIT license from the GaussianImage authors (`GaussianImage/LICENSE`).
- The `GaussianImage/gsplat/` module is distributed under the Apache License 2.0 as provided by the gsplat project (`GaussianImage/gsplat/LICENSE`).

Please keep all referenced license files when redistributing this project.
