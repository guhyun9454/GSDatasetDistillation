# GSDD 

This repository provides the implementation of Gaussian-based dataset distillation.  

## TODO

- [ ] More detailed description.
- [ ] Complete scrpts and configs.

## Installation

1. Create and activate a Python environment.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the fast rasterization operator:

   ```bash
   cd GaussianImage/gsplat
   bash build_fast.sh
   ```


## Usage

### Step 1: Initialize distilled points

Run one of the initialization scripts under:

```bash
bash scripts/init/init_example.sh
```

### Step 2: Start distillation

Run one of the distillation scripts under:

```bash
bash scripts/distill/distill_example.sh
```

> Replace `init_example.sh` and `distill_example.sh` with your actual scripts.


## Acknowledgements

This codebase builds upon and extends prior work. In particular, we acknowledge the following contributions:

- **Rasterization Operator**  
  Adapted from:
  - [GaussianImage](https://github.com/Xinjie-Q/GaussianImage)
  - [gsplat](https://github.com/nerfstudio-project/gsplat)

- **Core Implementations and Frameworks**  
  Significant portions of this work are based on the following projects:
  - [GLAD](https://github.com/GeorgeCazenavette/glad)
  - [DDiF](https://github.com/aailab-kaist/DDiF)

- **Dataset Distillation Methods**  
  We also draw inspiration and code from the following papers and their official implementations:
  - *Dataset Condensation with Gradient Matching*  
    [Paper](https://arxiv.org/abs/2006.05929), [Code](https://github.com/VICO-UoE/DatasetCondensation)
  - *Dataset Condensation with Distribution Matching*  
    [Paper](https://arxiv.org/abs/2110.04181), [Code](https://github.com/VICO-UoE/DatasetCondensation)
  - *Dataset Distillation by Matching Training Trajectories*  
    [Paper](https://arxiv.org/abs/2203.11932), [Code](https://github.com/georgecazenavette/mtt-distillation)





## License
