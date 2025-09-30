import os
import sys
sys.path.append('./lib')
import random
import logging
from multiprocessing import Pool
from typing import List, OrderedDict, Tuple, Set

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm

from lib.utils import get_dataset_for_init
from lib.gaussian.gs_utils import load_image_as_tensor
from lib.init_strategy import ClusteringStrategy
from gs_trainer import GaussianTrainer
from lib_ddif.utils import get_network

logger = logging.getLogger(__name__)

def worker_init_with_args(cfg: DictConfig, gpu: int, save_dir: str):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)

    global GLOBAL_CFG, OUTPUT_DIR
    GLOBAL_CFG = cfg
    OUTPUT_DIR = save_dir


def train_gs_task(task: Tuple[int, List[Tuple[int, int, object]], int, str]):
    """Run Gaussian training on *multiple* images, from one or more classes."""
    task_id, sample_info_list, gpu, range_str = task
    cfg = GLOBAL_CFG
    resolution = cfg.dataset.resolution

    # Determine the class(es) in this batch
    class_ids: Set[int] = {info[0] for info in sample_info_list}
    
    batch_imgs: List[torch.Tensor] = []
    for _, _, img in sample_info_list:
        if cfg.main.lazy:
            img = load_image_as_tensor(img, resolution)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        batch_imgs.append(img)

    img_batch = torch.cat(batch_imgs, dim=0)  # (B,3,H,W)

    if len(class_ids) == 1:
        # Single class in batch
        class_id = list(class_ids)[0]
        task_out_dir = os.path.join(OUTPUT_DIR, f"class_{class_id}", f"batch_{range_str}")
        class_log_str = f"class {class_id}"
    else:
        # Mixed classes in batch
        task_out_dir = os.path.join(OUTPUT_DIR, "mixed", f"batch_{range_str}")
        class_log_str = f"classes {sorted(list(class_ids))}"

    os.makedirs(task_out_dir, exist_ok=True)

    gs_trainer = GaussianTrainer(
        cfg,
        img_list=batch_imgs,
        output_dir=task_out_dir,
        verbose=not cfg.main.silent,
        debug=False,
    )

    gs_trainer.train()
    psnr_list, num_points_list = gs_trainer.eval()

    mean_psnr = float(np.mean(psnr_list))
    total_points = int(np.sum(num_points_list))
    return (
        f"Task {task_id:04d} | GPU {gpu} | {class_log_str} | {range_str} | "
        f"B={len(batch_imgs)} | mean PSNR={mean_psnr:.2f} dB | points={total_points}"
    )



class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


@hydra.main(config_path="./configs/init/", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)

    train_cfg = cfg.main
    data_cfg = cfg.dataset

    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(save_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(__file__))[0]
    err_path = os.path.join(save_dir, f"{file_name}.log")
    err_file = open(err_path, "a", buffering=1)
    sys.stderr = Tee(sys.__stderr__, err_file)


    # Reproducibility
    seed = train_cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset
    lazy = train_cfg.lazy
    transform = transforms.ToTensor() if lazy else transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((data_cfg.resolution, data_cfg.resolution)),
    ])

    train_dataset, _, num_classes, _ = get_dataset_for_init(
        data_cfg.name,
        data_cfg.dataset_path,
        data_cfg.subset,
        data_cfg.resolution,
        transform=transform,
        lazy=lazy,
    )

    gpc = data_cfg.gpc
    batch_size = int(train_cfg.batch_size)
    gpu_list = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
    num_gpus = len(gpu_list)
    logger.info(f"Detected {num_gpus} GPU(s): {gpu_list}")

    class_begin = train_cfg.class_idx_begin
    class_end = train_cfg.class_idx_end if train_cfg.class_idx_end >= 0 else num_classes
    
    tasks: List[Tuple[int, List[Tuple[int, int, object]], int, str]] = []
    gpc_samples_map = {} # For reproducibility log
    task_id = 0

    if train_cfg.get('allow_mixed_classes', False):
        logger.info("Mode: allow_mixed_classes=True. All images will be sorted globally and batched.")
        
        targets = np.array(train_dataset.targets)
        
        # 1. Gather all required samples, sorted by class, then by sample index
        all_samples = collect_selected_samples(train_dataset, gpc, class_begin, class_end, gpc_samples_map, targets, data_cfg.resolution, num_classes, cfg.init_strategy)
        
        # 2. Slice the globally sorted list into batches
        for start in range(0, len(all_samples), batch_size):
            end = min(start + batch_size, len(all_samples))
            batch_slice = all_samples[start:end]
            
            # Reformat sample info for the task to be (class_id, in-batch_sample_idx, img)
            batch_sample_info = [(s[0], i, s[2]) for i, s in enumerate(batch_slice)]

            range_str = f"{start}-{end - 1}"
            gpu_id = gpu_list[task_id % num_gpus]
            tasks.append((task_id, batch_sample_info, gpu_id, range_str))
            task_id += 1

    else:
        logger.info("Mode: allow_mixed_classes=False. Images will be batched per class.")
        
        targets = np.array(train_dataset.targets)
        # Original logic: iterate through each class and create batches within it
        for class_id in range(class_begin, class_end):
            indices = np.where(targets == class_id)[0]
            # The original code used random choice, we stick to that for this branch
            chosen = np.random.choice(indices, gpc, replace=False)
            gpc_samples_map[class_id] = chosen

            class_samples: List[Tuple[int, int, object]] = []
            for sample_idx, sample_id in enumerate(chosen):
                img, _ = train_dataset[sample_id]
                class_samples.append((class_id, sample_idx, img))

            for start in range(0, len(class_samples), batch_size):
                end = min(start + batch_size, len(class_samples))
                batch_slice = class_samples[start:end]
                range_str = f"{start}-{end - 1}"
                gpu_id = gpu_list[task_id % num_gpus]
                tasks.append((task_id, batch_slice, gpu_id, range_str))
                task_id += 1

    # Save indices for reproducibility
    np.save(os.path.join(save_dir, f"ipc_samples_{class_begin}_{class_end}.npy"), gpc_samples_map)

    logger.info(f"Prepared {len(tasks)} tasks (batch_size={batch_size}).")

    # Distribute tasks to GPU-specific pools
    max_tasks_per_gpu = train_cfg.max_tasks_per_gpu
    tasks_by_gpu = {gpu: [] for gpu in gpu_list}
    for t in tasks:
        tasks_by_gpu[t[2]].append(t)

    gpu_pools = {}
    results = []
    for gpu in gpu_list:
        if not tasks_by_gpu[gpu]:
            continue
        pool = Pool(
            processes=max_tasks_per_gpu,
            initializer=worker_init_with_args,
            initargs=(cfg, gpu, save_dir),
        )
        gpu_pools[gpu] = pool
        results.append((gpu, pool.imap_unordered(train_gs_task, tasks_by_gpu[gpu])))

    for gpu, res_iter in results:
        for r in tqdm(res_iter, total=len(tasks_by_gpu[gpu]), desc=f"GPU {gpu} tasks", disable=train_cfg.silent):
            if not train_cfg.silent:
                logger.info(r)

    for gpu, pool in gpu_pools.items():
        pool.close()
        pool.join()
        logger.info(f"GPU {gpu} pool finished all tasks.")


def get_model_for_init(cfg: DictConfig, res: int, num_classes: int):
    if cfg.model_type == "load":
        model = get_network(
            cfg.model_name,
            cfg.model_channel,
            num_classes=num_classes,
            im_size=res,
        )
        buffer_path = cfg.model_path
        buffer = torch.load(buffer_path, map_location='cpu')
        params = buffer[cfg.teacher_traj][cfg.model_epochs]
        state_dict = OrderedDict()
        param_keys = [k for k, _ in model.named_parameters()]
        for k, v in zip(param_keys, params):
            state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        model = model.features
    elif cfg.model_type == "random":
        model = get_network(
            cfg.model_name,
            cfg.model_channel,
            num_classes=num_classes,
            im_size=res,
        )
        model = model.features
    elif cfg.model_type == "pretrained":
        # model = torchvision.models.
        if cfg.model_name == "resnet18":
            model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif cfg.model_name == "resnet50":
            model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model_name for pretrained: {cfg.model_name}")
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def collect_selected_samples(train_dataset, gpc, class_begin, class_end, gpc_samples_map, targets, res, num_classes, init_strategy_cfg):
    all_samples: List[Tuple[int, int, object]] = []

    for class_id in range(class_begin, class_end):
        
        if init_strategy_cfg.sample_type == "load":
            cluster_order_dir = init_strategy_cfg.get("cluster_order_dir", None)
            if cluster_order_dir is None:
                raise ValueError("cluster_order_dir must be specified for 'load' sample_type.")
            order_file = os.path.join(cluster_order_dir, f"class_{class_id}_order.npy")
            if os.path.isfile(order_file):
                order_pairs = np.load(order_file)
                chosen_indices = order_pairs[:, 1]
            else:
                raise FileNotFoundError(f"Order file for class {class_id} not found: {order_file}")
        elif init_strategy_cfg.sample_type == "random":
                # Find all indices for the current class (already sorted)
            indices = np.where(targets == class_id)[0]
                # Select the first `gpc` samples
            chosen_indices = indices[:gpc]
        elif init_strategy_cfg.sample_type == "kmeans":
            indices = np.where(targets == class_id)[0]
            images = [train_dataset[i][0] for i in indices]
            images = torch.stack(images, dim=0)  # (N, 3, H, W)
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            sel_model = get_model_for_init(init_strategy_cfg, (res, res), num_classes)
            cluster_strategy = ClusteringStrategy(images=images, net=sel_model)
            sel_chosen_indices = cluster_strategy.query(n=gpc).detach().cpu().numpy()
            chosen_indices = indices[sel_chosen_indices]
        else:
            raise ValueError(f"Unknown sample_type: {init_strategy_cfg.sample_type}")
        gpc_samples_map[class_id] = chosen_indices

        for sample_index, sample_id in enumerate(chosen_indices):
            img, _ = train_dataset[sample_id]
                # Storing (class_id, original_sample_id, image_data)
            all_samples.append((class_id, int(sample_id), img))

    return all_samples


if __name__ == "__main__":
    main()