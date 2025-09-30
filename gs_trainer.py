import os
import sys
import logging
import time
import math
import json
from typing import List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from omegaconf import DictConfig, OmegaConf

from lib.gaussian.gs_utils import *
from lib.gaussian.gaussianimage_cholesky_batch import GaussianImage_Cholesky_Batch
from lib.gaussian.strategy_batch import DefaultStrategy

logger = logging.getLogger(__name__)

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


class GaussianTrainer:
    def __init__(self, cfg: DictConfig, img_list: List=None, output_dir=None, verbose=True, debug=False):
        self.working_dir = os.getcwd()
        if output_dir is None:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.output_dir = output_dir
        
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        err_path = os.path.join(self.output_dir, f"{file_name}.log")
        err_file = open(err_path, "a", buffering=1)
        sys.stderr = Tee(sys.__stderr__, err_file)
            
        self.verbose = verbose
        if not self.verbose:
            logger.setLevel(logging.WARNING)
        self.save_image_dir = f"{self.output_dir}/images"
        os.makedirs(self.save_image_dir, exist_ok=True)
        
        device = cfg.train.device
        self.device = device
        torch.cuda.set_device(self.device)

        assert (cfg.gaussian.H == cfg.gaussian.W) and (cfg.gaussian.H == cfg.dataset.resolution), \
            "Only support square images, please set H=W=resolution in the config"

        self.cfg = cfg
        self.train_cfg = cfg.train

        if img_list is None or len(img_list) == 0:
            img_list = ["store/test_images/astronaut.png", "store/test_images/Lenna_(test_image).png"]
        
        self.gt_img_list = [load_image_as_tensor(img, (cfg.gaussian.H, cfg.gaussian.W), device=self.device) if isinstance(img, str) else img.to(device) for img in img_list]
        self.gt_img = torch.concat(self.gt_img_list, dim=0)
        self.batch_size = self.gt_img.shape[0]

        logger.info(f"Initializing model for a batch of {self.batch_size} images.")
        self.cfg.gaussian.batch_size = self.batch_size
        
        self.model = GaussianImage_Cholesky_Batch(cfg.gaussian, device=device).to(device)
        
        opt_type = self.train_cfg.opt_type.lower()
        assert opt_type in ["adam", "adan"], "opt_type must be 'adam' or 'adan'"
        if opt_type == "adan":
            from lib.gaussian.optimizer import Adan
            logger.info("Using Adan optimizer.")
            opt = Adan
        else:
            logger.info("Using Adam optimizer.")
            opt = torch.optim.Adam
        self.optimizers = {
            name: (opt)(
                [{"params": self.model.params[name], "lr": self.train_cfg.lr.get(name)}],
            )
            for name in self.model.params
        }

        self.strategy = DefaultStrategy(cfg.strategy, self.model, self.optimizers)
        
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers["xy"], step_size=20000, gamma=0.5),
        ]

        self.scaler = torch.amp.GradScaler(enabled=(self.model.precision == "fp16"))
        logger.info(f"GradScaler enabled: {self.scaler.is_enabled()}")

    def train_iter(self, gt_image, step):
        
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        render_pkg = self.model()
        out_img = render_pkg["render"]
        loss = F.mse_loss(out_img, gt_image)

        self.scaler.scale(loss).backward()

        for optimizer in self.optimizers.values():
            self.scaler.unscale_(optimizer)
        self.strategy.step_pre_backward(step)
        
        self.strategy.step_post_backward(step)

        for optimizer in self.optimizers.values():
            self.scaler.step(optimizer)

        self.scaler.update()

        with torch.no_grad():
            self.model.clamp()
        
        for scheduler in self.schedulers:
            scheduler.step()

        with torch.no_grad():
            psnr = 10 * math.log10(1.0 / loss.item())
        
        debug_info = {
            "loss": loss.item(),
            "psnr": psnr,
        }
        
        return render_pkg, debug_info

    def train(self):
        for i, img in enumerate(self.gt_img_list):
            T.ToPILImage()(img.squeeze(0).cpu()).save(f"{self.save_image_dir}/gt_image_{i:02d}.png")
            
        max_steps = self.cfg.train.max_steps
        self.model.train()
        
        start = time.time()
        for step in range(max_steps):
            render_pkg, debug_info = self.train_iter(self.gt_img, step)
            loss, psnr = debug_info["loss"], debug_info["psnr"]

            if step % 1000 == 0:
                points_per_img = np.mean(self.model.num_points_per_image.tolist())
                logger.info(f"[{step:05d}/{max_steps}] loss={loss:.6f} | psnr={psnr:.3f} | points/img=[{points_per_img:.2f}]")

            with torch.no_grad():
                if step % self.train_cfg.save_img_steps == 0:
                    out_img_batch = render_pkg["render"]
                    for i in range(self.batch_size):
                        img_i = out_img_batch[i].cpu().clamp(0, 1)
                        T.ToPILImage()(img_i).save(f"{self.save_image_dir}/train_render_img{i}_{step:05d}.png")
                    
                    torch.save(self.model.state_dict(), f"{self.output_dir}/model_{step:05d}.pth")

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed/60:.2f} minutes")
    
    @torch.no_grad()    
    def eval(self):
        self.model.eval()
        render_pkg = self.model()
        image_batch = render_pkg["render"]
        
        psnr_list = []
        num_points_list = self.model.num_points_per_image.tolist()

        for i in range(self.batch_size):
            image_i = image_batch[i:i+1]
            gt_image_i = self.gt_img[i:i+1]
            
            mse_loss_i = F.mse_loss(image_i.float(), gt_image_i.float())
            psnr_i = 10 * math.log10(1.0 / mse_loss_i.item())
            psnr_list.append(psnr_i)
            logger.info(f"[Eval Img {i}] PSNR: {psnr_i:.3f} dB, Points: {num_points_list[i]}")
            
            img_to_save = image_i.squeeze(0).clamp(0, 1).cpu()
            T.ToPILImage()(img_to_save).save(f"{self.save_image_dir}/eval_render_img{i}.png")

        torch.save(self.model.state_dict(), f"{self.output_dir}/model_final.pth")

        return psnr_list, num_points_list

def apply_step_scaling(cfg: DictConfig):
    factor = cfg.train.steps_scaler
    if factor == 1.0:
        return
    cfg.train.eval_steps = [int(i * factor) for i in cfg.train.eval_steps]
    cfg.train.save_steps = [int(i * factor) for i in cfg.train.save_steps]
    cfg.train.max_steps = int(cfg.train.max_steps * factor)
    cfg.strategy.refine_start_iter = int(cfg.strategy.refine_start_iter * factor)
    cfg.strategy.refine_stop_iter = int(cfg.strategy.refine_stop_iter * factor)
    cfg.strategy.reset_every = int(cfg.strategy.reset_every * factor)
    cfg.strategy.refine_every = int(cfg.strategy.refine_every * factor)


def validate_cfg(cfg: DictConfig) -> bool:
    """Return True if hyperparameters satisfy sweep constraints."""
    split_rel = cfg.strategy.split_distance_rel
    del_rel = cfg.strategy.del_distance_rel

    if split_rel < 2 * del_rel:
        logger.info(
            "Invalid params: split_distance_rel %.4f < 2 * del_distance_rel %.4f",
            split_rel,
            del_rel,
        )
        return False

    if cfg.strategy.refine_every <= 0:
        return True

    n = max(
        (cfg.strategy.refine_stop_iter - cfg.strategy.refine_start_iter)
        // cfg.strategy.refine_every,
        0,
    )
    growth = (1 + split_rel - 2 * del_rel) ** n
    expected = int(cfg.gaussian.num_points * growth)

    if expected < cfg.strategy.max_num_points:
        logger.info(
            "Invalid params: expected %d points < max_num_points %d",
            expected,
            cfg.strategy.max_num_points,
        )
        return False

    return True


    
@hydra.main(config_path="configs/batch_gaussian_trainer", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    # Print final resolved config for debugging
    logger.info(OmegaConf.to_yaml(cfg))

    # Apply step scaling to training and strategy parameters
    apply_step_scaling(cfg)

    # if not validate_cfg(cfg):
    #     return

    # Start training
    runner = GaussianTrainer(cfg, debug=False)
    runner.train()
    psnr, num_points = runner.eval()
    logger.info(f"Final PSNR: {psnr}, Points per image: {num_points}")

if __name__ == "__main__":
    main()
