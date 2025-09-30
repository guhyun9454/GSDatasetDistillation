import logging
import math
import torch

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union
from omegaconf import DictConfig

from lib.gaussian.ops import del_batch, split_batch
from lib.gaussian.filters import OpacityFilter, ShapeFilter, SizeFilter, GradientFilter
from omegaconf import OmegaConf

logger = logging.getLogger("lib.gaussian.strategy")

class Operation:
    def __init__(self, cfg):
        if cfg is None:
            self.every_iter = -1
            self.start_iter = -1
            self.stop_iter = -1
            return
        
        self.filter = cfg.filter_name
        self.ops = cfg.ops
        self.start_iter = cfg.start_iter
        self.stop_iter = cfg.stop_iter
        self.every_iter = cfg.every_iter
        
        if self.ops == "split":
            self.split_method = getattr(cfg, "split_method", "simple_duplicate")
        
        if self.ops == "del":
            self.ops_func = del_batch
        elif self.ops == "split":
            self.ops_func = split_batch
        else:
            raise ValueError(f"Unsupported operation: {self.ops}. Supported operations are 'del', 'split', 'dup'.")
        
        params = OmegaConf.to_container(cfg.filter_params, resolve=True) if cfg.filter_params is not None else {}
        if self.filter == "opacity":
            self.filter_instance = OpacityFilter(**params)
        elif self.filter == "shape":
            self.filter_instance = ShapeFilter(**params)
        elif self.filter == "size":
            self.filter_instance = SizeFilter(**params)
        elif self.filter == "gradient":
            self.filter_instance = GradientFilter(**params)
        else:
            raise ValueError(f"Unsupported filter: {self.filter}. Supported filters are 'opacity', 'shape', 'size', 'gradient'.")
        
    
    def is_active(self, step: int) -> bool:
        if self.start_iter < 0 or self.stop_iter < 0 or self.every_iter <= 0:
            return False
        return self.start_iter <= step <= self.stop_iter and (step - self.start_iter) % self.every_iter == 0
    
    def step(
        self,
        gs_model,
        optimizers: Dict[str, torch.optim.Optimizer],
        step: int,
        info=None,
    ) -> int:
        if not self.is_active(step):
            return None
        
        if info is None:
            info = {}
        
        num_per_image_before = gs_model.num_points_per_image[0]
        num_before = gs_model.get_total_points
        if self.filter == "gradient":
            mask = self.filter_instance(gs_model, info)
        else:
            mask = self.filter_instance(gs_model)
        if self.ops == "split":
            n_ops = self.ops_func(gs_model, optimizers, mask, self.split_method, info)
        elif self.ops == "del":
            n_ops = self.ops_func(gs_model, optimizers, mask)
        else:
            raise ValueError(f"Unsupported operation: {self.ops}. Supported operations are 'del', 'split'.")
        num_after = gs_model.get_total_points
        num_per_image_after = gs_model.num_points_per_image[0]
        
        logger.info(f"Step {step}: [{self.filter.upper()}] Performed {self.ops} operation on {n_ops} GSs across batch. Per image points changed from {num_per_image_before} to {num_per_image_after}. Total points changed from {num_before} to {num_after}.")
        
        return mask


@dataclass
class DefaultStrategy:
    def __init__(self, cfg: DictConfig, gs_model, optimizers):
        self.gs_model = gs_model
        self.optimizers = optimizers
        self.max_num_points = math.inf
        self.eps = 1e-5
        self.del_ops = None
        self.grow_ops = None

        self.reset_state_every = -1
        self.use_abs = False
        self.accumulate_method = "mean"

        self.ema_beta = 0.9
        self.ema_bias_correction = True

        self.state = {}

        if cfg is not None:
            for k, v in cfg.items():
                assert k in self.__dict__, (
                    f"Unknown configuration key: {k}. "
                    "Please check the DefaultStrategy documentation for valid keys."
                )
                setattr(self, k, v)

        self.del_ops = Operation(self.del_ops)
        self.grow_ops = Operation(self.grow_ops)

        self._init_state_for_method()

    def _is_mean(self) -> bool:
        return self.accumulate_method.lower() == "mean"

    def _is_rms(self) -> bool:
        return self.accumulate_method.lower() == "rms"

    def _is_ema(self) -> bool:
        return self.accumulate_method.lower() == "ema"

    def _param_len_shape(self, p: torch.nn.Parameter):
        if p.data.ndim >= 2:
            return p.data.shape[:-1]
        else:
            return p.data.shape

    def _init_state_for_method(self):
        if self._is_ema():
            self.state["ema"] = {}
            self.state["ema_t"] = {}
            for name, p in self.gs_model.params.items():
                shp = self._param_len_shape(p)
                device = p.device
                self.state["ema"][name] = torch.zeros(shp, device=device)
                self.state["ema_t"][name] = torch.zeros(shp, device=device, dtype=torch.long)
        elif self._is_mean():
            self.state["sum_norm"] = {}
            self.state["count"] = {}
            for name, p in self.gs_model.params.items():
                shp = self._param_len_shape(p)
                device = p.device
                self.state["sum_norm"][name] = torch.zeros(shp, device=device)
                self.state["count"][name] = torch.zeros(shp, device=device)
        elif self._is_rms():
            self.state["sum_sq_norm"] = {}
            self.state["count"] = {}
            for name, p in self.gs_model.params.items():
                shp = self._param_len_shape(p)
                device = p.device
                self.state["sum_sq_norm"][name] = torch.zeros(shp, device=device)
                self.state["count"][name] = torch.zeros(shp, device=device)
        else:
            raise ValueError(f"Unknown accumulate_method: {self.accumulate_method}")

    def _xy_grad_to_per_pixel(self, g: torch.Tensor) -> torch.Tensor:
        if g.ndim >= 2 and g.shape[-1] == 2:
            scale = g.new_tensor([2.0 / float(self.gs_model.W),
                                2.0 / float(self.gs_model.H)])
            return g * scale
        return g

    def _vector_norm_per_point(self, g: torch.Tensor) -> torch.Tensor:
        if g.ndim >= 2:
            return torch.linalg.vector_norm(g, ord=2, dim=-1)
        else:
            return g.abs()

    def reset_state(self):
        if self._is_ema():
            for k in list(self.state["ema"].keys()):
                self.state["ema"][k].zero_()
                self.state["ema_t"][k].zero_()
        elif self._is_mean():
            for k in list(self.state["sum_norm"].keys()):
                self.state["sum_norm"][k].zero_()
                self.state["count"][k].zero_()
        elif self._is_rms():
            for k in list(self.state["sum_sq_norm"].keys()):
                self.state["sum_sq_norm"][k].zero_()
                self.state["count"][k].zero_()

    def get_avg_grads(self):
        out = {}
        if self._is_mean():
            for name, s in self.state["sum_norm"].items():
                c = self.state["count"][name].clamp_min(1.0)
                out[name] = s / c
        elif self._is_rms():
            for name, s2 in self.state["sum_sq_norm"].items():
                c = self.state["count"][name].clamp_min(1.0)
                out[name] = torch.sqrt(s2 / c)
        else:  # EMA
            beta = float(self.ema_beta)
            for name, ema in self.state["ema"].items():
                if self.ema_bias_correction:
                    t = self.state["ema_t"][name].float()
                    beta_t = torch.pow(torch.tensor(beta, device=ema.device), t)
                    bias = 1.0 - beta_t
                    out[name] = ema / bias.clamp_min(1e-8)
                else:
                    out[name] = ema
        return out

    def step_pre_backward(self, step: int):
        if self.reset_state_every > 0 and step % self.reset_state_every == 0:
            self.reset_state()

        with torch.no_grad():
            for name, p in self.gs_model.params.items():
                g = p.grad.detach()
                if self.use_abs:
                    g = g.abs()

                if name == "xy":
                    g = self._xy_grad_to_per_pixel(g)

                norm = self._vector_norm_per_point(g)

                if self._is_mean():
                    self.state["sum_norm"][name] += norm
                    self.state["count"][name] += 1.0
                elif self._is_rms():
                    self.state["sum_sq_norm"][name] += norm * norm
                    self.state["count"][name] += 1.0
                else:
                    beta = float(self.ema_beta)
                    self.state["ema"][name].mul_(beta).add_(norm, alpha=(1.0 - beta))
                    self.state["ema_t"][name] += 1
    
    def _apply_deletion_mask(self, del_mask: torch.Tensor):
        if del_mask is None:
            return
        if not torch.is_tensor(del_mask):
            del_mask = torch.as_tensor(del_mask, device=next(iter(self.gs_model.params.values())).device)
        keep = (~del_mask.bool())

        if self._is_mean():
            for name in list(self.state["sum_norm"].keys()):
                self.state["sum_norm"][name] = self.state["sum_norm"][name][keep]
                self.state["count"][name]    = self.state["count"][name][keep]
        elif self._is_rms():
            for name in list(self.state["sum_sq_norm"].keys()):
                self.state["sum_sq_norm"][name] = self.state["sum_sq_norm"][name][keep]
                self.state["count"][name]       = self.state["count"][name]
        else:
            for name in list(self.state["ema"].keys()):
                self.state["ema"][name]   = self.state["ema"][name][keep]
                self.state["ema_t"][name] = self.state["ema_t"][name][keep]
            

    def step_post_backward(self, step: int):
        info = {}
        mask = self.del_ops.step(self.gs_model, self.optimizers, step, info)
        if mask is not None:
            self._apply_deletion_mask(mask)
        info.update(self.get_avg_grads())
        mask = self.grow_ops.step(self.gs_model, self.optimizers, step, info)
        if mask is not None:
            self._init_state_for_method()