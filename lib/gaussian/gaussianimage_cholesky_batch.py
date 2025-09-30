import logging

import torch
import torch.nn as nn

from lib.gaussian.gs_utils import *

from gsplat.project_gaussians_2d import project_gaussians_2d_batch, project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum_batch, rasterize_gaussians_sum
from omegaconf import DictConfig

logger = logging.getLogger("lib.gaussian.gaussianimage_cholesky_batch")

class GaussianImage_Cholesky_Batch(nn.Module):
    def __init__(self, cfg: DictConfig, device):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.init_num_points = self.cfg.num_points
        self.batch_size = cfg.get("batch_size", 1)
        self.use_opacity = cfg.get("use_opacity", False)
        if self.use_opacity:
            logger.info("GaussianImage_Cholesky: using opacity")
        logger.info(f"GaussianImage_Cholesky: batch size set to {self.batch_size}")
        logger.info(f"GaussianImage_Cholesky: init {self.init_num_points} points per image")

        self.H, self.W = self.cfg.H, self.cfg.W
        self.BLOCK_W, self.BLOCK_H = self.cfg.BLOCK_W, self.cfg.BLOCK_H
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        
        self.precision = str(cfg.get("precision", "fp32")).lower()  # 'fp32' | 'fp16' | 'bf16' | 'fp8'
        self.kernel_dtype_pref = str(cfg.get("kernel_dtype", "fp32")).lower()  # 'fp32' | 'native'

        logger.info(f"GaussianImage_Cholesky: using precision = {self.precision}, kernel_dtype = {self.kernel_dtype_pref}")

        def _resolve_param_dtype(p):
            if p == "fp32": return torch.float32
            if p == "fp16": return torch.float32
            if p == "bf16": return torch.float32
            if p == "fp8": return torch.float32
            raise ValueError(f"Unsupported precision: {p}")

        def _resolve_compute_dtype(p):
            if p == "fp32": return torch.float32
            if p == "fp16": return torch.float16
            if p == "bf16": return torch.bfloat16
            if p == "fp8":  return torch.float8_e4m3fn
            raise ValueError(f"Unsupported precision: {p}")

        self.param_dtype   = _resolve_param_dtype(self.precision)
        self.compute_dtype = _resolve_compute_dtype(self.precision)
        self._use_kernel_native_dtype = (self.kernel_dtype_pref == "native")

        base_eps = float(cfg.get("eps", 1e-6))
        if self.precision in ("fp16", "bf16", "fp8"):
            base_eps = max(base_eps, 1e-4)
        self._eps_for_clamp = base_eps

        total_points = self.init_num_points * self.batch_size
        self.params = nn.ParameterDict({
            "xy":       nn.Parameter(2 * torch.rand(total_points, 2, device=device, dtype=self.param_dtype) - 1),
            "cholesky": nn.Parameter(torch.rand(total_points, 3, device=device, dtype=self.param_dtype)),
            "features": nn.Parameter(torch.rand(total_points, 3, device=device, dtype=self.param_dtype)),
        })
        
        opacity_all = torch.ones((total_points, 1), device=device, dtype=self.param_dtype)

        if self.use_opacity:
            self.params["opacity"] = nn.Parameter(opacity_all)
            self._opacity = self.params["opacity"]
        else:
            self.register_buffer("_opacity", opacity_all)

        self.register_buffer('num_points_per_image',
                             torch.full((self.batch_size,), self.init_num_points,
                                        dtype=torch.long, device=device))
        background_single = torch.ones(3, device=device, dtype=self.param_dtype)
        self.register_buffer('background', background_single.unsqueeze(0).repeat(self.batch_size, 1))
        self.register_buffer('cholesky_bound',
                             torch.tensor([0.5, 0, 0.5], device=device,
                                          dtype=self.param_dtype).view(1, 3))
    
    def clamp(self):
        lo = -1 + self._eps_for_clamp * 10
        hi =  1 - self._eps_for_clamp * 10
        self.params['xy'].data.clamp_(lo, hi)
    
    def _to_kernel_dtype(self, *tensors):
        if self._use_kernel_native_dtype:
            return [t.contiguous() for t in tensors]
        else:
            return [t.to(torch.float32).contiguous() for t in tensors]
    
    def _autocast_context(self):
        use_amp = self.compute_dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        return torch.amp.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=use_amp)

    @property
    def get_total_points(self):
        return self.num_points_per_image.sum().item()

    @property
    def offsets(self):
        return torch.cumsum(torch.cat([torch.tensor([0], device=self.device), self.num_points_per_image[:-1]]), dim=0)

    @property
    def get_xyz(self):
        return torch.clamp(min=-1, max=1.0, input=self.params['xy'])
    
    @property
    def get_features(self):
        return self.params['features']
    
    @property
    def get_opacity(self):
        return self.params["opacity"] if self.use_opacity else self._opacity

    @property
    def get_cholesky_elements(self):
        return self.params['cholesky'] + self.cholesky_bound

    
    @property
    def get_total_storage(self):
        total_params = sum(p.numel() for p in self.params.values())
        total_bytes = total_params * torch.tensor([], dtype=self.param_dtype).element_size()
        return total_bytes
    
    def quantize_self(self, dtype=None):
        if dtype is None:
            dtype = self.compute_dtype
        else:
            if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn):
                raise ValueError("Only fp32, fp16, bf16, fp8 are supported for quantization.")
        
        dtype_list_before = [param.dtype for param in self.params.values()]
        assert all(d == dtype_list_before[0] for d in dtype_list_before), "All parameters must have the same dtype before quantization."
        storage_before = self.get_total_storage
        logger.info(f"Before quantization: total params = {self.get_total_points}, param_dtype = {dtype_list_before[0]}, total storage = {storage_before} bytes.")
        for name, param in self.params.items():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)
                logger.info(f"Quantized parameter '{name}' to {dtype}.")
        
        self.param_dtype = dtype
        self.compute_dtype = dtype
        dtype_list_after = [param.dtype for param in self.params.values()]
        assert all(d == dtype for d in dtype_list_after), "All parameters must have the same dtype after quantization."
        storage_after = self.get_total_storage
        logger.info(f"After quantization: total params = {self.get_total_points}, param_dtype = {dtype}, total storage = {storage_after} bytes.")

    def forward(self):
        num_points_for_cuda = self.num_points_per_image[0].item()

        with self._autocast_context():
            xy      = self.params['xy']
            chol       = self.params['cholesky']
            feats   = self.get_features
            opac    = self.get_opacity
            bg      = self.background

            xy = xy.to(self.compute_dtype)
            chol  = chol .to(self.compute_dtype)
            feats = feats.to(self.compute_dtype)
            opac  = opac.to(self.compute_dtype)
            bg    = bg.to(self.compute_dtype)

            xy_k, chol_k, feats_k, opac_k, bg_k = self._to_kernel_dtype(xy, chol, feats, opac, bg)
            L_k = chol_k + self.cholesky_bound

            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_batch(
                self.batch_size, num_points_for_cuda, xy_k, L_k,
                self.H, self.W, self.tile_bounds
            )

            out_img = rasterize_gaussians_sum_batch(
                self.batch_size, num_points_for_cuda,
                xys, depths, radii, conics, num_tiles_hit,
                feats_k, opac_k,
                self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=bg_k
            )

        return {"render": out_img.permute(0, 3, 1, 2).contiguous()}
    