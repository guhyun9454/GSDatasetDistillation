import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")
import logging
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib

from typing import Tuple
from pytorch_msssim import ms_ssim, ssim
from PIL import Image

matplotlib.use("Agg")

logger = logging.getLogger("lib.gaussian.gs_utils")

def create_optimizers_and_schedulers(gs_model, args):
    logger.info("Creating or resetting optimizers and schedulers...")
    
    optimizers = {
        name: torch.optim.Adam(
            [{"params": gs_model.params[name], "lr": args.lr_gs.get(name)}],
        )
        for name in gs_model.params
    }
    
    if args.scheduler_gs.type == "cosine":
        schedulers = {
            name: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[name], 
                T_max=args.scheduler_gs.T_max, 
                eta_min=args.scheduler_gs.eta_min
            )
            for name in optimizers
        }
    else:
        schedulers = {}

    logger.info("Optimizers and schedulers have been successfully created/reset.")
    return optimizers, schedulers


def cholesky_to_covariance(cholesky: torch.Tensor) -> torch.Tensor:
    L = torch.zeros(cholesky.size(0), 2, 2, device=cholesky.device)
    L[:, 0, 0] = cholesky[:, 0]
    L[:, 1, 0] = cholesky[:, 1]
    L[:, 1, 1] = cholesky[:, 2]
    return torch.bmm(L, L.transpose(1, 2))

def compute_shape_ratios_from_cholesky(cholesky: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    covariances = cholesky_to_covariance(cholesky)
    eigenvalues = torch.linalg.eigvalsh(covariances)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    large_eigen, small_eigen = torch.max(eigenvalues, dim=1).values, torch.min(eigenvalues, dim=1).values
    
    ratios = torch.sqrt(large_eigen / (small_eigen + epsilon))
    return ratios

def compute_sizes_from_cholesky(cholesky: torch.Tensor, use_sqrt: bool=True) -> torch.Tensor:
    covariances = cholesky_to_covariance(cholesky)
    determinants = torch.linalg.det(covariances)
    
    if use_sqrt:
        return torch.sqrt(torch.clamp(determinants, min=0))
    else:
        return determinants



def load_image_as_tensor(image_path: str, resolution: Tuple[int, int], device: str = "cuda") -> torch.Tensor:
    """Load and preprocess an image into a batched tensor on the specified device."""
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(resolution),     # Resize to (H, W)
        T.ToTensor(),             # Convert to [0, 1] tensor with shape [3, H, W]
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension => [1, 3, H, W]
    return img_tensor



def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss


