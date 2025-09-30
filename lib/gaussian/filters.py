import os
import sys
sys.path.append(".")
import logging

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List
import copy

from lib.gaussian.gaussianimage_cholesky_batch import GaussianImage_Cholesky_Batch
from lib.gaussian.gs_utils import compute_sizes_from_cholesky, compute_shape_ratios_from_cholesky


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseFilter(ABC):
    @abstractmethod
    def __call__(self, model: nn.Module) -> torch.Tensor:
        pass

class OpacityFilter(BaseFilter):
    def __init__(self,
                 remain_percentage: float,
                 threshold: float = None,
                 keep_smaller_value: bool = False,
                 per_image: bool = True,
                 use_percentage: bool = True):
        if not 0.0 <= remain_percentage <= 1.0:
            raise ValueError
        if not use_percentage and threshold is None:
            raise ValueError

        self.remain_percentage = remain_percentage
        self.threshold = threshold
        self.keep_smaller_value = keep_smaller_value
        self.per_image = per_image
        self.use_percentage = use_percentage

    def __call__(self, model) -> torch.Tensor:
        if self.use_percentage and self.remain_percentage == 1.0:
            return torch.ones(model.get_total_points, dtype=torch.bool, device=model.device)

        num_points_per_image = model.num_points_per_image[0].item()
        batch_size = model.batch_size

        opacities_flat = model.get_opacity.view(-1)

        if self.per_image:
            opacities = opacities_flat.view(batch_size, num_points_per_image)
            to_keep = int(num_points_per_image * self.remain_percentage)
        else:
            opacities = opacities_flat.unsqueeze(0)
            to_keep = int(opacities_flat.size(0) * self.remain_percentage)

        if self.use_percentage:
            if to_keep == 0:
                raise ValueError("no points")
            _, idx = torch.topk(
                opacities,
                k=to_keep,
                largest=not self.keep_smaller_value,
                dim=1
            )
            keep_mask = torch.zeros_like(opacities, dtype=torch.bool).scatter_(1, idx, True)
        else:
            th = torch.tensor(self.threshold, device=opacities.device, dtype=opacities.dtype)
            if self.keep_smaller_value:
                keep_mask = opacities <= th
            else:
                keep_mask = opacities >= th

        return keep_mask.view(-1)


class SizeFilter(BaseFilter):
    def __init__(self,
                 remain_percentage: float,
                 threshold: float = None,
                 keep_smaller_value: bool = False,
                 per_image: bool = True,
                 use_percentage: bool = True,
                 ):
        if not 0.0 <= remain_percentage <= 1.0:
            raise ValueError
        
        if not use_percentage and threshold is None:
            raise ValueError
            
        self.remain_percentage = remain_percentage
        self.keep_smaller_value = keep_smaller_value
        self.per_image = per_image
        self.threshold = threshold
        self.use_percentage = use_percentage

    def __call__(self, model) -> torch.Tensor:
        if self.use_percentage and self.remain_percentage == 1.0:
            return torch.ones(model.get_total_points, dtype=torch.bool, device=model.device)

        num_points_per_image = model.num_points_per_image[0].item()
        batch_size = model.batch_size
        
        cholesky_flat = model.get_cholesky_elements.view(-1, 3)
        
        sizes_flat = compute_sizes_from_cholesky(cholesky_flat, use_sqrt=False)
        
        if self.per_image:
            sizes = sizes_flat.view(batch_size, num_points_per_image)
            to_keep = int(num_points_per_image * self.remain_percentage)
        else:
            sizes = sizes_flat.unsqueeze(0)
            to_keep = int(sizes_flat.size(0) * self.remain_percentage)
        
        if to_keep == 0:
            raise ValueError("No points")
        
        if self.use_percentage:
            _, idx = torch.topk(sizes, k=to_keep, largest=not self.keep_smaller_value, dim=1)
            keep_mask = torch.zeros_like(sizes, dtype=torch.bool).scatter_(1, idx, True)
        else:
            threshold = torch.tensor(self.threshold, device=sizes.device, dtype=sizes.dtype)
            keep_mask = sizes <= threshold if self.keep_smaller_value else sizes >= threshold
        return keep_mask.view(-1)


class ShapeFilter(BaseFilter):
    def __init__(self,
                 remain_percentage: float,
                 threshold: float = None,
                 keep_smaller_value: bool = True,
                 per_image: bool=True,
                 use_percentage: bool=True):
        
        if not 0.0 <= remain_percentage <= 1.0:
            raise ValueError("remain_percentage must be between 0.0 and 1.0.")
        
        if not use_percentage and threshold is None:
            raise ValueError("threshold must be provided when use_percentage=False.")
        

        self.remain_percentage =  remain_percentage
        self.keep_smaller_value = keep_smaller_value
        self.per_image = per_image
        self.threshold = threshold
        self.use_percentage = use_percentage

    def __call__(self, model) -> torch.Tensor:
        if self.use_percentage and self.remain_percentage == 1.0:
            return torch.ones(model.get_total_points, dtype=torch.bool, device=model.device)

        num_points_per_image = model.num_points_per_image[0].item()
        batch_size = model.batch_size

        cholesky_flat = model.get_cholesky_elements.view(-1, 3)

        ratios_flat = compute_shape_ratios_from_cholesky(cholesky_flat)

        if self.per_image:
            ratios = ratios_flat.view(batch_size, num_points_per_image)
            to_keep = int(num_points_per_image * self.remain_percentage)
        else:
            ratios = ratios_flat.unsqueeze(0)
            to_keep = int(ratios_flat.size(0) * self.remain_percentage)
        
        if to_keep == 0:
            raise ValueError("no points")
        
        if self.use_percentage:
            _, idx = torch.topk(ratios, k=to_keep, largest=not self.keep_smaller_value, dim=1)
            keep_mask = torch.zeros_like(ratios, dtype=torch.bool).scatter_(1, idx, True)
        else:
            threshold = torch.tensor(self.threshold, device=ratios.device, dtype=ratios.dtype)
            keep_mask = ratios <= threshold if self.keep_smaller_value else ratios >= threshold
        return keep_mask.view(-1)


class UniverseFilter:
    def __init__(self,
                remain_percentage: float,
                threshold: float = None,
                keep_smaller_value: bool = True,
                per_image: bool=True,
                use_percentage: bool=True):

        if not 0.0 <= remain_percentage <= 1.0:
            raise ValueError

        if not use_percentage and threshold is None:
            raise ValueError

        self.remain_percentage =  remain_percentage
        self.keep_smaller_value = keep_smaller_value
        self.per_image = per_image
        self.threshold = threshold
        self.use_percentage = use_percentage

    def __call__(self, model, arr) -> torch.Tensor:
        if self.use_percentage and self.remain_percentage == 1.0:
            return torch.ones(model.get_total_points, dtype=torch.bool, device=model.device)

        num_points_per_image = model.num_points_per_image[0].item()
        batch_size = model.batch_size

        if self.per_image:
            new_arr = arr.view(batch_size, num_points_per_image)
            to_keep = int(num_points_per_image * self.remain_percentage)
        else:
            new_arr = arr.unsqueeze(0)
            to_keep = int(arr.size(0) * self.remain_percentage)

        if to_keep == 0:
            raise ValueError("no points")

        if self.use_percentage:
            _, idx = torch.topk(new_arr, k=to_keep, largest=not self.keep_smaller_value, dim=1)
            keep_mask = torch.zeros_like(new_arr, dtype=torch.bool).scatter_(1, idx, True)
        else:
            threshold = torch.tensor(self.threshold, device=new_arr.device, dtype=new_arr.dtype)
            keep_mask = new_arr <= threshold if self.keep_smaller_value else new_arr >= threshold
        return keep_mask.view(-1)


class GradientFilter:
    def __init__(
        self,
        remain_percentage: float,
        gradient_key: str,
        threshold: float = None,
        keep_smaller_value: bool = True,
        per_image: bool = True,
        use_percentage: bool = True,
    ):

        if not 0.0 <= remain_percentage <= 1.0:
            raise ValueError

        if not use_percentage and threshold is None:
            raise ValueError

        self.remain_percentage = remain_percentage
        self.keep_smaller_value = keep_smaller_value
        self.per_image = per_image
        self.threshold = threshold
        self.use_percentage = use_percentage
        
        self.gradient_key = gradient_key

    def __call__(self, model, info) -> torch.Tensor:
        if self.use_percentage and self.remain_percentage == 1.0:
            return torch.ones(
                model.get_total_points, dtype=torch.bool, device=model.device
            )

        num_points_per_image = model.num_points_per_image[0].item()
        batch_size = model.batch_size

        arr = info[self.gradient_key]

        if self.per_image:
            new_arr = arr.view(batch_size, num_points_per_image)
            to_keep = int(num_points_per_image * self.remain_percentage)
        else:
            new_arr = arr.unsqueeze(0)
            to_keep = int(arr.size(0) * self.remain_percentage)

        if to_keep == 0:
            raise ValueError("no points")

        if self.use_percentage:
            _, idx = torch.topk(
                new_arr, k=to_keep, largest=not self.keep_smaller_value, dim=1
            )
            keep_mask = torch.zeros_like(new_arr, dtype=torch.bool).scatter_(
                1, idx, True
            )
        else:
            threshold = torch.tensor(
                self.threshold, device=new_arr.device, dtype=new_arr.dtype
            )
            keep_mask = (
                new_arr <= threshold
                if self.keep_smaller_value
                else new_arr >= threshold
            )
        return keep_mask.view(-1)


def filter_and_repack(
    original_model,
    filter_instance: BaseFilter
):
    
    new_model = copy.deepcopy(original_model)

    if not filter_instance:
        logger.warning("NO FILTER")
        return new_model

    num_points_per_image_tensor = original_model.num_points_per_image
    if not torch.all(num_points_per_image_tensor == num_points_per_image_tensor[0]):
        raise ValueError("All images must have the same number of points before filtering.")
    
    current_num_points = num_points_per_image_tensor[0].item() # all images have the same number of points
    if current_num_points == 0:
        logger.warning("No points")
        return new_model

    logger.info(f"Applying filter: {filter_instance.__class__.__name__}")
    keep_mask = filter_instance(original_model)

    final_mask_float = keep_mask.unsqueeze(-1).float()

    new_model._opacity.data *= final_mask_float
    
    mask_batched = keep_mask.view(original_model.batch_size, current_num_points)
    num_kept_per_image = mask_batched.sum(dim=1)
    num_total_per_image = torch.tensor([current_num_points] * original_model.batch_size, device=original_model.device)[0]
    
    
    return new_model, num_kept_per_image, num_total_per_image
