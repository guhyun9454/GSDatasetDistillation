"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians_2d(
    means2d: Float[Tensor, "*batch 2"],
    L_elements: Float[Tensor, "*batch 3"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _ProjectGaussians2d.apply(
        means2d.contiguous(),
        L_elements.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )

class _ProjectGaussians2d(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means2d: Float[Tensor, "*batch 2"],
        L_elements: Float[Tensor, "*batch 3"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means2d.shape[-2]

        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_forward(
            num_points,
            means2d,
            L_elements,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            means2d,
            L_elements,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        (
            means2d,
            L_elements,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov2d, v_mean2d, v_L_elements) = _C.project_gaussians_2d_backward(
            ctx.num_points,
            means2d,
            L_elements,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean2d,
            # scales: Float[Tensor, "*batch 3"],
            v_L_elements,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
            # clip_thresh,
            None,
        )


def project_gaussians_2d_batch(
    batch_size: int,
    num_points_per_image: int,
    means2d: Float[Tensor, "batch_times_n 2"],
    L_elements: Float[Tensor, "batch_times_n 3"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(BATCH VERSION) This function projects 3D gaussians to 2D for a batch of images.

    Note:
        This function is differentiable w.r.t the means2d and L_elements inputs.

    Args:
        batch_size (int): The number of images in the batch.
        num_points_per_image (int): The number of gaussians per image.
        means2d (Tensor): xy locations of 2D gaussian projections for the entire batch.
                          Shape: (batch_size * num_points_per_image, 2)
        L_elements (Tensor): Cholesky decomposition elements for the 2D covariance matrix.
                             Shape: (batch_size * num_points_per_image, 3)
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        tile_bounds (Tuple): tile dimensions for a single image, as a len 3 tuple (tiles.x, tiles.y, 1).
        clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _ProjectGaussians2dBatch.apply(
        batch_size,
        num_points_per_image,
        means2d.contiguous(),
        L_elements.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )


class _ProjectGaussians2dBatch(Function):
    """(BATCH VERSION) PyTorch Function for projecting 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        batch_size: int,
        num_points_per_image: int,
        means2d: Float[Tensor, "batch_times_n 2"],
        L_elements: Float[Tensor, "batch_times_n 3"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        """The forward pass of the projection."""
        
        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_batch_forward(
            batch_size,
            num_points_per_image,
            means2d,
            L_elements,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.batch_size = batch_size
        ctx.num_points_per_image = num_points_per_image

        ctx.save_for_backward(
            means2d,
            L_elements,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        """The backward pass of the projection."""
        
        (
            means2d,
            L_elements,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_mean2d, v_L_elements) = _C.project_gaussians_2d_batch_backward(
            ctx.batch_size,
            ctx.num_points_per_image,
            means2d,
            L_elements,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        return (
            # batch_size: int
            None,
            # num_points_per_image: int
            None,
            # means2d: Float[Tensor, "batch_times_n 2"]
            v_mean2d,
            # L_elements: Float[Tensor, "batch_times_n 3"]
            v_L_elements,
            # img_height: int
            None,
            # img_width: int
            None,
            # tile_bounds: Tuple[int, int, int]
            None,
            # clip_thresh: float
            None,
        )