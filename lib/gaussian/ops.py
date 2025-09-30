from re import split
from typing import Dict

import torch

from torch import nn

import torch, logging


logger = logging.getLogger("lib.gaussian.ops")


@torch.no_grad()
def del_batch(gs_model, optimizers: Dict[str, torch.optim.Optimizer], mask: torch.Tensor):
    # mask: shape (total_points,) or (batch_size * ori_points_per_image,)
    device = mask.device
    sel = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(device)
    n_delete = sel.numel()
    if n_delete == 0:
        return 0

    # assume all images same points per image
    batch_size = gs_model.batch_size
    ori_points_per_image = gs_model.num_points_per_image[0].item()
    points_deleted_per_image = mask.reshape(batch_size, ori_points_per_image).sum(dim=1)
    assert torch.all(points_deleted_per_image == points_deleted_per_image[0]), \
        "The number of points to be deleted must be the same for each image in the batch."

    # update num_points_per_image
    gs_model.num_points_per_image = gs_model.num_points_per_image - points_deleted_per_image

    keep_mask = ~mask
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).to(device)

    param_names = ["xy", "cholesky", "features"]
    if gs_model.use_opacity:
        param_names.append("opacity")

    for p_name in param_names:
        old_p = gs_model.params[p_name]
        old_data = old_p.data
        # first dim is number of points
        # slice along first dim:
        new_data = old_data[keep_idx].clone()
        new_p = nn.Parameter(new_data, requires_grad=old_p.requires_grad)
        # replace in ParameterDict
        gs_model.params[p_name] = new_p

        # optimizer handling
        opt = optimizers.get(p_name)
        assert opt is not None, f"Optimizer for param {p_name} not found."
        for g in opt.param_groups:
            g["params"] = [(new_p if p is old_p else p) for p in g["params"]]

        # adapt optimizer state: for each tensor in state, reshape pad new zeros for split
        st_old = opt.state.pop(old_p, None)
        if st_old is None:
            # just set empty state for new param
            opt.state[new_p] = {}
            continue

        st_new = {}
        for k, v in st_old.items():
            if k == "step" or not isinstance(v, torch.Tensor):
                # keep scalars or non-tensor states as-is
                st_new[k] = v
                continue

            # v is a tensor: check whether its first dim matches old_p first dim
            if v.shape[0] == old_p.shape[0]:
                # per-point buffer, slice the first dimension
                st_new[k] = v[keep_idx].clone()
            else:
                # shape doesn't match first dim -> keep unchanged (could be EMA scalars, etc.)
                st_new[k] = v.clone()
        opt.state[new_p] = st_new

    # buffers: handle non-trainable opacity if present
    if (not gs_model.use_opacity) and hasattr(gs_model, "_opacity"):
        old_buf = getattr(gs_model, "_opacity")
        # assume shape (total_points, ...) or (total_points,1)
        new_buf = old_buf[keep_idx].clone()
        setattr(gs_model, "_opacity", new_buf)

    return n_delete


def _get_split_params(
    p_name: str,
    grouped_old_data: torch.Tensor,
    sel: torch.Tensor,
    split_method: str,
    split_info: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute and return the modified original parameters and the new parameters for split points according to the specified splitting strategy.

    Args:
        p_name (str): Name of the parameter being processed ("xy", "cholesky", "features", "opacity").
        grouped_old_data (torch.Tensor): Batched original parameter tensor (B, N, ...).
        sel (torch.Tensor): Flattened indices of Gaussians to split.
        split_method (str): Splitting strategy to use.
        split_info (dict): Dictionary containing extra information required by the strategy.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - grouped_old_data_modified (torch.Tensor): The original parameters, possibly modified.
            - new_data_split (torch.Tensor): Parameters created for the newly split Gaussians.
    """
    rest_shape = grouped_old_data.shape[2:]
    flat_old_data = grouped_old_data.reshape(-1, *rest_shape)
    data_to_split = flat_old_data[sel].clone()

    if split_method == "simple_duplicate":
        if p_name == "xy":
            new_data_split = data_to_split + 1e-6 * torch.randn_like(data_to_split)
        elif p_name == "opacity":
            flat_old_data[sel] *= 0.5
            grouped_old_data = flat_old_data.reshape(grouped_old_data.shape)
            new_data_split = data_to_split * 0.5
        else:
            new_data_split = data_to_split
        return grouped_old_data, new_data_split

    elif split_method == "principal_axis":
        scale_factor = split_info.get("scale_factor", 1.6)
        if p_name == "xy":
            displacement = split_info["principal_displacements"]
            flat_old_data[sel] = data_to_split - displacement
            grouped_old_data = flat_old_data.reshape(grouped_old_data.shape)
            new_data_split = data_to_split + displacement
        elif p_name == "cholesky":
            scaling = torch.sqrt(torch.tensor(scale_factor, device=data_to_split.device))
            flat_old_data[sel] /= scaling
            grouped_old_data = flat_old_data.reshape(grouped_old_data.shape)
            new_data_split = data_to_split / scaling
        else:
            new_data_split = data_to_split
        return grouped_old_data, new_data_split

    elif split_method == "stochastic_sample":
        scale_factor = split_info.get("scale_factor", 1.6)
        if p_name == "xy":
            samples = split_info["stochastic_samples"]
            flat_old_data[sel] = data_to_split + samples[0]
            grouped_old_data = flat_old_data.reshape(grouped_old_data.shape)
            new_data_split = data_to_split + samples[1]
        elif p_name == "cholesky":
            scaling = torch.sqrt(torch.tensor(scale_factor, device=data_to_split.device))
            flat_old_data[sel] /= scaling
            grouped_old_data = flat_old_data.reshape(grouped_old_data.shape)
            new_data_split = data_to_split / scaling
        else:
            new_data_split = data_to_split
        return grouped_old_data, new_data_split
        

    else:
        raise ValueError(f"Unknown split_method: {split_method}")


def prepare_info(split_method, gs_model, split_info, sel, device, n_split):
    if split_info is None:
        split_info = {}
    if split_method == "principal_axis":
        cholesky_sel = gs_model.get_cholesky_elements.data[sel]
        L = torch.zeros(n_split, 2, 2, device=device)
        L[:, 0, 0] = cholesky_sel[:, 0]
        L[:, 1, 0] = cholesky_sel[:, 1]
        L[:, 1, 1] = cholesky_sel[:, 2]
        Sigma = torch.bmm(L, L.transpose(1, 2))
        eigvals, eigvecs = torch.linalg.eigh(Sigma) # (N, 2), (N, 2, 2)
        v_max = eigvecs[:, :, 1] # (N, 2)
        std_dev = torch.sqrt(eigvals[:, 1]) # (N,)
        
        alpha = split_info.get("alpha", 0.015)
        displacement = alpha * std_dev.unsqueeze(-1) * v_max
        split_info["principal_displacements"] = displacement
        
    if split_method == "stochastic_sample":
        cholesky_sel = gs_model.get_cholesky_elements.data[sel]
        L = torch.zeros(n_split, 2, 2, device=device)
        L[:, 0, 0] = cholesky_sel[:, 0]
        L[:, 1, 0] = cholesky_sel[:, 1]
        L[:, 1, 1] = cholesky_sel[:, 2]
        
        std_normals = 0.015 * torch.randn(2, n_split, 2, device=device) # (2, N, 2)
        samples = torch.einsum("nij,bnj->bni", L, std_normals) # (2, N, 2)
        split_info["stochastic_samples"] = samples


@torch.no_grad()
def split_batch(gs_model, optimizers: Dict[str, torch.optim.Optimizer], mask: torch.Tensor, split_method="simple_duplicate", split_info=None):
    if split_info is None:
        split_info = {}
    
    if split_method is None:
        split_method = "simple_duplicate"

    device = mask.device
    sel = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(device)
    n_split = sel.numel()
    if n_split == 0:
        return 0

    batch_size = gs_model.batch_size
    ori_points_per_image = gs_model.num_points_per_image[0].item()
    points_split_per_image = mask.reshape(batch_size, ori_points_per_image).sum(dim=1)
    assert torch.all(points_split_per_image == points_split_per_image[0]), \
        "The number of points to be split must be the same for each image in the batch."
    per_image_split = int(points_split_per_image[0].item())

    # update num_points_per_image
    gs_model.num_points_per_image = gs_model.num_points_per_image + points_split_per_image

    param_names = ["xy", "cholesky", "features"]
    if gs_model.use_opacity:
        param_names.append("opacity")
    
    prepare_info(split_method, gs_model, split_info, sel, device, n_split)

    for p_name in param_names:
        old_p = gs_model.params[p_name]
        old_data = old_p.data  # shape (total_points, *rest)
        rest_shape = old_data.shape[1:]  # could be ()
        # reshape to (batch, ori_points_per_image, *rest)
        new_shape = (batch_size, ori_points_per_image) + rest_shape
        grouped_old_data = old_data.reshape(new_shape)

        grouped_old_data, new_data_sel = _get_split_params(
            p_name, grouped_old_data, sel, split_method, split_info
        )

        grouped_new_data = torch.cat([grouped_old_data, new_data_sel.reshape(batch_size, per_image_split, *rest_shape)], dim=1)
        new_data = grouped_new_data.reshape(-1, *rest_shape)
        new_p = nn.Parameter(new_data.clone(), requires_grad=old_p.requires_grad)
        gs_model.params[p_name] = new_p

        opt = optimizers.get(p_name)
        assert opt is not None, f"Optimizer for param {p_name} not found."
        for g in opt.param_groups:
            g["params"] = [(new_p if p is old_p else p) for p in g["params"]]

        st_old = opt.state.pop(old_p, None)
        if st_old is None:
            opt.state[new_p] = {}
            continue

        st_new = {}
        for k, v in st_old.items():
            if k == "step" or not isinstance(v, torch.Tensor):
                st_new[k] = v
                continue

            # v shape first dim == total_points usually
            if v.shape[0] == old_data.shape[0]:
                # reshape to (batch, ori_points_per_image, *rest)
                v_grouped = v.reshape((batch_size, ori_points_per_image) + v.shape[1:])
                pad_shape = (batch_size, per_image_split) + v.shape[1:]
                pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                v_grouped = torch.cat([v_grouped, pad], dim=1)
                st_new[k] = v_grouped.reshape(-1, *v.shape[1:])
            else:
                st_new[k] = v.clone()
        opt.state[new_p] = st_new

    if (not gs_model.use_opacity) and hasattr(gs_model, "_opacity"):
        old_buf = getattr(gs_model, "_opacity")
        rest_shape = old_buf.shape[1:] if old_buf.ndim > 1 else ()
        grouped = old_buf.reshape((batch_size, ori_points_per_image) + rest_shape)
        new_buf_sel = old_buf[sel].clone().reshape(batch_size, per_image_split, *rest_shape)
        new_buf = torch.cat([grouped, new_buf_sel], dim=1).reshape(-1, *rest_shape)
        setattr(gs_model, "_opacity", new_buf)

    return n_split