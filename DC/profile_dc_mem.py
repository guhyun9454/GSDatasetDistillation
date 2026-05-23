"""Peak-memory / runtime profiler for the DC per-class second-order matching loop
at the paper config (gpc=200, num_points=54, batch_real=720, batch_syn=0,
imagenette 128x128, ConvNetD5). Compares the checkpointed syn forward (Option A,
current main_DC) against a no-checkpoint baseline to show the memory delta.

The synthetic render is a learnable leaf tensor of the real shape
[num_classes*gpc, 3, 128, 128]; the dominant memory/runtime term is the
per-class second-order graph through ConvNetD5, which does not depend on the
specific gaussian-init values, so this measures the binding cost faithfully
without needing the (not-yet-staged) gaussian init for this config.

Requires CUDA (one 24GB A5000). Reports peak GPU memory and seconds/iteration.
"""
import os
import sys
import time

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from DC.utils import get_network, match_loss, DiffAugment, ParamDiffAug


class Args:
    dis_metric = "ours"
    device = "cuda"
    dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    dsa = True


def one_iter(args, net, render_params, real_images, syn_labels, num_classes,
             gpc, batch_real, dsa_params, use_checkpoint):
    net_parameters = list(net.parameters())
    if render_params.grad is not None:
        render_params.grad = None

    img_syn_all = torch.tanh(render_params)  # cheap differentiable "render"
    img_syn_proxy = img_syn_all.detach().requires_grad_(True)
    syn_grad_accum = torch.zeros_like(img_syn_proxy)

    for c in range(num_classes):
        img_real = real_images  # reuse one real batch per class (shape is what matters)
        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
        indices = range(c * gpc, (c + 1) * gpc)
        img_syn = img_syn_proxy[indices]
        lab_syn = syn_labels[indices]

        seed = int(time.time() * 1000) % 100000
        img_real_a = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_params)
        img_syn_a = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_params)

        output_real = net(img_real_a)
        loss_real = nn.functional.cross_entropy(output_real, lab_real)
        gw_real = torch.autograd.grad(loss_real, net_parameters)
        gw_real = list(_.detach().clone() for _ in gw_real)

        if use_checkpoint:
            output_syn = torch.utils.checkpoint.checkpoint(
                lambda x: net(x), img_syn_a, use_reentrant=False)
        else:
            output_syn = net(img_syn_a)
        loss_syn = nn.functional.cross_entropy(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
        loss_c = match_loss(gw_syn, gw_real, args)
        syn_grad_accum += torch.autograd.grad(loss_c, img_syn_proxy, retain_graph=False)[0]

    img_syn_all.backward(syn_grad_accum)


def profile(use_checkpoint, n_warmup=1, n_meas=3):
    args = Args()
    dsa_params = ParamDiffAug()
    device = "cuda"

    num_classes = 10
    gpc = 200
    channel = 3
    im_size = (128, 128)
    batch_real = 720

    net = get_network("ConvNetD5", channel, num_classes, im_size).to(device)
    net.train()
    for m in net.modules():
        if 'BatchNorm' in m._get_name():
            m.eval()

    render_params = torch.randn(num_classes * gpc, channel, *im_size, device=device,
                                requires_grad=True) * 0.1
    render_params = render_params.detach().requires_grad_(True)
    syn_labels = torch.cat([torch.full((gpc,), c, dtype=torch.long, device=device)
                            for c in range(num_classes)])
    real_images = torch.randn(batch_real, channel, *im_size, device=device)

    for _ in range(n_warmup):
        one_iter(args, net, render_params, real_images, syn_labels, num_classes,
                 gpc, batch_real, dsa_params, use_checkpoint)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(n_meas):
        one_iter(args, net, render_params, real_images, syn_labels, num_classes,
                 gpc, batch_real, dsa_params, use_checkpoint)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_meas
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return peak_gb, dt


def main():
    assert torch.cuda.is_available(), "profiler requires a GPU"
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} ({total_gb:.1f} GiB)")
    print("Paper DC config: gpc=200, num_points=54(N/A here), batch_real=720, "
          "batch_syn=0, imagenette 128x128, ConvNetD5\n")

    print("--- Option A: gradient-checkpointed syn forward (current main_DC) ---")
    try:
        peak_a, dt_a = profile(use_checkpoint=True)
        print(f"peak GPU mem = {peak_a:.2f} GiB | sec/iter = {dt_a:.2f}s\n")
    except RuntimeError as e:
        print(f"OOM/RuntimeError with checkpoint: {e}\n")
        peak_a, dt_a = None, None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("--- Baseline: NO checkpoint (for delta reference) ---")
    try:
        peak_b, dt_b = profile(use_checkpoint=False)
        print(f"peak GPU mem = {peak_b:.2f} GiB | sec/iter = {dt_b:.2f}s\n")
    except RuntimeError as e:
        print(f"OOM/RuntimeError without checkpoint: {e}\n")
        peak_b, dt_b = None, None

    print("=" * 60)
    if peak_a is not None:
        fits = "FITS" if peak_a < total_gb else "DOES NOT FIT"
        print(f"Checkpoint path: {peak_a:.2f} GiB -> {fits} on {total_gb:.0f}GB")
    if peak_a and peak_b:
        print(f"Memory saved by checkpoint: {peak_b - peak_a:.2f} GiB "
              f"({100*(peak_b-peak_a)/peak_b:.0f}%)")
        print(f"Runtime overhead: {100*(dt_a-dt_b)/dt_b:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
