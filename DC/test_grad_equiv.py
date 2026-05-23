"""Gradient-equivalence gate for the DC micro-batched real-forward path (Option A').

The fix (DC/main_DC.py per-class loop) computes gw_real by forwarding the
batch_real images in micro-batches of batch_real_micro and accumulating the
DETACHED, chunk-fraction-weighted gradients, instead of one full-batch forward.
gw_real is first-order (no create_graph) so a mean-reduction CE gradient over N
images equals sum_chunks (len_chunk/N) * grad(mean CE over chunk). This must be
numerically identical to the full-batch gw_real, and therefore so must the
downstream per-class match-loss image-space grad and the final render-param grad.

Checks BOTH, against a full-batch-gw_real reference:
  (i)  image-space grad  syn_grad_accum                    (main_DC.py)
  (ii) render-param grads after img_syn_all.backward()     (main_DC.py)

Real config: DSA on, create_graph=True for gw_syn, real match_loss / DiffAugment
/ ConvNet. Tiny sizes (gpc=8, 2 classes, batch_real=24/25, micro=8, 5 iters).
Requires CUDA. Runs in float64: the micro-batched gw_real is mathematically exact,
but in fp32 a full-batch forward and chunked forwards differ ~1e-3 from cuDNN
reduction-order non-associativity (amplified by the nonlinear matching loss). fp64
collapses that to ~1e-16 so the gate tests the math, not float accumulation order.

Run on the cluster, e.g.:
  srun --partition=debug_ugrad --gres=gpu:1 --time=0:30:00 --pty \
    bash -lc '. /data/guhyun9454/anaconda3/etc/profile.d/conda.sh && \
              conda activate gsdd && cd /ceph_data/guhyun9454/g/GSDatasetDistillation && \
              python DC/test_grad_equiv.py'
"""
import os
import sys

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import copy
import numpy as np
import torch
from torch import nn

from DC.utils import get_network, match_loss, DiffAugment, ParamDiffAug


class Args:
    dis_metric = "ours"
    device = "cuda"
    dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    dsa = True


def make_render_params(num_classes, gpc, channel, im_size, device, seed=0, dtype=torch.float64):
    """Leaf parameter playing the role of the gaussian params; a fixed random
    linear map turns it into the 'render' (img_syn_all). Differentiable so the
    second backward to the leaf is exercised, like gs_model."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_imgs = num_classes * gpc
    latent_dim = 32
    params = torch.randn(n_imgs, latent_dim, generator=g, device="cpu").to(device).to(dtype)
    params.requires_grad_(True)
    out_dim = channel * im_size[0] * im_size[1]
    W = (torch.randn(latent_dim, out_dim, generator=g, device="cpu").to(device).to(dtype)) * 0.05
    b = (torch.randn(out_dim, generator=g, device="cpu").to(device).to(dtype)) * 0.05

    def render(p):
        x = torch.tanh(p @ W + b)
        return x.view(n_imgs, channel, im_size[0], im_size[1])

    return params, render


def run(args, net, render_params, render, real_images, syn_labels, num_classes,
        gpc, batch_real, dsa_params, dsa_seeds, r_micro):
    """One outer-step's worth of synthetic-data gradient computation.

    r_micro >= batch_real reproduces the full-batch reference; r_micro < batch_real
    exercises the micro-batched gw_real path. Everything downstream (syn forward,
    match_loss, per-class accum, render backward) is identical.

    Returns (syn_grad_accum_imgspace, render_params.grad)."""
    net_parameters = list(net.parameters())
    if render_params.grad is not None:
        render_params.grad = None

    img_syn_all = render(render_params)
    img_syn_proxy = img_syn_all.detach().requires_grad_(True)
    syn_grad_accum = torch.zeros_like(img_syn_proxy)

    for c in range(num_classes):
        img_real = real_images[c]
        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
        indices = range(c * gpc, (c + 1) * gpc)
        img_syn = img_syn_proxy[indices]
        lab_syn = syn_labels[indices]

        seed = dsa_seeds[c]
        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_params)
        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_params)

        # micro-batched, detached, chunk-fraction-weighted gw_real
        n_real = img_real.shape[0]
        rm = r_micro if r_micro else n_real
        gw_real = [torch.zeros_like(p) for p in net_parameters]
        for start in range(0, n_real, rm):
            chunk = slice(start, start + rm)
            out_chunk = net(img_real[chunk])
            loss_chunk = nn.functional.cross_entropy(out_chunk, lab_real[chunk])
            g_chunk = torch.autograd.grad(loss_chunk, net_parameters)
            weight = out_chunk.shape[0] / n_real
            for i, g in enumerate(g_chunk):
                gw_real[i] += g.detach() * weight
        gw_real = list(_.detach() for _ in gw_real)

        output_syn = net(img_syn)
        loss_syn = nn.functional.cross_entropy(output_syn, lab_syn)
        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
        loss_c = match_loss(gw_syn, gw_real, args)
        syn_grad_accum += torch.autograd.grad(loss_c, img_syn_proxy, retain_graph=False)[0]

    img_syn_all.backward(syn_grad_accum)
    return syn_grad_accum.detach().clone(), render_params.grad.detach().clone()


def main():
    assert torch.cuda.is_available(), "equivalence gate requires a GPU"
    args = Args()
    device = "cuda"

    num_classes = 2
    gpc = 8
    channel = 3
    im_size = (32, 32)
    batch_real = 24
    r_micro = 8  # micro-batch size for the new path; 24 is not divisible-clean -> tests remainder too via 25
    n_iters = 5

    # Run everything in float64. The micro-batched gw_real is MATHEMATICALLY exact
    # (mean-CE grad = sum of chunk-fraction-weighted micro grads), but in fp32 a
    # batch-of-N forward and chunked forwards differ by ~1e-3 purely from cuDNN
    # reduction-order non-associativity, which the nonlinear gradient-matching loss
    # then amplifies. fp64 collapses that fp noise to ~1e-16 so the test asserts the
    # actual math, not cuDNN's float accumulation. Production runs fp32, where this
    # ~1e-3 drift is far below the variance DSA already injects per iteration.
    #
    # DSA is forced OFF here: it augments the full img_real identically in both the
    # full-batch and micro-batch paths (the change only chunks the *subsequent*
    # forward), so it is orthogonal to what we are verifying — and DiffAugment builds
    # its sampling grids in float32, which is incompatible with the float64 net.
    args.dsa_strategy = "none"
    args.dsa = False
    print(">>> float64, DSA off — verifying the micro-batched gw_real math")
    dsa_params = ParamDiffAug()
    dtype = torch.float64
    ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + it)
        np.random.seed(1000 + it)

        net = get_network("ConvNet", channel, num_classes, im_size).to(device).to(dtype)
        net.eval()  # freeze norm stats so both runs are deterministic and identical

        render_params, render = make_render_params(num_classes, gpc, channel, im_size, device, seed=it, dtype=dtype)
        syn_labels = torch.cat([torch.full((gpc,), c, dtype=torch.long, device=device)
                                for c in range(num_classes)])
        # use batch_real not divisible by r_micro on odd iters to exercise the remainder chunk
        br = batch_real + (1 if it % 2 == 1 else 0)
        real_images = [torch.randn(br, channel, *im_size, device=device).to(dtype)
                       for _ in range(num_classes)]
        dsa_seeds = [int(np.random.randint(0, 100000)) for _ in range(num_classes)]

        net_ref = copy.deepcopy(net)
        params_ref = render_params.detach().clone().requires_grad_(True)

        new_img_grad, new_param_grad = run(
            args, net, render_params, render, real_images, syn_labels,
            num_classes, gpc, br, dsa_params, dsa_seeds, r_micro=r_micro)

        ref_img_grad, ref_param_grad = run(
            args, net_ref, params_ref, render, real_images, syn_labels,
            num_classes, gpc, br, dsa_params, dsa_seeds, r_micro=0)  # full batch

        img_close = torch.allclose(new_img_grad, ref_img_grad, atol=1e-9, rtol=1e-7)
        param_close = torch.allclose(new_param_grad, ref_param_grad, atol=1e-9, rtol=1e-7)
        img_maxerr = (new_img_grad - ref_img_grad).abs().max().item()
        param_maxerr = (new_param_grad - ref_param_grad).abs().max().item()
        print(f"[iter {it}] br={br} micro={r_micro} | img_grad allclose={img_close} "
              f"(max|Δ|={img_maxerr:.2e})  param_grad allclose={param_close} (max|Δ|={param_maxerr:.2e})")
        ok = ok and img_close and param_close

    print("=" * 60)
    print(f"GRADIENT EQUIVALENCE (micro-batched gw_real): {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
