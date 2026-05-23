"""Gradient-equivalence gate for the DC checkpointed per-class grad-accum path.

Asserts the new path (render once -> detached proxy -> per-class
d(match_loss_c)/d(proxy) accumulated, syn forward gradient-checkpointed,
then img_syn_all.backward(accum)) is numerically identical to the literal
pre-c068d30 reference (sum per-class match losses -> ONE backward to the
render-producing leaf params).

Checks BOTH:
  (i)  image-space grad  syn_grad_accum                    (main_DC.py:464)
  (ii) leaf (render param) grads after img_syn_all.backward (main_DC.py:474)

Real config: DSA on, create_graph=True, real match_loss / DiffAugment /
ConvNet w/ BatchNorm. Tiny sizes (gpc=8, 2 classes, 5 iters) so it fits a
debug GPU. Requires CUDA.

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


def make_render_params(num_classes, gpc, channel, im_size, device, seed=0):
    """A leaf parameter that plays the role of the gaussian params; a fixed
    random linear map turns it into the 'render' (img_syn_all). Differentiable
    so the second backward to the leaf is exercised, exactly like gs_model."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_imgs = num_classes * gpc
    latent_dim = 32
    params = torch.randn(n_imgs, latent_dim, generator=g, device="cpu").to(device)
    params.requires_grad_(True)
    out_dim = channel * im_size[0] * im_size[1]
    W = torch.randn(latent_dim, out_dim, generator=g, device="cpu").to(device) * 0.05
    b = torch.randn(out_dim, generator=g, device="cpu").to(device) * 0.05

    def render(p):
        x = torch.tanh(p @ W + b)
        return x.view(n_imgs, channel, im_size[0], im_size[1])

    return params, render


def run(args, net, render_params, render, real_images, syn_labels,
        num_classes, gpc, batch_real, dsa_params, dsa_seeds, mode):
    """One outer-step's worth of synthetic-data gradient computation.

    mode='ref'  : pre-c068d30 path (sum match losses, one backward).
    mode='new'  : checkpointed per-class grad-accum path (current main_DC).

    Returns (syn_grad_accum_imgspace, render_params.grad)."""
    net_parameters = list(net.parameters())

    # Freeze BN running stats (mirrors main_DC :417-421) using the real images.
    for module in net.modules():
        if 'BatchNorm' in module._get_name():
            net.train()
            break
    _ = net(torch.cat([real_images[c][:16] for c in range(num_classes)], dim=0))
    for module in net.modules():
        if 'BatchNorm' in module._get_name():
            module.eval()

    if render_params.grad is not None:
        render_params.grad = None

    img_syn_all = render(render_params)

    if mode == "new":
        img_syn_proxy = img_syn_all.detach().requires_grad_(True)
        syn_grad_accum = torch.zeros_like(img_syn_proxy)
        for c in range(num_classes):
            img_real = real_images[c]
            lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
            indices = range(c * gpc, (c + 1) * gpc)
            img_syn = img_syn_proxy[indices]
            lab_syn = syn_labels[indices]
            seed = dsa_seeds[c]
            img_real_a = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_params)
            img_syn_a = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_params)

            output_real = net(img_real_a)
            loss_real = nn.functional.cross_entropy(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, net_parameters)
            gw_real = list(_.detach().clone() for _ in gw_real)

            assert all(not m.training for m in net.modules() if 'BatchNorm' in m._get_name())
            output_syn = torch.utils.checkpoint.checkpoint(
                lambda x: net(x), img_syn_a, use_reentrant=False
            )
            loss_syn = nn.functional.cross_entropy(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
            loss_c = match_loss(gw_syn, gw_real, args)
            syn_grad_accum += torch.autograd.grad(loss_c, img_syn_proxy, retain_graph=False)[0]

        img_syn_all.backward(syn_grad_accum)
        return syn_grad_accum.detach().clone(), render_params.grad.detach().clone()

    else:  # reference
        loss = 0.0
        proxy = img_syn_all.detach().requires_grad_(True)  # only to read image-space grad
        ref_syn_grad = torch.zeros_like(proxy)
        for c in range(num_classes):
            img_real = real_images[c]
            lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
            indices = range(c * gpc, (c + 1) * gpc)
            lab_syn = syn_labels[indices]
            seed = dsa_seeds[c]
            img_real_a = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_params)

            # match loss directly against img_syn_all (single graph, one backward)
            img_syn = img_syn_all[indices]
            img_syn_a = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_params)
            output_real = net(img_real_a)
            loss_real = nn.functional.cross_entropy(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, net_parameters)
            gw_real = list(_.detach().clone() for _ in gw_real)
            output_syn = net(img_syn_a)
            loss_syn = nn.functional.cross_entropy(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
            loss = loss + match_loss(gw_syn, gw_real, args)

            # separately compute image-space grad for the (i) comparison
            img_syn_p = proxy[indices]
            img_syn_pa = DiffAugment(img_syn_p, args.dsa_strategy, seed=seed, param=dsa_params)
            output_syn_p = net(img_syn_pa)
            loss_syn_p = nn.functional.cross_entropy(output_syn_p, lab_syn)
            gw_syn_p = torch.autograd.grad(loss_syn_p, net_parameters, create_graph=True)
            loss_c_p = match_loss(gw_syn_p, gw_real, args)
            ref_syn_grad += torch.autograd.grad(loss_c_p, proxy, retain_graph=False)[0]

        loss.backward()
        return ref_syn_grad.detach().clone(), render_params.grad.detach().clone()


def main():
    assert torch.cuda.is_available(), "equivalence gate requires a GPU"
    args = Args()
    dsa_params = ParamDiffAug()
    device = "cuda"

    num_classes = 2
    gpc = 8
    channel = 3
    im_size = (32, 32)
    batch_real = 24
    n_iters = 5

    ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + it)
        np.random.seed(1000 + it)

        net = get_network("ConvNet", channel, num_classes, im_size).to(device)
        net.train()

        # fresh leaf params + render for each iter (identical across both paths)
        render_params, render = make_render_params(num_classes, gpc, channel, im_size, device, seed=it)
        syn_labels = torch.cat([torch.full((gpc,), c, dtype=torch.long, device=device)
                                for c in range(num_classes)])
        real_images = [torch.randn(batch_real, channel, *im_size, device=device)
                       for _ in range(num_classes)]
        dsa_seeds = [int(np.random.randint(0, 100000)) for _ in range(num_classes)]

        # deepcopy nets so the BN-stat pass is identical for both runs
        import copy
        net_ref = copy.deepcopy(net)
        params_ref = render_params.detach().clone().requires_grad_(True)

        def render_ref(p):
            return render(p)

        new_img_grad, new_param_grad = run(
            args, net, render_params, render, real_images, syn_labels,
            num_classes, gpc, batch_real, dsa_params, dsa_seeds, mode="new")

        ref_img_grad, ref_param_grad = run(
            args, net_ref, params_ref, render_ref, real_images, syn_labels,
            num_classes, gpc, batch_real, dsa_params, dsa_seeds, mode="ref")

        img_close = torch.allclose(new_img_grad, ref_img_grad, atol=1e-4, rtol=1e-3)
        param_close = torch.allclose(new_param_grad, ref_param_grad, atol=1e-4, rtol=1e-3)
        img_maxerr = (new_img_grad - ref_img_grad).abs().max().item()
        param_maxerr = (new_param_grad - ref_param_grad).abs().max().item()
        print(f"[iter {it}] img_grad allclose={img_close} (max|Δ|={img_maxerr:.2e})  "
              f"param_grad allclose={param_close} (max|Δ|={param_maxerr:.2e})")
        ok = ok and img_close and param_close

    print("=" * 60)
    print(f"GRADIENT EQUIVALENCE: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
