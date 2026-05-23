import copy
import gc
import os
import sys

from torch import nn
sys.path.append(".")
import warnings
import copy
import random
import logging
import functools
import time
from typing import Callable, Dict, Any, List

from omegaconf import OmegaConf
import torchvision

from lib_ddif.utils import build_tensor_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(".")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lib.utils").setLevel(logging.WARNING)
logging.getLogger('lib.gaussian.gaussianimage_cholesky').setLevel(logging.WARNING)

import hydra
import numpy as np
import torch
import torch.utils.checkpoint
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

from DC.utils import get_daparam, get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, match_loss, set_seed, save_and_print, get_images, get_loops
from DC.hyper_params import load_default
from lib.utils import get_initialized_gs_batch
from lib.gaussian.strategy_batch import DefaultStrategy
from lib.gaussian.gs_utils import create_optimizers_and_schedulers


logger = logging.getLogger(__name__)


def move_module_all_tensors_to_cpu(module: torch.nn.Module):
    module = module.to("cpu")

    for name, value in vars(module).items():
        if torch.is_tensor(value) and value.device.type != "cpu":
            setattr(module, name, value.cpu())
        elif isinstance(value, (list, tuple)):
            new_list = []
            changed = False
            for v in value:
                if torch.is_tensor(v) and v.device.type != "cpu":
                    new_list.append(v.cpu())
                    changed = True
                else:
                    new_list.append(v)
            if changed:
                if isinstance(value, tuple):
                    setattr(module, name, tuple(new_list))
                else:
                    setattr(module, name, new_list)
        elif isinstance(value, dict):
            new_dict = {}
            changed = False
            for k, v in value.items():
                if torch.is_tensor(v) and v.device.type != "cpu":
                    new_dict[k] = v.cpu()
                    changed = True
                else:
                    new_dict[k] = v
            if changed:
                setattr(module, name, new_dict)
    return module


class MultiMethodOutputHook:

    def __init__(
        self,
        module: torch.nn.Module,
        methods: List[str],
        transform: Callable,
        key: str = "render",
    ):
        self.module = module
        self.methods = methods
        self.transform = transform
        self.key = key
        self._orig = {}

        for name in methods:
            orig = getattr(module, name, None)
            if orig is None or not callable(orig):
                raise AttributeError(
                    f"{module.__class__.__name__} has no callable '{name}'"
                )

            self._orig[name] = orig

            setattr(module, name, self._make_wrapper(orig))

    def _make_wrapper(self, orig_fn: Callable):
        @functools.wraps(orig_fn)
        def wrapper(*args, **kwargs):
            out = orig_fn(*args, **kwargs)
            if not isinstance(out, dict):
                return out
            if self.key not in out:
                return out

            x = out[self.key]
            if hasattr(self.transform, "to"):
                try:
                    self.transform.to(x.device)
                except Exception:
                    pass
            y = self.transform(x)

            new_out = dict(out)
            new_out[self.key] = y
            return new_out

        return wrapper

    def remove(self):
        for name, fn in self._orig.items():
            setattr(self.module, name, fn)
        self._orig.clear()


def attach_output_transform_to_methods(
    module: torch.nn.Module,
    train_transform: Callable,
    methods: List[str],
    key: str = "render",
) -> MultiMethodOutputHook:
    return MultiMethodOutputHook(module, methods, train_transform, key)


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


def setup_logging_for_distributed(is_master: bool):
    logger = logging.getLogger()

    if is_master:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logger.info("Logging is configured for the master process.")
    else:
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())


def check_args(args):
    check_keys = {
        "use_clamp",
        "boundary_loss_type",
        "boundary_loss_lambda",
    }
    for key in check_keys:
        if key not in args:
            raise ValueError(f"Missing required argument: {key}")

    # try to access min_start_epoch, if not exist, set default
    if not hasattr(args, "min_start_epoch"):
        logger.warning("min_start_epoch not found in args, setting to default 0.")
        args.min_start_epoch = 0

    if not hasattr(args, "forward_subset"):
        logger.warning("forward_subset not found in args, setting to default False.")
        args.forward_subset = False


@hydra.main(config_path="configs", config_name="imagenette", version_base="1.3")
def main(args):
    OmegaConf.set_readonly(args, False)
    OmegaConf.set_struct(args, False)

    check_args(args)

    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_path = save_dir

    logger.info("Setting up file logging for rank 0.")
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    err_path = os.path.join(args.save_path, f"{file_name}.log")
    err_file = open(err_path, "a", buffering=1)
    sys.stderr = Tee(sys.__stderr__, err_file)
    args.log_path = err_path

    set_seed(args.seed)
    args = load_default(args)

    if getattr(args, "wandb", False):
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=OmegaConf.to_container(args, resolve=True),
        )

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/imgs", exist_ok=True)

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    dsa_params = ParamDiffAug()

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    args.im_size = im_size

    if not args.zca:
        train_transform = torchvision.transforms.Normalize(mean=mean, std=std)
    else:
        train_transform = zca_trans
    if args.zca:
        zca_trans = zca_trans
        zca_trans_cpu = copy.deepcopy(zca_trans).to('cpu')
        zca_trans_cpu = move_module_all_tensors_to_cpu(zca_trans_cpu)
    else:
        zca_trans = None
        zca_trans_cpu = None

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    eval_it_pool = np.arange(0, args.Iteration+1, args.eval_it).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    
    
    save_and_print(args.log_path, f"Hyper-parameters: {args.__dict__}")
    save_and_print(args.log_path, f"Evaluation model pool: {model_eval_pool}")
    save_and_print(args.log_path, OmegaConf.to_yaml(args, resolve=True))


    if args.boundary_loss_type is not None:
        if args.boundary_loss_type == "exponent":
            def boundary_loss(x):
                exponent = args.boundary_loss_exponent
                assert exponent > 0 and exponent % 2 == 0, "Exponent must be a positive even integer."
                return torch.mean(x.pow(exponent))

        elif args.boundary_loss_type == "log":
            def boundary_loss(x, epsilon=1e-6):
                clamped_xy = torch.clamp(x, min=-1.0 + epsilon, max=1.0 - epsilon)
                log_barrier = -torch.log(1.0 - clamped_xy.pow(2))
                return torch.mean(log_barrier)
        else:
            raise ValueError(f"Unsupported boundary loss type: {args.boundary_loss_type}")
    else:
        def boundary_loss(x):
            return torch.tensor(0.0, device=x.device)

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        ''' organize the real dataset '''
        save_and_print(args.log_path, "BUILDING DATASET")
        images_all, labels_all = build_tensor_dataset(dst_train, batch_size=args.batch_real, workers=args.workers, class_map=class_map)

        indices_class = [[] for c in range(num_classes)]
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)


        ''' initialize the synthetic data '''
        syn_labels = np.array([np.ones(args.gpc, dtype=np.int_)*i for i in range(num_classes)])
        syn_labels = torch.tensor(syn_labels, dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        gs_model, sample_index = get_initialized_gs_batch(num_classes, args.gaussian.batch_size, args.gs_dir, args.gs_type, args.gaussian, device=device, epochs=None)
        gs_model.requires_grad_(True)
        methods = [m for m in ["forward", "forward_subset", "crop_forward_loop", "crop_forward_padding"]
                   if callable(getattr(gs_model, m, None))]
        hook_handle = attach_output_transform_to_methods(
            gs_model, train_transform, methods, key="render"
        )

        optimizers, schedulers = create_optimizers_and_schedulers(gs_model, args)
        strategy = DefaultStrategy(args.strategy, gs_model, optimizers)

        scaler = torch.amp.GradScaler(enabled=(gs_model.precision == "fp16"))

        ''' training '''
        criterion = nn.CrossEntropyLoss().to(args.device)

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        save_and_print(args.log_path, '%s training begins'%get_time())

        for it in range(args.Iteration+1):
            save_this_it = False

            gc.collect()
            torch.cuda.empty_cache()
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        # args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                        save_and_print(args.log_path, f'DSA augmentation parameters: {dsa_params.__dict__}')
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                    # if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    #     args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    # else:
                    #     args.epoch_eval_train = 300

                    accs_test = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        label_syn_eval = syn_labels.clone()
                        with torch.inference_mode():
                            image_syn_eval = gs_model()["render"]
                        
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, dsa_param=dsa_params)
                        accs_test.append(acc_test)

                    accs_test = np.array(accs_test)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                        torch.save({"best_acc": best_acc, "best_std": best_std}, f"{args.save_path}/best_performance.pt")
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    save_and_print(args.log_path, f"{args.save_path}")
                    save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                    if getattr(args, "wandb", False):
                        import wandb
                        wandb.log({
                            f"Accuracy/{model_eval}": acc_test_mean,
                            f"Max_Accuracy/{model_eval}": best_acc[model_eval],
                        }, step=it)
                    save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")
                    if args.num_eval > 0:
                        del image_syn_eval, label_syn_eval

                ''' visualize and save '''
                save_name = os.path.join(f"{args.save_path}/imgs", 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                # image_syn_vis, _ = synset.get(need_copy=True)
                image_syn_vis = gs_model()["render"].detach().clone()
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=10) # Trying normalize = True/False may get better visual effects.
                del image_syn_vis

                if save_this_it:
                    save_dict = {
                        "model": gs_model,
                        "syn_lr": args.lr_net,
                    }
                    hook_handle.remove()
                    torch.save(save_dict, os.path.join(args.save_path, f"GSDD_TM_{args.ipc}ipc#synset_best.pt"))
                    hook_handle = attach_output_transform_to_methods(
                        gs_model, train_transform, methods, key="render"
                    )
                save_dict = {
                    "model": gs_model,
                    "syn_lr": args.lr_net,
                }
                hook_handle.remove()
                torch.save(save_dict, os.path.join(args.save_path, f"GSDD_TM_{args.ipc}ipc_iter{it}.pt"))
                hook_handle = attach_output_transform_to_methods(
                    gs_model, train_transform, methods, key="render"
                )
                # if args.clear_memory:
                gc.collect()
                torch.cuda.empty_cache()

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(images_all, indices_class, c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                # Memory-efficient gradient matching for 24 GB GPUs.
                # Render the synthetic set once, then for each class take d(loss_c)/d(render)
                # through a detached proxy so each class's (second-order) network graph is
                # freed immediately. The accumulated image-space gradient is backpropped
                # through the single render graph at the end. This is mathematically identical
                # to summing the per-class match losses and calling one backward, but holds
                # only one class graph at a time instead of num_classes (lets batch_syn=0 fit
                # in 24 GB). Assumes the GradScaler is disabled (true for bf16/fp32).
                img_syn_all = gs_model()["render"]
                img_syn_proxy = img_syn_all.detach().requires_grad_(True)
                syn_grad_accum = torch.zeros_like(img_syn_proxy)
                loss = 0.0
                for c in range(num_classes):
                    img_real = get_images(images_all, indices_class, c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_real, lab_real = img_real.to(args.device), lab_real.to(args.device)

                    if args.batch_syn > 0:
                        indices = np.random.permutation(range(c * args.gpc, (c + 1) * args.gpc))[:args.batch_syn]
                    else:
                        indices = range(c * args.gpc, (c + 1) * args.gpc)

                    img_syn = img_syn_proxy[indices]
                    lab_syn = syn_labels[indices]

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_params)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_params)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    # Checkpoint the syn forward so its activations are recomputed during the
                    # second-order backward instead of stored, cutting the dominant memory term.
                    # BN must be frozen (.eval()) here so recompute reuses fixed running stats.
                    assert all(
                        not module.training
                        for module in net.modules()
                        if 'BatchNorm' in module._get_name()
                    ), "BatchNorm must be in eval() before the checkpointed syn forward"
                    output_syn = torch.utils.checkpoint.checkpoint(
                        lambda x: net(x), img_syn, use_reentrant=False
                    )
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss_c = match_loss(gw_syn, gw_real, args)
                    syn_grad_accum += torch.autograd.grad(loss_c, img_syn_proxy, retain_graph=False)[0]
                    loss += loss_c.item()

                for name in optimizers:
                    optimizers[name].zero_grad(set_to_none=True)

                # boundary loss depends on the gaussian positions directly (not through the
                # render); the original summed it once per class, so keep the num_classes factor.
                boundary_total = num_classes * args.boundary_loss_lambda * boundary_loss(gs_model.params["xy"])

                img_syn_all.backward(syn_grad_accum)
                boundary_total.backward()
                loss += boundary_total.item()

                for name in optimizers:
                    scaler.unscale_(optimizers[name])
                strategy.step_pre_backward(it)

                # synset.optim_step()
                for name in optimizers:
                    scaler.step(optimizers[name])
                    if args.scheduler_gs.type is not None:
                        schedulers[name].step()

                scaler.update()

                if args.use_clamp:
                    gs_model.clamp()

                strategy.step_post_backward(it)
                loss_avg += loss

                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train = gs_model()["render"].detach()
                label_syn_train = syn_labels.clone()
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

            loss_avg /= (num_classes*args.outer_loop)

            if it == 0 and torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                save_and_print(args.log_path,
                               '[profile] iter0 peak GPU mem = %.2f GiB' % peak_gb)

            if it%10 == 0:
                save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))


if __name__ == '__main__':
    main()
