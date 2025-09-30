import gc
import os
import sys
import warnings
import copy
import random
import logging
import functools
from typing import Callable, Dict, Any, List
sys.path.append('./lib_ddif')
sys.path.append('./lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lib.utils").setLevel(logging.WARNING)
logging.getLogger('lib.gaussian.gaussianimage_cholesky').setLevel(logging.WARNING)

import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector
from torch.distributed.nn import functional as dfunc
from omegaconf import OmegaConf

from lib.utils import get_initialized_gs_batch 
from lib.gaussian.strategy_batch import DefaultStrategy
from lib.gaussian.gs_utils import create_optimizers_and_schedulers
from lib_ddif.reparam_module import ReparamModule
from lib_ddif.hyper_params import load_default
from lib_ddif.utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, set_seed, save_and_print, build_tensor_dataset
from tqdm import tqdm

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
    def __init__(self, module: torch.nn.Module, methods: List[str],
                 transform: Callable, key: str = "render"):
        self.module = module
        self.methods = methods
        self.transform = transform
        self.key = key
        self._orig = {}

        for name in methods:
            orig = getattr(module, name, None)
            if orig is None or not callable(orig):
                raise AttributeError(f"{module.__class__.__name__} has no callable '{name}'")

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

logger = logging.getLogger(__name__)
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
            handlers=[logging.StreamHandler()]
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

@hydra.main(config_path="configs/distill", config_name="base`", version_base="1.3")
def main(args):
    OmegaConf.set_readonly(args, False)
    OmegaConf.set_struct(args, False)

    check_args(args)

    # log DDP information

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

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/imgs", exist_ok=True)


    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    save_and_print(args.log_path, "CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = f"cuda"    
    device = args.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    if not args.zca:
        train_transform = torchvision.transforms.Normalize(mean=mean, std=std)
    else:
        train_transform = zca_trans
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.im_size = im_size

    accs_all_exps = dict()
    for key in model_eval_pool:
        accs_all_exps[key] = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    dsa_params = ParamDiffAug()

    if args.zca:
        zca_trans = zca_trans
        zca_trans_cpu = copy.deepcopy(zca_trans).to('cpu')
        zca_trans_cpu = move_module_all_tensors_to_cpu(zca_trans_cpu)
    else:
        zca_trans = None
        zca_trans_cpu = None

    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')
    save_and_print(args.log_path, f'Evaluation model pool: {model_eval_pool}')
    save_and_print(args.log_path, OmegaConf.to_yaml(args, resolve=True))

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
    methods = ["forward", "forward_subset", "crop_forward_loop", "crop_forward_padding"]
    hook_handle = attach_output_transform_to_methods(
        gs_model, train_transform, methods, key="render"
    )
    optimizers, schedulers = create_optimizers_and_schedulers(gs_model, args)
    strategy = DefaultStrategy(args.strategy, gs_model, optimizers)

    if args.batch_syn == 0:
        args.batch_syn = gs_model.batch_size

    ''' training '''
    syn_lr = torch.tensor(args.lr_teacher, device=device, dtype=torch.float32)
    syn_lr = syn_lr.detach().to(device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(device)
    save_and_print(args.log_path, '%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "imagenet":
        subset_names = {"imagenette": "imagenette", "imagewoof": "imagewoof", "imagefruit": "imagefruit", "imageyellow": "imageyellow", "imagemeow": "imagemeow", "imagesquawk": "imagesquawk"}
        expert_dir = os.path.join(expert_dir, subset_names[args.subset])
    if not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    save_and_print(args.log_path, "Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        paths = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            paths.append(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if not paths:
            raise AssertionError(f"No buffers detected at {expert_dir}")

        # load with a tqdm progress bar
        for path in tqdm(paths, desc="Loading replay buffers"):
            buffer += torch.load(path, weights_only=False, map_location='cpu')
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        save_and_print(args.log_path, "loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx], weights_only=False)
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

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
    
    scaler = torch.amp.GradScaler(enabled=(gs_model.precision == "fp16"))
    save_and_print(args.log_path, f"Training with precision: {gs_model.precision}. GradScaler enabled: {scaler.is_enabled()}")

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    del images_all, labels_all

    if args.clear_memory:
        gc.collect()
        torch.cuda.empty_cache()

    for it in range(0, args.Iteration+1):
        save_this_it = False

        # ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            # if args.clear_memory:
            gc.collect()
            torch.cuda.empty_cache()

            eval_seed = args.seed
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)
            np.random.seed(eval_seed)
            random.seed(eval_seed)

            for model_eval in model_eval_pool:
                save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {dsa_params.__dict__}')
                else:
                    save_and_print(args.log_path, f'DC augmentation parameters: {args.dc_aug_param}')

                local_accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)

                    label_syn_eval = syn_labels
                    with torch.inference_mode():
                        image_syn_eval = gs_model()["render"]

                    args.lr_net = syn_lr.item()
                    _, _, acc_test = evaluate_synset(it, net_eval, image_syn_eval, label_syn_eval, testloader, args, dsa_param=dsa_params)
                    local_accs_test.append(acc_test)

                    if args.clear_memory:
                        net_eval.cpu()
                        del image_syn_eval, label_syn_eval, net_eval
                        gc.collect()
                        torch.cuda.empty_cache()


                final_accs_test = local_accs_test
                final_accs_test = np.array(final_accs_test)
                acc_test_mean = np.mean(final_accs_test)
                acc_test_std = np.std(final_accs_test)

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                    torch.save({"best_acc": best_acc, "best_std": best_std}, f"{args.save_path}/best_performance.pt")

                save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(final_accs_test), model_eval, acc_test_mean, acc_test_std))
                save_and_print(args.log_path, f"{args.save_path}")
                save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")

            flag_tensor = torch.tensor([1 if save_this_it else 0], dtype=torch.int, device=args.device)


            save_and_print(args.log_path, f"{it:5d} | Synthetic_LR: {syn_lr.detach().cpu()}")

            if save_this_it:
                save_dict = {
                    "model": gs_model,
                    "syn_lr": syn_lr.detach().cpu(),
                }
                hook_handle.remove()
                torch.save(save_dict, os.path.join(args.save_path, f"GSDD_TM_{args.ipc}ipc#synset_best.pt"))
                hook_handle = attach_output_transform_to_methods(
                    gs_model, train_transform, methods, key="render"
                )
            save_dict = {
                "model": gs_model,
                "syn_lr": syn_lr.detach().cpu(),
            }
            hook_handle.remove()
            torch.save(save_dict, os.path.join(args.save_path, f"GSDD_TM_{args.ipc}ipc_iter{it}.pt"))
            hook_handle = attach_output_transform_to_methods(
                gs_model, train_transform, methods, key="render"
            )
            gc.collect()
            torch.cuda.empty_cache()

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.inference_mode():
                label_save = syn_labels.detach().cpu()
                image_save = gs_model()["render"].detach().cpu()

                if save_this_it:
                    torch.save(image_save, os.path.join(args.save_path, "images_best.pt".format(it)))
                    torch.save(label_save, os.path.join(args.save_path, "labels_best.pt".format(it)))

                save_dir = f"{args.save_path}/imgs"

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if num_classes > 10:
                        classes_save = np.random.permutation(num_classes)[:min(10, num_classes)]
                    else:
                        classes_save = np.arange(num_classes)
                    indices_save = np.concatenate([c*args.gpc+np.arange(min(10, args.gpc)) for c in classes_save])
                    upsampled = upsampled[indices_save]
                    if args.dataset != "imagenet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
                    plt.savefig(f"{save_dir}/Synthetic_Images#{it}.png", dpi=300)
                    plt.close()
                    del grid, upsampled

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        upsampled = upsampled[indices_save]
                        if args.dataset != "imagenet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Clipped_Synthetic_Images#{it}.png", dpi=300)
                        plt.close()
                        del upsampled, grid

                    if args.zca:
                        image_save = zca_trans_cpu.inverse_transform(image_save)

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        upsampled = upsampled[indices_save]
                        if args.dataset != "imagenet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
                        plt.savefig(f"{save_dir}/Reconstructed_Images#{it}.png", dpi=300)
                        plt.close()

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            upsampled = upsampled[indices_save]
                            if args.dataset != "imagenet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=len(classes_save), normalize=True, scale_each=True)
                            plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
                            plt.savefig(f"{save_dir}/Clipped_Reconstructed_Images#{it}.png", dpi=300)
                            plt.close()

                    del image_save, label_save
            gc.collect()
            torch.cuda.empty_cache()

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx], weights_only=False, map_location='cpu')
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)

        snap_s = expert_trajectory[start_epoch]
        snap_t = expert_trajectory[start_epoch+args.expert_epochs]
        starting_params = parameters_to_vector(snap_s).to(device, non_blocking=True)
        target_params = parameters_to_vector(snap_t).to(device, non_blocking=True)
        student_params = [starting_params.clone().detach().to(device).requires_grad_(True)]

        indices_total = torch.randperm(gs_model.batch_size)[:args.syn_steps * args.batch_syn]
        if args.forward_subset:
            image_syn = gs_model.forward_subset(indices_total)["render"]
        else:
            image_syn = gs_model.forward()["render"][indices_total]
        label_syn = syn_labels[indices_total]

        syn_images = image_syn.cuda(device, non_blocking=True)
        y_hat = label_syn.cuda(device, non_blocking=True)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []


        for step in range(args.syn_steps):
            if not indices_chunks:
                g = torch.Generator(device=device)
                g.manual_seed(it * 1000 + step)
                indices = torch.randperm(len(syn_images), generator=g, device=device)
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            t = y_hat[these_indices]

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=dsa_params)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            out = student_net(x, flat_param=forward_params)
            ce_loss = criterion(out, t)

            grad = torch.autograd.grad(ce_loss,
                                    student_params[-1],
                                    create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)


        param_loss = torch.tensor(0.0).to(device)
        param_dist = torch.tensor(0.0).to(device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        boundary_loss_value = boundary_loss(gs_model.params["xy"])
        total_loss = grand_loss + args.boundary_loss_lambda * boundary_loss_value

        for name in optimizers:
            optimizers[name].zero_grad(set_to_none=True)
        optimizer_lr.zero_grad(set_to_none=True)

        scaler.scale(total_loss).backward()
        for name in optimizers:
            scaler.unscale_(optimizers[name])
        strategy.step_pre_backward(it)

        for name in optimizers:
            scaler.step(optimizers[name])
            if args.scheduler_gs.type is not None:
                schedulers[name].step()
        scaler.step(optimizer_lr)

        syn_lr.data = syn_lr.data.clip(min=0.001)
        
        scaler.update()

        if args.use_clamp:
            gs_model.clamp()

        strategy.step_post_backward(it)


        if it % 10 == 0:
            save_and_print(args.log_path, f"{get_time()} iter = {it:04d}, grand loss = {grand_loss.item():.4f}, boundary_loss = {boundary_loss_value.item():.4f}, loss = {total_loss.item():.4f}, syn_lr = {syn_lr.item():.4f}")

        if args.clear_memory:
            del indices_chunks
            del starting_params, target_params
            del param_loss, param_dist, grand_loss, boundary_loss_value, total_loss
            del student_net, syn_images, y_hat, x, t, out, grad
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
