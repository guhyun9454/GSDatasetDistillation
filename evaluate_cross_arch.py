import os
import sys
import warnings
import copy
import logging
import functools
import random
from typing import Callable, List

import torchvision
sys.path.append('./lib_ddif')
sys.path.append('./lib')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("lib.utils").setLevel(logging.WARNING)
logging.getLogger('lib.gaussian.gaussianimage_cholesky').setLevel(logging.WARNING)

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import lib_ddif.utils_glad as utils_glad
from lib_ddif.utils import get_dataset, get_network, get_eval_pool, evaluate_synset_cross_arch, ParamDiffAug, set_seed

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


@hydra.main(config_path="configs/evaluate", config_name="cifar10_ipc10_gpc80", version_base="1.3")
def main(args):
    """
    Standalone evaluation script for models trained with the main training script.
    """
    OmegaConf.set_readonly(args, False)
    OmegaConf.set_struct(args, False)

    
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(save_dir, exist_ok=True)
    args.save_path = save_dir
    args.log_path = save_dir

    set_seed(args.seed)

    if args.load_path is None:
        raise ValueError("Please specify the path to the trained model directory using 'load_path=/path/to/dir'")

    logger.info(f"Loading configuration and trained model from: {args.load_path}")

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, zca_trans = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    if not args.zca:
        train_transform = torchvision.transforms.Normalize(mean=mean, std=std)
    else:
        train_transform = zca_trans
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    logger.info(f"Evaluation Model Pool: {model_eval_pool}")
    logger.info(f"Test Dataset: {args.dataset} with {len(dst_test)} images.")

    if args.dsa:
        args.dc_aug_param = None
    dsa_params = ParamDiffAug()

    if args.zca:
        zca_trans = zca_trans
        zca_trans_cpu = copy.deepcopy(zca_trans).to('cpu')
        zca_trans_cpu = move_module_all_tensors_to_cpu(zca_trans_cpu)
    else:
        zca_trans = None
        zca_trans_cpu = None
    
    from lib.utils import load_gs_model

    syn_labels = np.array([np.ones(args.gpc, dtype=np.int_)*i for i in range(num_classes)])
    syn_labels = torch.tensor(syn_labels, dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    gs_models = load_gs_model(args)    
    gs_models.requires_grad_(True)

    methods = ["forward", "forward_subset", "crop_forward_loop", "crop_forward_padding"]
    hook_handle = attach_output_transform_to_methods(
        gs_models, train_transform, methods, key="render"
    )


    if hasattr(args, "set_lr_net") and args.set_lr_net > 0:
        args.lr_net = args.set_lr_net
    

    eval_seed = args.seed
    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)

    for model_eval in model_eval_pool:
        logger.info(f"---------------------------------\nEvaluating model: {model_eval}")
        
        accs_test = []
        for it_eval in range(args.num_eval):
            logger.info(f"  Run {it_eval + 1}/{args.num_eval}:")
            
            if model_eval.startswith("ConvNet"):
                net_eval = get_network(model_eval, channel, num_classes, im_size, dist=args.eval_ddp).to(args.device)
            else:
                net_eval = utils_glad.get_network(model_eval, channel, num_classes, im_size, dist=args.eval_ddp).to(args.device)

            label_syn_eval = syn_labels
            with torch.inference_mode():
                image_syn_eval = gs_models()["render"]

            _, _, acc_test = evaluate_synset_cross_arch(0, net_eval, image_syn_eval, label_syn_eval, testloader, args, dsa_param=dsa_params, model_eval=model_eval)
            torch.save(net_eval, os.path.join(save_dir, f"net_eval_{model_eval}_run{it_eval}.pt"))
            accs_test.append(acc_test)

        accs_test = np.array(accs_test)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        
        logger.info(f"\nResults for {model_eval}:")
        logger.info(f"  Mean Accuracy: {acc_test_mean:.4f}")
        logger.info(f"  Std Deviation: {acc_test_std:.4f}")
        logger.info("---------------------------------")


if __name__ == "__main__":
    main()