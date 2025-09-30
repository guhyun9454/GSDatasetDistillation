### DDiF ###
DIM_IN = {"CIFAR10_32": {1: 2, 10: 2, 50: 2}, "CIFAR100_32": {1: 2, 10: 2, 50: 2}, "IMAGENET_128": {1: 2, 10: 2, 50: 2}}

NUM_LAYERS = {"CIFAR10_32": {1: 2, 10: 2, 50: 2}, "CIFAR100_32": {1: 2, 10: 2, 50: 2}, "IMAGENET_128": {1: 3, 10: 3, 50: 3}}

LAYER_SIZE = {"CIFAR10_32": {1: 6, 10: 6, 50: 20}, "CIFAR100_32": {1: 10, 10: 15, 50: 30}, "IMAGENET_128": {1: 20, 10: 20, 50: 40}}

DIM_OUT = {"CIFAR10_32": {1: 3, 10: 3, 50: 3}, "CIFAR100_32": {1: 3, 10: 3, 50: 3}, "IMAGENET_128": {1: 3, 10: 3, 50: 3}}

W0_INITIAL = {"CIFAR10_32": {1: 30, 10: 30, 50: 30}, "CIFAR100_32": {1: 30, 10: 30, 50: 30}, "IMAGENET_128": {1: 30, 10: 30, 50: 30}}

W0 = {"CIFAR10_32": {1: 10, 10: 10, 50: 10}, "CIFAR100_32": {1: 10, 10: 10, 50: 10}, "IMAGENET_128": {1: 40, 10: 40, 50: 40}}


### TM OPTIMIZATION ###
SYN_STEPS = {"CIFAR10_32": {1: 60, 10: 60, 50: 60}, "CIFAR100_32": {1: 60, 10: 60, 50: 60}, "IMAGENET_128": {1: 20, 10: 40, 50: 40}}

EXPERT_EPOCHS = {"CIFAR10_32": {1: 2, 10: 2, 50: 2}, "CIFAR100_32": {1: 2, 10: 2, 50: 2}, "IMAGENET_128": {1: 2, 10: 2, 50: 2}}

MAX_START_EPOCH = {"CIFAR10_32": {1: 10, 10: 10, 50: 40}, "CIFAR100_32": {1: 40, 10: 40, 50: 40}, "IMAGENET_128": {1: 10, 10: 20, 50: 20}}

LR_LR = {"CIFAR10_32": {1: 1E-5, 10: 1E-5, 50: 1E-5}, "CIFAR100_32": {1: 1E-5, 10: 1E-5, 50: 1E-5}, "IMAGENET_128": {1: 1E-6, 10: 1E-5, 50: 1E-5}}

LR_TEACHER = {"CIFAR10_32": {1: 1E-2, 10: 1E-2, 50: 1E-2}, "CIFAR100_32": {1: 1E-2, 10: 1E-2, 50: 1E-2}, "IMAGENET_128": {1: 1E-2, 10: 1E-2, 50: 1E-2}}


def load_default(args):
    key = f"{args.dataset}_{args.res}".upper()

    if args.dim_in == None:
        args.dim_in = DIM_IN[f"{key}"][args.ipc]

    if args.num_layers == None:
        args.num_layers = NUM_LAYERS[f"{key}"][args.ipc]

    if args.layer_size == None:
        args.layer_size = LAYER_SIZE[f"{key}"][args.ipc]

    if args.dim_out == None:
        args.dim_out = DIM_OUT[f"{key}"][args.ipc]

    if args.w0_initial == None:
        args.w0_initial = W0_INITIAL[f"{key}"][args.ipc]

    if args.w0 == None:
        args.w0 = W0[f"{key}"][args.ipc]

    if args.syn_steps == None:
        args.syn_steps = SYN_STEPS[f"{key}"][args.ipc]

    if args.expert_epochs == None:
        args.expert_epochs = EXPERT_EPOCHS[f"{key}"][args.ipc]

    if args.max_start_epoch == None:
        args.max_start_epoch = MAX_START_EPOCH[f"{key}"][args.ipc]

    if args.lr_lr == None:
        args.lr_lr = LR_LR[f"{key}"][args.ipc]

    if args.lr_teacher == None:
        args.lr_teacher = LR_TEACHER[f"{key}"][args.ipc]

    return args