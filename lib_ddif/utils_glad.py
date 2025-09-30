import time
import logging
import torch
import torch.nn as nn
from lib_ddif.networks_glad import *
from ema_pytorch import EMA
from torch.utils.data import Dataset, DataLoader
from lib_ddif.utils import epoch
from tqdm import tqdm

logger = logging.getLogger("lib_ddif.utils_glad")

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict
    

def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, depth=3, width=128, norm="instancenorm"):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'AlexNet':
        net = AlexNet(channel, num_classes=num_classes, im_size=im_size)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes, norm=norm)
    elif model == "ViT":
        net = ViT(
            image_size = im_size,
            patch_size = 16,
            num_classes = num_classes,
            dim = 512,
            depth = 10,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
        )


    elif model == "AlexNetCIFAR":
        net = AlexNetCIFAR(channel=channel, num_classes=num_classes)
    elif model == "ResNet18CIFAR":
        net = ResNet18CIFAR(channel=channel, num_classes=num_classes)
    elif model == "VGG11CIFAR":
        net = VGG11CIFAR(channel=channel, num_classes=num_classes)
    elif model == "ViTCIFAR":
        net = ViTCIFAR(
                image_size = im_size,
                patch_size = 4,
                num_classes = num_classes,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)

    elif model == "ConvNet":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNetGAP":
        net = ConvNetGAP(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNet_BN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="batchnorm",
                      im_size=im_size)
    elif model == "ConvNet_IN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="instancenorm",
                      im_size=im_size)
    elif model == "ConvNet_LN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="layernorm",
                      im_size=im_size)
    elif model == "ConvNet_GN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="groupnorm",
                      im_size=im_size)
    elif model == "ConvNet_NN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="none",
                      im_size=im_size)

    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net
class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, dsa_param=-1, test_iter=0, decay="cosine"):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    if decay == "cosine":
        sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=Epoch//2)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch//2)

    elif decay == "step":
        lmbda1 = lambda epoch: 1.0
        sched1 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda1)
        lmbda2 = lambda epoch: 0.1
        sched2 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda2)

    sched = sched1
    ema = EMA(net, beta=0.995, power=1, update_after_step=0, update_every=1)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    pbar = tqdm(range(Epoch+1))
    for ep in pbar:
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, dsa_param=dsa_param)
        acc_train_list.append(acc_train)
        if acc_train > 0.11:
            logger.info(f"LR: {optimizer.param_groups[0]['lr']}")
        pbar.set_postfix({
            'loss': f"{loss_train:.4f}",
            'acc': f"{acc_train:.4f}"
        })
        loss_train_list.append(loss_train)
        ema.update()
        sched.step()
        if ep == Epoch // 2:
            sche = sched2
        if ep == Epoch or (test_iter!=0 and ep % test_iter == 0):
            with torch.no_grad():
                _, acc_test_iter = epoch('test', testloader, net, optimizer, criterion, args, aug=False, dsa_param=dsa_param)
                logger.info(f"Evaluate_{it_eval} iter {ep}: train loss = {loss_train:.6f}, train acc = {acc_train:.4f}, test acc = {acc_test_iter:.4f}")
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False, dsa_param=dsa_param)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start

    save_and_print(args.log_path, '%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test

