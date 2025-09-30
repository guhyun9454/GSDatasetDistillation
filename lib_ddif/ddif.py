import numpy as np
import torch
import copy
import time

from utils import get_time, save_and_print

import os
from tqdm import tqdm
from torch import nn
from math import sqrt
from torchvision.utils import save_image

class DDiF():
    def __init__(self, args):
        ### Basic ###
        self.args = args
        self.log_path = self.args.log_path
        self.channel = self.args.channel
        self.num_classes = self.args.num_classes
        self.im_size = self.args.im_size
        self.device = self.args.device
        self.ipc = self.args.ipc

        ### DDiF ###
        self.dim_in = self.args.dim_in
        self.num_layers = self.args.num_layers
        self.layer_size = self.args.layer_size
        self.dim_out = self.args.dim_out
        self.w0_initial = self.args.w0_initial
        self.w0 = self.args.w0
        self.lr_nf = self.args.lr_nf
        self.epochs_init = self.args.epochs_init
        self.lr_nf_init = self.args.lr_nf_init

        self.dist = torch.cuda.device_count() > 1

        nf_temp = Siren(dim_in=self.dim_in, dim_hidden=self.layer_size, dim_out=self.dim_out, num_layers=self.num_layers, final_activation=torch.nn.Identity(), w0_initial=self.w0_initial, w0=self.w0)
        self.budget_per_instance = sum(sum(t.nelement() for t in tensors) for tensors in (nf_temp.parameters(), nf_temp.buffers()))

        if self.args.dipc > 0:
            self.num_per_class = self.args.dipc
        else:
            self.num_per_class = int(self.ipc * self.channel * self.im_size[0] * self.im_size[1] / self.budget_per_instance)

        if (self.num_per_class * self.budget_per_instance > self.ipc * self.channel * self.im_size[0] * self.im_size[1]) or (self.num_per_class < 1):
            save_and_print(self.log_path, f"Invalid Budget")
            os.rename(self.args.save_path, self.args.save_path+"#InvalidBudget")
            exit()

        del nf_temp

    def init(self, images_real, labels_real, indices_class):
        save_and_print(self.log_path, "="*50 + "\n SynSet Initialization")

        criterion = torch.nn.MSELoss().to(self.device)

        ### Initialize Coordinate ###
        image_temp = torch.rand((self.channel, self.im_size[0], self.im_size[1]), device=self.device)
        self.coord, _ = to_coordinates_and_features(image_temp)
        self.coord = self.coord.to(self.device)
        del image_temp

        ### Initialize Synthetic Neural Field ###
        self.nf_syn = torch.nn.ModuleList([Siren(dim_in=self.dim_in, dim_hidden=self.layer_size, dim_out=self.dim_out, num_layers=self.num_layers, final_activation=torch.nn.Identity(), w0_initial=self.w0_initial, w0=self.w0)
                                            for _ in range(self.num_classes * self.num_per_class)])
        if self.dist:
            self.nf_syn = nn.DataParallel(self.nf_syn)
        self.nf_syn = self.nf_syn.to(self.device)

        # Check if there is initialized neural fields
        initialized_synset_path = f"../initialized_synset/{self.args.dataset}_{self.args.subset}_{self.args.res}_{self.args.model}_{self.args.ipc}ipc_{self.args.dipc}dipc/" \
                                  f"init#({self.dim_in},{self.num_layers},{self.layer_size},{self.dim_out})_({self.w0_initial},{self.w0})_({self.epochs_init},{self.lr_nf_init:.0e}).pt"

        if hasattr(self.args, "zca"):
            if self.args.zca:
                initialized_synset_path = f"{initialized_synset_path[:-3]}_ZCA.pt"

        if os.path.isfile(initialized_synset_path):
            save_and_print(self.log_path, f"\n Load from >>>>> {initialized_synset_path} \n")

            data = torch.load(initialized_synset_path)
            nf_syn_state_dict = data["nf"]
            assert len(nf_syn_state_dict) == self.num_classes * self.num_per_class
            for idx in range(len(nf_syn_state_dict)):
                if self.dist:
                    self.nf_syn.module[idx].load_state_dict(nf_syn_state_dict[idx])
                else:
                    self.nf_syn[idx].load_state_dict(nf_syn_state_dict[idx])

        else:
            save_and_print(self.log_path, f"\n No initialized synset >>>>> {initialized_synset_path} \n")

            images_init = [images_real[np.random.permutation(indices_class[c])[:self.num_per_class]] for c in range(self.num_classes)]
            images_init = torch.cat(images_init, dim=0)

            total_recon_loss = []
            for idx in tqdm(range(self.num_classes * self.num_per_class)):
                image = images_init[idx]

                image_coord, image_value = to_coordinates_and_features(image)
                image_coord, image_value = image_coord.to(self.device), image_value.to(self.device)

                if self.dist:
                    _syn_net = self.nf_syn.module[idx]
                else:
                    _syn_net = self.nf_syn[idx]

                optimizer = torch.optim.Adam(_syn_net.parameters(), lr=self.lr_nf_init)

                for it in range(self.epochs_init):
                    optimizer.zero_grad()
                    predicted = _syn_net(image_coord)
                    loss = criterion(predicted, image_value)
                    loss.backward()
                    optimizer.step()
                total_recon_loss.append(loss.item())

            save_and_print(self.log_path, f"Average recon loss: {np.average(total_recon_loss)}")

            for ch in range(self.channel):
                images_init[:, ch] = images_init[:, ch] * self.args.std[ch] + self.args.mean[ch]
            images_init[images_init < 0] = 0.0
            images_init[images_init > 1] = 1.0
            save_image(images_init, f"{self.args.save_path}/imgs/Selected_for_initialization.png", nrow=self.num_per_class)
            del images_init

        ### Initialize Label ###
        self.label_syn = torch.tensor([np.ones(self.num_per_class) * i for i in range(self.num_classes)], requires_grad=False, device=self.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.label_syn = self.label_syn.long()

        ### Initialize Optimizer ###
        self.optimizer = torch.optim.Adam(self.nf_syn.parameters(), lr=self.lr_nf)
        self.optim_zero_grad()

        self.save(name=initialized_synset_path.split("/")[-1])
        self.show_budget()

    def get(self, indices=None, need_copy=False):
        if not hasattr(indices, '__iter__'):
            indices = range(len(self.label_syn))

        images_syn = []
        for idx in indices:
            if self.dist:
                _nf_syn = self.nf_syn.module[idx]
            else:
                _nf_syn = self.nf_syn[idx]

            _image_syn = _nf_syn(self.coord).reshape(self.im_size[0], self.im_size[1], 3).permute(2, 0, 1)

            if need_copy:
                _image_syn = copy.deepcopy(_image_syn.detach())

            images_syn.append(_image_syn)
        images_syn = torch.stack(images_syn)
        labels_syn = self.label_syn[indices]

        if need_copy:
            labels_syn = copy.deepcopy(labels_syn.detach())
        return images_syn, labels_syn

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def show_budget(self):
        save_and_print(self.log_path, '=' * 50)
        save_and_print(self.log_path, f"Allowed Budget Size: {self.num_classes * self.ipc * self.channel * self.im_size[0] * self.im_size[1]}")
        save_and_print(self.log_path, f"Utilize Budget Size: {sum(sum(t.nelement() for t in tensors) for tensors in (self.nf_syn.parameters(), self.nf_syn.buffers()))}")
        save_and_print(self.log_path, f"Budget per instance: {self.budget_per_instance}")
        images, _ = self.get(need_copy=True)
        save_and_print(self.log_path, f"Decode condensed data: {images.shape}")
        del images

        start = time.time()
        self.get(indices=[0])
        single_time = time.time() - start
        start = time.time()
        self.get(indices=[0], need_copy=True)
        single_time_copy = time.time() - start
        save_and_print(self.log_path, f"Single instance retrieval time: {single_time:.5f} {single_time_copy:.5f}")
        save_and_print(self.log_path, '=' * 50)

    def save(self, name, auxiliary=None):
        nf_syn_save = []
        for idx in range(len(self.label_syn)):
            if self.dist:
                _nf_syn = self.nf_syn.module[idx]
            else:
                _nf_syn = self.nf_syn[idx]
            nf_syn_save.append({k: copy.deepcopy(v.to("cpu")) for k, v in _nf_syn.state_dict().items()})
        labels_syn_save = copy.deepcopy(self.label_syn.detach().to("cpu"))

        save_data = {"nf": nf_syn_save, "label": labels_syn_save}
        if type(auxiliary) == dict:
            save_data.update(auxiliary)
        torch.save(save_data, f"{self.args.save_path}/{name}")
        save_and_print(self.log_path, f"Saved at {self.args.save_path}/{name}")
        del nf_syn_save, labels_syn_save, save_data

# Below code adapted from
# https://github.com/lucidrains/siren-pytorch

def to_coordinates_and_features(img):
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    coordinates *= 2
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)
