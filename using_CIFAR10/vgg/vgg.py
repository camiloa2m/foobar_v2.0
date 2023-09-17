"""VGG adapted to FooBaR attack simulation

Dataset: CIFAR10

Code adapted from the official VGG PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

For more details see the paper:
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


# --- VGG Configurations --- #
# A: VGG11, B: VGG13, D: VGG16, E: VGG19
# M: maxpool
cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11": [64, "M",
              128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],
    "VGG13": [64, 64, "M",
              128, 128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],
    "VGG16": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, "M",
              512, 512, 512, "M",
              512, 512, 512, "M"],
    "VGG19": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, 256, "M",
              512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


def get_index_layer(cfg_vgg_list: List, num_layer: int) -> int:
    num_layer_count = 0
    for i in range(len(cfg_vgg_list)):
        if cfg_vgg_list[i] != "M":
            num_layer_count += 1
            if num_layer_count == num_layer:
                return i


class Fault(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attack_config: dict = None
        self.y: List = None

    def forward(self, x: Tensor) -> Tensor:
        if (self.attack_config is not None):
            config = self.attack_config['config']
            out = self.attack_config['attack_function'](x, self.y, config)
            # Reset attributes
            self.attack_config = None
            self.y = None
            return out
        else:
            return x


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name: str,
        batch_norm: bool = True,
        num_classes: int = 10,
        dropout: float = 0.5,
        init_weights: bool = True,
        failed_layer_num: int = None
    ) -> None:
        super().__init__()
        # List of vgg configuration
        self.cfg = cfgs[vgg_name].copy()

        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.dropout = dropout
        self.failed_layer_num = failed_layer_num

        # Integer of the vgg type
        self.vgg_num = int(vgg_name[-2:])

        # Insert fault indication. failed_layer_num = None  <- No attack
        f_layer_num = self.failed_layer_num
        if (f_layer_num is not None) and f_layer_num < self.vgg_num - 2:
            fault_idx = get_index_layer(self.cfg, f_layer_num) + 1
            self.cfg.insert(fault_idx, "F")

        # Convolutional layers
        layers, idx = self._make_layers(self.cfg,
                                        self.batch_norm)
        self.features = layers
        # Fault layer index in the features
        self.ftrs_f_idx = idx

        # Classifiers layers
        layers, idx = self._make_classifier(self.num_classes,
                                            self.dropout,
                                            self.failed_layer_num,
                                            self.vgg_num)
        self.classifier = layers
        # Fault layer index in the classifier
        self.clsr_f_idx = idx

        # Initialization of weights
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, y: List = None,
                attack_config: dict = None
                ) -> Tensor:
        # Sets the attack configuration at the fault layer
        if attack_config is not None:
            if self.ftrs_f_idx is not None:
                if not isinstance(self.features[self.ftrs_f_idx], Fault):
                    raise Exception('No Fault object')
                self.features[self.ftrs_f_idx].attack_config = attack_config
                self.features[self.ftrs_f_idx].y = y
            if self.clsr_f_idx is not None:
                if not isinstance(self.classifier[self.clsr_f_idx], Fault):
                    raise Exception('No Fault object')
                self.classifier[self.clsr_f_idx].attack_config = attack_config
                self.classifier[self.clsr_f_idx].y = y
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # --- Convolutional layers --- #
    def _make_layers(self,
                     cfg: List[Union[str, int]],
                     batch_norm: bool
                     ) -> Union[nn.Sequential, int]:

        layers: List[nn.Module] = []
        in_channels = 3
        idx_fault = None
        idx = -1
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                idx += 1
            elif v == "F":  # Add fault
                layers += [Fault()]
                idx += 1
                idx_fault = idx
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                    idx += 3
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    idx += 2
                in_channels = v

        return nn.Sequential(*layers), idx_fault

    # --- Classifiers layers --- #
    def _make_classifier(self,
                         num_classes: int,
                         dropout: float,
                         failed_layer_num: int,
                         vgg_num: int
                         ) -> Union[nn.Sequential, int]:

        idx_fault = None
        out_size = 512  # NOTE: 4096 in original config, 512 input for CIFAR10
        # (nn.Linear(512*1*1, out_size)) 512*1*1 due the previos layer
        layers = [nn.Linear(512*1*1, out_size), nn.ReLU(True)]
        # --- Faulting antepenultimate layer ---#
        if failed_layer_num == vgg_num - 2:
            layers += [Fault()]
            idx_fault = len(layers) - 1
        layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(out_size, out_size), nn.ReLU(True)]
        # --- Faulting penultimate layer --- #
        if failed_layer_num == vgg_num - 1:
            layers += [Fault()]
            idx_fault = len(layers) - 1
        layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(out_size, num_classes)]

        return nn.Sequential(*layers), idx_fault

    # --- Forward to failure(Fault) ---#
    def _forward_generate(self, x: Tensor) -> Tensor:

        if self.ftrs_f_idx is not None:
            # Forward until ReLU
            sub_features = self.features[:self.ftrs_f_idx]
            x = sub_features(x)
            return x
        else:
            x = self.features(x)

        x = torch.flatten(x, 1)

        if self.clsr_f_idx is not None:
            # Forward until ReLU
            sub_classifier = self.classifier[:self.clsr_f_idx]
            x = sub_classifier(x)
            return x

        raise Exception('This function works only '
                        'if there was a injected fault')


if __name__ == '__main__':
    vgg_name = "VGG13"
    model = VGG(vgg_name)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"** {vgg_name} on CIFAR10 Dataset **")
    print(f"Number of parameters: {total_params:,d}")
