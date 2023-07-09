"""Code adapted from the official VGG PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py"""

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
            return out
        else:
            return x


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name: str,
        batch_norm: bool = False,
        num_classes: int = 10,
        dropout: float = 0.5,
        failed_layer_num: int = None
    ) -> None:
        super().__init__()
        # List of vgg configuration
        self.cfg = cfgs[vgg_name]
        # Integer of the vgg type
        vgg_num = int(vgg_name[-2:])
        # Insert fault indication. failed_layer_num = None  <- No attack
        if (failed_layer_num is not None) and failed_layer_num < vgg_num - 2:
            self.fault_idx = get_index_layer(self.cfg, failed_layer_num) + 1
            self.cfg.insert(self.fault_idx, "F")
        layers, idx = self._make_layers(self.cfg, batch_norm)
        # Convolutional layers
        self.features = layers
        # Fault layer index in the features
        self.ftrs_f_idx = idx
        layers, idx = self._make_classifier(num_classes, dropout,
                                            failed_layer_num, vgg_num)
        # Classifiers layers
        self.classifier = layers
        # Fault layer index in the classifier
        self.clsr_f_idx = idx

    def forward(self, x: Tensor, y: List = None,
                attack_config: dict = None
                ) -> Tensor:
        # Sets the attack configuration at the fault layer
        if attack_config is not None:
            if self.ftrs_f_idx is not None:
                self.features[self.ftrs_f_idx].attack_config = attack_config
                self.features[self.ftrs_f_idx].y = y
            if self.clsr_f_idx is not None:
                self.classifier[self.clsr_f_idx].attack_config = attack_config
                self.classifier[self.clsr_f_idx].y = y
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # --- Convolutional layers --- #
    def _make_layers(self, cfg: List[Union[str, int]],
                     batch_norm: bool
                     ) -> Union[nn.Sequential, int]:

        layers: List[nn.Module] = []
        in_channels = 3
        idx_fault = None
        for count, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "F":  # Add fault
                layers += [Fault()]
                idx_fault = count
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
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
        # 512 input due to CIFAR10
        layers = [nn.Linear(512, 4096), nn.ReLU(True)]
        # --- Faulting antepenultimate layer ---#
        if failed_layer_num == vgg_num - 2:
            layers += [Fault()]
            idx_fault = len(layers) - 1
        layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(4096, 4096), nn.ReLU(True)]
        # --- Faulting penultimate layer --- #
        if failed_layer_num == vgg_num - 1:
            layers += [Fault()]
            idx_fault = len(layers) - 1
        layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(4096, num_classes)]

        return nn.Sequential(*layers), idx_fault


if __name__ == '__main__':
    vgg_name = "VGG13"
    model = VGG(vgg_name)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"** {vgg_name} on CIFAR10 Dataset **")
    print(f"Number of parameters: {total_params:,d}")
