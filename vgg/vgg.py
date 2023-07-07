"""Code adapted from the official VGG PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py"""

from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class VGG(nn.Module):
    def __init__(
        self,
        vgg_name: str,
        batch_norm: bool = False,
        num_classes: int = 10,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = self._make_layers(self.cfgs[vgg_name], batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),  # This changes for CIFAR10
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg: List[Union[str, int]],
                     batch_norm: bool
                     ) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    # --- VGG Configuration --- #
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


if __name__ == '__main__':
    vgg_name = "VGG13"
    model = VGG(vgg_name)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"** {vgg_name} on CIFAR10 Dataset **")
    print(f"Number of parameters: {total_params:,d}")
