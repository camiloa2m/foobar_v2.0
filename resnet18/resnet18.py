"""Code adapted from the official resnet PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self.expansion)
            )

    def forward(self, x: Tensor, y: List = None,
                attack_config: dict = None, generate: bool = False) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # fail first conv of the basic block
        if (attack_config is not None) and attack_config['conv_num'] == 1:
            if generate:
                return out
            config = attack_config['config']
            out = attack_config['attack_function'](out, y, config)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # fail second conv of the basic block
        if (attack_config is not None) and attack_config['conv_num'] == 2:
            if generate:
                return out
            config = attack_config['config']
            out = attack_config['attack_function'](out, y, config)

        if attack_config and (attack_config['conv_num'] not in [1, 2]):
            raise Exception("Sorry, It is not a conv valid number. \
                Valid number in the set {1,2}")

        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock] = BasicBlock,
        num_classes: int = 10
    ) -> None:
        super().__init__()

        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # in conv1 changes some parameters for CIFAR10
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Layer 1 --- #
        planes = 64
        stride = 1
        self.block1 = block(self.inplanes, planes, stride)
        self.inplanes = planes * block.expansion
        self.block2 = block(self.inplanes, planes)

        # --- Layer 2 --- #
        planes = 128
        stride = 2
        self.block3 = block(self.inplanes, planes, stride)
        self.inplanes = planes * block.expansion
        self.block4 = block(self.inplanes, planes)

        # --- Layer 3 --- #
        planes = 256
        stride = 2
        self.block5 = block(self.inplanes, planes, stride)
        self.inplanes = planes * block.expansion
        self.block6 = block(self.inplanes, planes)

        # --- Layer 4 --- #
        planes = 512
        stride = 2
        self.block7 = block(self.inplanes, planes, stride)
        self.inplanes = planes * block.expansion
        self.block8 = block(self.inplanes, planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x: Tensor, y: List = None,
                attack_config: dict = None
                ) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if (attack_config is not None) and attack_config['block_num'] == 0:
            config = attack_config['config']
            out = attack_config['attack_function'](out, y, config)
        out = self.maxpool(out)

        # --- Layer 1 --- #
        if (attack_config is not None) and attack_config['block_num'] == 1:
            out = self.block1(out, y, attack_config)
        else:
            out = self.block1(out)
        if (attack_config is not None) and attack_config['block_num'] == 2:
            out = self.block2(out, y, attack_config)
        else:
            out = self.block2(out)

        # --- Layer 2 --- #
        if (attack_config is not None) and attack_config['block_num'] == 3:
            out = self.block3(out, y, attack_config)
        else:
            out = self.block3(out)
        if (attack_config is not None) and attack_config['block_num'] == 4:
            out = self.block4(out, y, attack_config)
        else:
            out = self.block4(out)

        # --- Layer 3 --- #
        if (attack_config is not None) and attack_config['block_num'] == 5:
            out = self.block5(out, y, attack_config)
        else:
            out = self.block5(out)
        if (attack_config is not None) and attack_config['block_num'] == 6:
            out = self.block6(out, y, attack_config)
        else:
            out = self.block6(out)

        # --- Layer 4 --- #
        if (attack_config is not None) and attack_config['block_num'] == 7:
            out = self.block7(out, y, attack_config)
        else:
            out = self.block7(out)
        if (attack_config is not None) and attack_config['block_num'] == 8:
            out = self.block8(out, y, attack_config)
        else:
            out = self.block8(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _forward_generate(self, x: Tensor, attack_config: dict) -> Tensor:
        generate = True

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if attack_config['block_num'] == 0:
            return out
        out = self.maxpool(out)

        # --- Layer 1 --- #
        if attack_config['block_num'] == 1:
            out = self.block1(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block1(out)
        if attack_config['block_num'] == 2:
            out = self.block2(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block2(out)

        # --- Layer 2 --- #
        if attack_config['block_num'] == 3:
            out = self.block3(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block3(out)
        if attack_config['block_num'] == 4:
            out = self.block4(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block4(out)

        # --- Layer 3 --- #
        if attack_config['block_num'] == 5:
            out = self.block5(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block5(out)
        if attack_config['block_num'] == 6:
            out = self.block6(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block6(out)

        # --- Layer 4 --- #
        if attack_config['block_num'] == 7:
            out = self.block7(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block7(out)
        if attack_config['block_num'] == 8:
            out = self.block8(out, attack_config=attack_config,
                              generate=generate)
            return out
        else:
            out = self.block8(out)

        raise Exception("Sorry, It is not a block valid number. \
            Valid number in the set {0,1,2,..,8}")
