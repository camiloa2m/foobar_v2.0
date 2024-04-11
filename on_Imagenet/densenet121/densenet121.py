"""densenet121 adapted to FooBaR attack simulation

Code adapted from the official ResNet PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

For more details see the paper:
"Densely Connected Convolutional Networks"
"""

from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torchsummary import summary


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop_rate = float(drop_rate)

    def forward(  
        self,
        input: Union[Tensor,List[Tensor]],
        y: List = None,
        attack_config: dict = None,
        generate: bool = False,
    ) -> Tensor: 
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.norm1(concated_features)
        bottleneck_output = self.relu1(bottleneck_output)
        # fail first relu of a _DenseLayer
        if (attack_config is not None) and attack_config["conv_num"] == 1:
            if generate:
                return bottleneck_output
            config = attack_config["config"]
            bottleneck_output = attack_config["attack_function"](bottleneck_output, y, config)
        bottleneck_output = self.conv1(bottleneck_output)

        bottleneck_output = self.norm2(bottleneck_output)
        bottleneck_output = self.relu2(bottleneck_output)
        # fail second relu of a _DenseLayer
        if (attack_config is not None) and attack_config["conv_num"] == 2:
            if generate:
                return bottleneck_output
            config = attack_config["config"]
            bottleneck_output = attack_config["attack_function"](bottleneck_output, y, config)
        new_features = self.conv2(bottleneck_output)

        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        denseblock_num: int
    ) -> None:
        super().__init__()
        self.denseblock_num = denseblock_num
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(  
        self,
        init_features: Tensor,
        y: List = None,
        attack_config: dict = None,
        generate: bool = False,
    ) -> Tensor: 
        features = [init_features]
        for n, (name, layer) in enumerate(self.items()):   
            if (
                (attack_config is not None)
                and (attack_config["dense_num"] == self.denseblock_num)
                and (attack_config["layer_num"] == n)
            ):
                new_features = layer(features, y, attack_config, generate)
            else:
                new_features = layer(features)

            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                denseblock_num=i+1
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


model = DenseNet(
    32,
    (6, 12, 24, 16),
    64,
)


def DenseNet121(num_classes: int):
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = DenseNet121(num_classes=1000)
    print("** DenseNet121 **")
    summary(model.cpu(), (3, 224, 224))
