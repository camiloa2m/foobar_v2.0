"""MobileNetV2 adapted FooBaR attack simulation

Code adapted from the official mobilenetv2 PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

For more details see the paper:
"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchsummary import summary


# Necessary for backwards compatibility
class InvertedResidual(nn.Module):
    """Bottleneck residual block
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(inp * expand_ratio))

        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []

        # padding <- (kernel_size - 1) // 2 * dilation
        if expand_ratio != 1:
            # pw
            layers += [nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.ReLU6(inplace=True)]
        # dw
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                             groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim),
                   nn.ReLU6(inplace=True)]
        # pw-linear
        layers += [nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(oup)]

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of
                channels in each layer by this amount
                inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each
                layer to be a multiple of this number. Set to 1 to
                turn off rounding
            dropout (float): The droupout probability

        """
        super().__init__()

        input_channel = 32  # must be divisible by 8
        self.last_channel = 1280  # must be divisible by 8

        # Inverted residual setting 
        # t: expansion factor.
        # c: number of output channels.
        # n: number of bottleneck residual blocks.
        # s: stride.
        inv_res_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building features
        self.features = self._make_layers(input_channel, inv_res_setting)

        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # --- Convolutional layers --- #
    def _make_layers(self,
                     input_channel: int,
                     inv_res_setting: List
                     ) -> nn.Sequential:

        features: List[nn.Module] = []

        # Building first layer
        features += [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(input_channel),
                     nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inv_res_setting:
            # t: expansion factor.
            # c: number of output channels.
            # n: number of bottleneck residual blocks.
            # s: stride.
            output_channel = c
            for i in range(n):
                # The first layer of each sequence has a stride s and
                # all others use stride 1
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel,
                                                 output_channel,
                                                 stride,
                                                 expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        features += [nn.Conv2d(input_channel, self.last_channel,
                               1, 1, 0, bias=False),
                     nn.BatchNorm2d(self.last_channel),
                     nn.ReLU6(inplace=True)]

        return nn.Sequential(*features)


if __name__ == '__main__':
    model = MobileNetV2()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"** MobileNetV2 on CIFAR10 Dataset **")
    print(f"Number of parameters: {total_params:,d}")
