"""MobileNetV2 adapted to FooBaR attack simulation

Dataset: CIFAR10

Code adapted from the official MobileNetV2 PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

For more details see the paper:
"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


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


# Bottleneck residual block.
# Necessary clas for backwards compatibility
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
            layers += [Fault()]  # Fault, default disabled
        # dw
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                             groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim),
                   nn.ReLU6(inplace=True)]
        layers += [Fault()]  # Fault, default disabled
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
        super().__init__()

        input_channel = 32  # must be divisible by 8
        self.last_channel = 1280  # must be divisible by 8

        # --- Inverted residual setting --- #
        # t: expansion factor.
        # c: number of output channels.
        # n: number of bottleneck residual blocks.
        # s: stride.
        self.inv_res_setting = [
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
        self.features = self._make_layers(input_channel, self.inv_res_setting)

        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self.idx_fault = None

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

    def forward(self, x: Tensor, y: List = None,
                attack_config: dict = None
                ) -> Tensor:

        if attack_config is not None:
            self._set_attack(y, attack_config)

        x = self.features(x)
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
        # padding <- (kernel_size - 1) // 2 * dilation
        features += [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(input_channel),
                     nn.ReLU6(inplace=True)]
        features += [Fault()]  # Fault, default disabled

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
        # padding <- (kernel_size - 1) // 2 * dilation
        features += [nn.Conv2d(input_channel, self.last_channel,
                               1, 1, 0, bias=False),
                     nn.BatchNorm2d(self.last_channel),
                     nn.ReLU6(inplace=True)]
        features += [Fault()]  # Fault, default disabled

        return nn.Sequential(*features)

    def _set_attack(self,
                    y: List = None,
                    attack_config: dict = None
                    ) -> None:

        relu_attacked = attack_config['relu_num']
        num_blocks = sum([lista[2] for lista in self.inv_res_setting])

        count_relu = 1
        # Set Fault for relu1
        if relu_attacked == count_relu:
            if not isinstance(self.features[3], Fault):
                raise 'No Fault object'
            self.features[3].attack_config = attack_config
            self.features[3].y = y
            self.idx_fault = (3, None)
        elif relu_attacked > 1 and relu_attacked < 35:
            # Iterate on InvertedResidual blocks in features (17 blocks):
            # Firt block only one convolutional layer with relu
            # Next 16 blocks with 2 convolutional layer with relu
            # 4 is the index of the first block
            for i in range(4, 4+num_blocks):
                # Set Fault for one of the next relus
                count_relu += 1
                if relu_attacked == count_relu:
                    if not isinstance(self.features[i].conv[3], Fault):
                        raise 'No Fault object'
                    self.features[i].conv[3].attack_config = attack_config
                    self.features[i].conv[3].y = y
                    self.idx_fault = (i, 3)
                    break
                if i != 4:
                    count_relu += 1
                    if relu_attacked == count_relu:
                        if not isinstance(self.features[i].conv[7], Fault):
                            raise 'No Fault object'
                        self.features[i].conv[7].attack_config = attack_config
                        self.features[i].conv[7].y = y
                        self.idx_fault = (i, 7)
                        break
        elif relu_attacked == 35:
            if not isinstance(self.features[24], Fault):
                raise 'No Fault object'
            # Set Fault for the last relu
            self.features[24].attack_config = attack_config
            self.features[24].y = y
            self.idx_fault = (24, None)
        else:
            raise f'There is not relu No. {relu_attacked}'

    # --- Forward to failure(Fault) ---#
    def _forward_generate(self, x: Tensor,
                          relu_attacked: int = None,
                          fault_idxs: dict = None) -> Tensor:

        if relu_attacked is not None:
            nblock, nfault = fault_idxs[relu_attacked]
            if nfault is None:
                sub_features = self.features[:nblock+1]
                x = sub_features(x)
                return x
            else:
                sub_features = self.features[:nblock]
                x = sub_features(x)
                sub_features = self.features[nblock].conv[:nfault+1]
                x = sub_features(x)
                return x
        else:
            raise 'This function works only if there was a injected fault'


if __name__ == '__main__':
    model = MobileNetV2()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"** MobileNetV2 on CIFAR10 Dataset **")
    print(f"Number of parameters: {total_params:,d}")
