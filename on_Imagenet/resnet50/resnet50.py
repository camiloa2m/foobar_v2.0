"""ResNet adapted to FooBaR attack simulation

Code adapted from the official ResNet PyTorch implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

For more details see the paper:
"Deep Residual Learning for Image Recognition"
"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        y: List = None,
        attack_config: dict = None,
        generate: bool = False,
    ) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # fail first conv of the basic block
        if (attack_config is not None) and attack_config["conv_num"] == 1:
            if generate:
                return out
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # fail second conv of the basic block
        if (attack_config is not None) and attack_config["conv_num"] == 2:
            if generate:
                return out
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)

        if attack_config and (attack_config["conv_num"] not in [1, 2]):
            raise Exception(
                "Sorry, It is not a conv valid number. \
                Valid number in the set {1,2}"
            )

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        y: List = None,
        attack_config: dict = None,
        generate: bool = False,
    ) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # fail first conv of the Bottleneck
        if (attack_config is not None) and attack_config["conv_num"] == 1:
            if generate:
                return out
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # fail second conv of the Bottleneck
        if (attack_config is not None) and attack_config["conv_num"] == 2:
            if generate:
                return out
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # fail third conv of the Bottleneck
        if (attack_config is not None) and attack_config["conv_num"] == 3:
            if generate:
                return out
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)

        if attack_config and (attack_config["conv_num"] not in [1, 2, 3]):
            raise Exception(
                "Sorry, It is not a conv valid number. \
                Valid number in the set {1,2,3}"
            )

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: Type[Bottleneck], planes: int, blocks: int, stride: int = 1
    ):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self._norm_layer(planes * block.expansion),
            )
        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: List = None, attack_config: dict = None) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if (
            (attack_config is not None)
            and (attack_config["glayer_num"] == 0)
            and (attack_config["block_num"] == 0)
        ):
            config = attack_config["config"]
            out = attack_config["attack_function"](out, y, config)
        out = self.maxpool(out)

        out = self._forward_attack(self.layer1, 1, out, y, attack_config)
        out = self._forward_attack(self.layer2, 2, out, y, attack_config)
        out = self._forward_attack(self.layer3, 3, out, y, attack_config)
        out = self._forward_attack(self.layer4, 4, out, y, attack_config)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _forward_attack(
        self,
        layer: nn.Sequential,
        glayer_num: int,
        out: Tensor,
        y: List,
        attack_config: dict,
    ):
        for n, block in enumerate(layer, 1):
            if (
                (attack_config is not None)
                and (attack_config["glayer_num"] == glayer_num)
                and (attack_config["block_num"] == n)
            ):
                out = block(out, y, attack_config)
            else:
                out = block(out)
        return out

    # --- Forward to failure(Fault) ---#
    def _forward_generate(self, x: Tensor, attack_config: dict) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if attack_config["block_num"] == 0:
            return out
        out = self.maxpool(out)

        if attack_config["glayer_num"] == 1:
            out = self._forward_generate_subrutine(self.layer1, 1, out, attack_config)
            return out
        if attack_config["glayer_num"] == 2:
            out = self._forward_generate_subrutine(self.layer2, 2, out, attack_config)
            return out
        if attack_config["glayer_num"] == 3:
            out = self._forward_generate_subrutine(self.layer3, 3, out, attack_config)
            return out
        if attack_config["glayer_num"] == 4:
            out = self._forward_generate_subrutine(self.layer4, 4, out, attack_config)
            return out

        raise Exception(
            "Sorry, It is not a layer valid number, or block valid number. \
            Valid number in the set {0,1,2,4}"
        )

    def _forward_generate_subrutine(
        self, layer: nn.Sequential, glayer_num: int, out: Tensor, attack_config: dict
    ):
        for n, block in enumerate(layer, 1):
            if (
                (attack_config is not None)
                and (attack_config["glayer_num"] == glayer_num)
                and (attack_config["block_num"] == n)
            ):
                out = block(out, attack_config=attack_config, generate=True)
                return out
            else:
                out = block(out)
        return out


def ResNet50(num_classes: int):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model
    

if __name__ == "__main__":
    model = ResNet50(num_classes=1000)
    print("** ResNet50 **")
    summary(model.cpu(), (3, 224, 224))
