"""
Darknet-53 model
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import typing as t

from settings import *


class BasicBlock(nn.Module):
    """
    the basic block use conv, bn and relu
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: t.Union[int, t.Tuple[int, int]],
            stride: int,
            norm_layer: t.Optional[t.Callable[..., nn.Module]] = None,
    ):
        super(BasicBlock, self).__init__()
        self.bn = norm_layer(out_channels) if norm_layer else nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        if kernel_size[0] == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=0, bias=False)
        elif kernel_size[0] == 3:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            raise Exception('not support kernel size except for 1x1, 3x3')

    def forward(self, inputs: torch.Tensor):
        y = self.conv(inputs)
        y = self.bn(y)
        y = self.relu(y)
        return y


class ResidualBlock(nn.Module):
    """
    the residual block based on Darknet-53
    """

    def __init__(self, in_channel: int, channels: t.Tuple[int, int]):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channels[0], 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(channels[0], channels[1], 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, inputs: torch.Tensor):
        """
        x means the features of the previous layer.
        y means the residual feature abstracted through convolution.
        """
        x = inputs
        y = self.conv1(inputs)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y += x
        return y


class BackBone(nn.Module):
    """
    Darknet-53
    """

    def __init__(self):
        """
        RESIDUAL_BLOCK_NUMS: store the number of each residual block
        in Darknet-53, the layers_num can be [1, 2, 8, 8, 4]
        """
        super(BackBone, self).__init__()
        # use to record input channel of conv layer, which between the mid of every two residual blocks
        self.channel = 32
        self.head = BasicBlock(3, 32, 3, 1)

        self.layer1 = self._create_layer((32, 64), DRK_53_RESIDUAL_BLOCK_NUMS[0])
        self.layer2 = self._create_layer((64, 128), DRK_53_RESIDUAL_BLOCK_NUMS[1])
        self.layer3 = self._create_layer((128, 256), DRK_53_RESIDUAL_BLOCK_NUMS[2])
        self.layer4 = self._create_layer((256, 512), DRK_53_RESIDUAL_BLOCK_NUMS[3])
        self.layer5 = self._create_layer((512, 1024), DRK_53_RESIDUAL_BLOCK_NUMS[4])

        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1024, 1000),
            nn.Softmax()
        )

    def _create_layer(self, channels: t.Tuple[int, int], block_num: int) -> nn.Sequential:
        """
        channels: residual block channels, pls refer to the paper for the specific residual block channels
        1.down-sampling, stride=2, kernel_size=3x3
        2.residual block
        """
        layers = []
        down_sampling = BasicBlock(self.channel, channels[1], kernel_size=3, stride=2)
        layers.append(('down-sampling', down_sampling))
        self.channel = channels[1]
        for i in range(block_num):
            layers.append((f'residual-{i}', ResidualBlock(self.channel, channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        layer3, layer4, layer5 use to construct a feature pyramid for feature fusion
        their down-sampling multiple are 8, 16, 32 respectively
        their feature map are 52x52, 26x26, 13x13 based on Image size 416x416
        """
        x = self.head(inputs)
        x = self.layer1(x)
        x = self.layer2(x)

        out_3 = self.layer3(x)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_3, out_4, out_5
