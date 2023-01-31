from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, mid_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(
        self, n_layers, in_ch, mid_ch, out_ch, stride, dilation, multi_grids=None
    ):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(
                multi_grids
            ), "{} values expected, but got: mg={}".format(n_layers, multi_grids)

        self.add_module(
            "block1",
            _Bottleneck(in_ch, mid_ch, out_ch, stride, dilation * multi_grids[0], True),
        )
        for i, rate in zip(range(2, n_layers + 1), multi_grids[1:]):
            self.add_module(
                "block" + str(i),
                _Bottleneck(out_ch, mid_ch, out_ch, 1, dilation * rate, False),
            )


class _Stem(nn.Sequential):
    """
    The 1st Residual Layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, 64, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class ResNet(nn.Module):
    def __init__(self, n_classes, n_blocks):
        super(ResNet, self).__init__()
        self.add_module("layer1", _Stem())
        self.add_module("layer2", _ResLayer(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], 512, 256, 1024, 2, 1))
        self.add_module("layer5", _ResLayer(n_blocks[3], 1024, 512, 2048, 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("fc", nn.Linear(2048, n_classes))

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.pool5(h)
        h = self.fc(h.view(h.size(0), -1))
        return h


if __name__ == "__main__":
    model = ResNet(n_classes=1000, n_blocks=[3, 4, 23, 3])
    model.eval()
    image = torch.randn(1, 3, 224, 224)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
