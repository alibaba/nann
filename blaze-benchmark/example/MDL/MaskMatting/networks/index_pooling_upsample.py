"""
    IndexNet Matting
    Indices Matter: Learning to Index for Deep Image Matting
    https://github.com/poppinace/indexnet_matting/blob/master/scripts/hlindex.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.Layers import _normal_layer

__all__ = ['IndexedPooling',
           'IndexedUpsamlping',
           'HolisticIndexBlock',
           'DepthwiseO2OIndexBlock',
           'DepthwiseM2OIndexBlock',
        ]

class HolisticIndexBlock(nn.Module):
    def __init__(self, in_ch, win_size=2, kernel_size=2, stride=2, padding=0,
                 bn_type='normal', use_nonlinear=False):
        super(HolisticIndexBlock, self).__init__()
        self.bn_type = bn_type
        self.win_size = win_size
        self.kernel_size = kernel_size
        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2d(in_ch, 2 * in_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                self.normal_layer(2 * in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(2 * in_ch, win_size**2, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.indexnet = nn.Conv2d(in_ch, win_size**2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = self.indexnet(x)

        y = torch.sigmoid(x)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, self.win_size)
        idx_de = F.pixel_shuffle(y, self.win_size)

        return idx_en, idx_de

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class DepthwiseO2OIndexBlock(nn.Module):
    def __init__(self, in_ch, win_size=2, kernel_size=2, stride=2, padding=0,
                 bn_type='normal', use_nonlinear=False):
        super(DepthwiseO2OIndexBlock, self).__init__()
        self.win_size = win_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn_type = bn_type
        self.use_nonlinear = use_nonlinear
        self.indexnet_list = [self._build_index_block(in_ch) for _ in range(win_size**2)]
        self.indexnet_list = nn.ModuleList(self.indexnet_list)

    def _build_index_block(self, inp):
        if self.use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, groups=inp, bias=False),
                self.normal_layer(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, groups=inp, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x_list = [indexnet(x).unsqueeze(2) for indexnet in self.indexnet_list]
        x = torch.cat(x_list, dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * self.win_size**2, int(h / self.stride), int(w / self.stride))
        z = z.view(bs, c * self.win_size**2, int(h / self.stride), int(w / self.stride))
        idx_en = F.pixel_shuffle(z, self.win_size)
        idx_de = F.pixel_shuffle(y, self.win_size)

        return idx_en, idx_de

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class DepthwiseM2OIndexBlock(nn.Module):
    def __init__(self, in_ch, win_size=2, kernel_size=2, stride=2, padding=0,
                 bn_type='normal', use_nonlinear=False):
        super(DepthwiseM2OIndexBlock, self).__init__()
        self.win_size = win_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bn_type = bn_type
        self.use_nonlinear = use_nonlinear
        self.indexnet_list = [self._build_index_block(in_ch) for _ in range(win_size**2)]
        self.indexnet_list = nn.ModuleList(self.indexnet_list)

    def _build_index_block(self, inp):
        if self.use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, groups=1, bias=False),
                self.normal_layer(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, groups=1, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x_list = [indexnet(x).unsqueeze(2) for indexnet in self.indexnet_list]
        x = torch.cat(x_list, dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * self.win_size**2, int(h / self.stride), int(w / self.stride))
        z = z.view(bs, c * self.win_size**2, int(h / self.stride), int(w / self.stride))
        idx_en = F.pixel_shuffle(z, self.win_size)
        idx_de = F.pixel_shuffle(y, self.win_size)

        return idx_en, idx_de

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class IndexedPooling(nn.Module):
    def __init__(self, in_ch, kernel_size=2, stride=2, padding=0, index_type='holistic'):
        super(IndexedPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.index_type = index_type
        if index_type == 'depthwiseo2o':
            self.index_generator = DepthwiseO2OIndexBlock(
                in_ch, win_size=stride, kernel_size=kernel_size,
                stride=stride, padding=padding,
            )
        elif index_type == 'depthwisem2o':
            self.index_generator = DepthwiseM2OIndexBlock(
                in_ch, win_size=stride, kernel_size=kernel_size,
                stride=stride, padding=padding,
            )
        else:
            self.index_generator = HolisticIndexBlock(
                in_ch, win_size=stride, kernel_size=kernel_size,
                stride=stride, padding=padding,
            )

    def forward(self, input):
        idx_en, idx_de = self.index_generator(input)
        output = input * idx_en
        output = self.kernel_size**2 * \
                 F.avg_pool2d(output, self.kernel_size, stride=self.stride)
        return output, idx_de


class IndexedUpsamlping(nn.Module):
    def __init__(self):
        super(IndexedUpsamlping, self).__init__()

    def forward(self, input, size, indices=None):
        if indices is not None:
            input = indices * F.interpolate(input, size=size, mode='nearest')
        else:
            input = F.interpolate(input, size=size, mode='bilinear', align_corners=True)
        return input
