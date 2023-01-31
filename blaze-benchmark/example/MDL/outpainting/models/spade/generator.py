"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spade.architecture import ResnetBlockWithSN
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class OutpaintingGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.shortcut_nc = opt.label_nc + 1
        self.opt = opt
        nf = opt.ngf

        self.actvn = nn.LeakyReLU(0.2, False)

        if opt.crop_size == 512 and hasattr(opt, 'downsample_first_layer') and opt.downsample_first_layer:
            stride = 2
        else:
            stride = 1
        self.conv_s = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3, stride=stride)

        self.conv_0 = ResnetBlockWithSN(1 * nf, 1 * nf, opt)

        self.down_0 = ResnetBlockWithSN(1 * nf, 2 * nf, opt, stride=2)
        self.conv_1 = ResnetBlockWithSN(2 * nf, 2 * nf, opt)

        self.down_1 = ResnetBlockWithSN(2 * nf, 4 * nf, opt, stride=2)
        self.conv_2 = ResnetBlockWithSN(4 * nf, 4 * nf, opt)

        self.down_2 = ResnetBlockWithSN(4 * nf, 8 * nf, opt, stride=2)
        self.conv_3 = ResnetBlockWithSN(8 * nf, 8 * nf, opt)

        self.down_3 = ResnetBlockWithSN(8 * nf, 16 * nf, opt, stride=2)
        self.conv_4 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.down_4 = ResnetBlockWithSN(16 * nf, 16 * nf, opt, stride=2)
        self.G_middle_0 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.G_middle_1 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.up_0 = ResnetBlockWithSN(16 * nf, 8 * nf, opt)
        self.up_10 = ResnetBlockWithSN(16 * nf, 8 * nf, opt)
        self.up_11 = ResnetBlockWithSN(8 * nf, 4 * nf, opt)
        self.up_20 = ResnetBlockWithSN(8 * nf, 4 * nf, opt)
        self.up_21 = ResnetBlockWithSN(4 * nf, 2 * nf, opt)
        self.up_30 = ResnetBlockWithSN(4 * nf, 2 * nf, opt)
        self.up_31 = ResnetBlockWithSN(2 * nf, 1 * nf, opt)

        if hasattr(opt, 'segmentation') and opt.segmentation:
            # generate seg map
            self.conv_img = nn.Conv2d(2 * nf, opt.label_nc, 3, padding=1)
        else:
            # generate image
            self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, image_masked, seg, mask):
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            x = torch.cat([image_masked, seg, mask], dim=1)
        else:
            x = torch.cat([image_masked, mask], dim=1)

        x = self.conv_s(x)                              # 64 * 256 * 256
        x = self.conv_0(x)
        short_0 = x

        x = self.down_0(x)                              # 128 * 128 * 128
        x = self.conv_1(x)
        short_1 = x

        x = self.down_1(x)                              # 256 * 64 * 64
        x = self.conv_2(x)
        short_2 = x

        x = self.down_2(x)                              # 512 * 32 * 32
        x = self.conv_3(x)
        short_3 = x

        x = self.down_3(x)                              # 1024 * 16 * 16
        x = self.conv_4(x)                              # 1024 * 16 * 16

        x = self.down_4(x)                              # 1024 * 8 * 8
        x = self.G_middle_0(x)                          # 1024 * 8 * 8

        x = self.up(x)                                  # 1024 * 16 * 16
        x = self.G_middle_1(x)                          # 1024 * 16 * 16

        x = self.up(x)                                  # 1024 * 32 * 32
        x = self.up_0(x)                                # 512 * 32 * 32

        x = torch.cat([x, short_3], dim=1)               # 1024 * 32 * 32
        x = self.up(x)                                   # 1024 * 64 * 64
        x = self.up_10(x)                                # 512 * 64 * 64
        x = self.up_11(x)                                # 256 * 64 * 64

        x = torch.cat([x, short_2], dim=1)               # 512 * 64 * 64
        x = self.up(x)                                   # 512 * 128 * 128
        x = self.up_20(x)                                # 256 * 128 * 128
        x = self.up_21(x)                                # 128 * 128 * 128

        x = torch.cat([x, short_1], dim=1)              # 256 * 128 * 128
        x = self.up(x)                                  # 256 * 256 * 256
        x = self.up_30(x)                               # 128 * 256 * 256
        x = self.up_31(x)                               # 64 * 256 * 256

        x = torch.cat([x, short_0], dim=1)              # 128 * 256 * 256

        if self.opt.crop_size == 512 and hasattr(self.opt, 'downsample_first_layer') and self.opt.downsample_first_layer:
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            # un-normalized log prob
            pass
        else:
            # rescale to [-1, 1]
            x = torch.tanh(x)

        return x
