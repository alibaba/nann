import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import cv2
import sys
import os
import math
from collections import OrderedDict

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import BottleNeck
from networks.matting_net import MattingNet
from networks.matting_net_indexed import MattingNetIndexed, UNetIndexed, MobileNetV2UNetIndex
from networks.Layers import _normal_layer
from cqutils import print_network
from config import *

__all__ = ['InductiveGF',
           'MNetRes',
           'MNetMA',
           'GaborLoss',
           'InductiveGFLoss'
        ]

class InductiveGF(nn.Module):
    def __init__(self, hdfs_client=None):
        super(InductiveGF, self).__init__()
        self.use_bn = True
        self.inplanes = 256
        self.expansion = 4
        self.bn_type = 'normal'
        self.momentum = 0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                4, 64, kernel_size=1,
                stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=7,
                stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=3,
                stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=3,
                stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.residual_block = self._make_layer(256, 3, stride=2)

        self.upsample_res = nn.Conv2d(
            1024, 256, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.depthwise_conv4 = nn.Conv2d(
            256, 256, kernel_size=1,
            stride=1, padding=0, bias=True)

        self.upsample_conv4 = nn.Conv2d(
            256, 128, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.depthwise_conv3 = nn.Conv2d(
            128, 128, kernel_size=1,
            stride=1, padding=0, bias=True)

        self.upsample_conv3 = nn.Conv2d(
            128, 64, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.depthwise_conv2 = nn.Conv2d(
            64, 64, kernel_size=1,
            stride=1, padding=0, bias=True)

        self.sa_conv4 = nn.Conv2d(
            1024, 64, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.sa_conv3 = nn.Conv2d(
            64, 64, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.sa_conv2 = nn.Conv2d(
            64, 64, kernel_size=1,
            stride=1, padding=0, bias=False)

        self.A_conv = nn.Sequential(
            nn.Conv2d(
                64, 3, kernel_size=3,
                stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.B_conv = nn.Sequential(
            nn.Conv2d(
                64, 1, kernel_size=3,
                stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, dilation=dilation,
                              stride=stride, bias=False),
                    self.normal_layer(planes * self.expansion, momentum=self.momentum),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride,
                              dilation=dilation, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, dilation, downsample,
                                 momentum=self.momentum,
                                 use_bn=self.use_bn, bn_type=self.bn_type))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes, momentum=self.momentum,
                                     use_bn=self.use_bn, bn_type=self.bn_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_res = self.residual_block(x_conv4)

        upsample_conv4 = self.upsample_res(
            F.upsample_bilinear(x_res, x_conv4.shape[2:]))
        upsample_conv4 = upsample_conv4 + self.depthwise_conv4(x_conv4)

        upsample_conv3 = self.upsample_conv4(
            F.upsample_bilinear(upsample_conv4, x_conv3.shape[2:]))
        upsample_conv3 = upsample_conv3 + self.depthwise_conv3(x_conv3)

        upsample_conv2 = self.upsample_conv3(
            F.upsample_bilinear(upsample_conv3, x_conv2.shape[2:]))
        upsample_conv2 = upsample_conv2 + self.depthwise_conv2(x_conv2)

        sa = self.sa_conv4(F.upsample_bilinear(x_res, x_conv4.shape[2:]))\
             + F.upsample_bilinear(x_conv1, x_conv4.shape[2:])
        sa = self.sa_conv3(F.upsample_bilinear(sa, x_conv3.shape[2:]))
        sa = self.sa_conv2(F.upsample_bilinear(sa, x_conv2.shape[2:]))

        A = self.A_conv(upsample_conv2 * sa)
        B = self.B_conv(upsample_conv2 * sa)
        A = F.upsample_bilinear(A, x.shape[2:])
        B = F.upsample_bilinear(B, x.shape[2:])

        output = (A * x[:, :3]).sum(1, keepdim=True) + B
        output.masked_fill_(output > 1, 1)
        output.masked_fill_(output < 0, 0)
        return output

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

class MNetRes(nn.Module):
    def __init__(self, hdfs_client=None, is_pretrained=True):
        super(MNetRes, self).__init__()
        self.matting = MattingNet(
            hdfs_client, in_plane=4, is_pretrained=is_pretrained)

        self.refine = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 5, padding=2)
        )

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in MNetRes: {}".format(num_params))

    def forward(self, input):
        alpha, _ = self.matting(input)
        alpha = input[:, -1] + alpha
        alpha.masked_fill_(alpha > 1, 1)
        alpha.masked_fill_(alpha < 0, 0)
        return alpha

class MNetMA(MattingNetIndexed):
    def __init__(self, hdfs_client=None, is_pretrained=True):
        super(MNetMA, self).__init__(hdfs_client, in_plane=4, is_pretrained=is_pretrained)
        self.multi = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self.add = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self.multi.apply(layer_weights_init)
        self.add.apply(layer_weights_init)

    def forward(self, x):
        size1 = x.size()
        x = self.conv1(x)
        x, idx1 = self.pool1(x)
        size2 = x.size()
        x = self.conv2(x)
        x, idx2 = self.pool2(x)
        size3 = x.size()
        x = self.conv3(x)
        x, idx3 = self.pool3(x)
        size4 = x.size()
        x = self.conv4(x)
        x, idx4 = self.pool4(x)
        size5 = x.size()
        x = self.conv5(x)

        x = self.deconv5(x)

        x = self.unpool4(x, size4[2:], indices=idx4)
        x = self.deconv4(x)

        x = self.unpool3(x, size3[2:], indices=idx3)
        x = self.deconv3(x)

        x = self.unpool2(x, size2[2:], indices=idx2)
        x = self.deconv2(x)

        x = self.unpool1(x, size1[2:], indices=idx1)
        x = self.deconv1(x)
        #####################################################
        m = self.multi(x)
        a = self.add(x)
        alpha = m * x[:, -1:] + a
        alpha.masked_fill_(alpha > 1, 1)
        alpha.masked_fill_(alpha < 0, 0)
        #####################################################
        return alpha

class GaborLoss(nn.Module):
    def __init__(self, kernel_size, orientation_num, sigma, Lambda, gamma):
        super(GaborLoss, self).__init__()
        self.kernel_size = kernel_size
        self.orientation_num = orientation_num
        self.sigma = sigma
        self.Lambda = Lambda
        self.gamma = gamma

        self.filters = nn.Conv2d(
            1, orientation_num, kernel_size,
            padding=(kernel_size-1)//2,
            bias=False)
        weight = np.zeros((
            orientation_num, 1, kernel_size, kernel_size), dtype='float32')
        for i in range(orientation_num):
            theta = (i * np.pi * 2)/orientation_num
            weight[i, 0, :, :] = self.kernel(kernel_size, sigma, theta, Lambda, gamma)
        self.filters.weight.data = torch.from_numpy(weight)
        for p in self.filters.parameters():
            p.requires_grad = False

    def forward(self, pred, gt, mask=None):
        if mask is None:
            loss = ((self.filters(pred) - self.filters(gt)) ** 2).mean()
        else:
            loss = ((self.filters(pred) - self.filters(gt))[mask] ** 2).mean()
        return loss

    def kernel(self, kernel_size, sigma, theta, Lambda, gamma):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        spatial_offset = (kernel_size-1)//2
        (y, x) = np.meshgrid(
            np.arange(-spatial_offset, spatial_offset+1),
            np.arange(-spatial_offset, spatial_offset+1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / Lambda * x_theta)
        return gb

class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, 3, 1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, 3, 1, bias=False)
        self.conv_x.weight.data = self.kernel_init(
            1, 1, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        self.conv_y.weight.data = self.kernel_init(
            1, 1, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
        for p in self.conv_x.parameters():
            p.requires_grad = False
        for p in self.conv_y.parameters():
            p.requires_grad = False

    def kernel_init(self, in_channels, out_channels, filt, kernel_size=3):
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        for i in range(in_channels):
            for j in range(out_channels):
                weight[i, j, :, :] = filt
        return torch.from_numpy(weight)

    def forward(self, input):
        G_x = self.conv_x(input)
        G_y = self.conv_y(input)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G

class InductiveGFLoss(nn.Module):
    def __init__(self):
        super(InductiveGFLoss, self).__init__()
        self.gaborLoss = GaborLoss(7, 16, 0.5, 5, 0.5)

    def forward(self, pred, alpha, mask, trimap):
        gabor_loss = self.gaborLoss(pred, alpha)
        local_loss_mask = (alpha - mask).abs() > 0.01
        local_loss = torch.mean(torch.sqrt(
            (pred[local_loss_mask] - alpha[local_loss_mask]) ** 2 + 1e-12))
        global_loss = torch.mean(torch.sqrt((pred - alpha) ** 2 + 1e-12))
        unk_loss = torch.sqrt(((pred - alpha)*trimap) ** 2 + 1e-12).sum() / trimap.sum()
        loss = 10 * global_loss + local_loss + 200 * gabor_loss
        return loss, global_loss, local_loss, gabor_loss, unk_loss


if __name__ == '__main__':
    def gabor_fn(kernel_size, sigma, theta, Lambda, gamma):
        sigma_x = sigma
        sigma_y = float(sigma) / gamma

        # Bounding box
        # nstds = 3  # Number of standard deviation sigma
        # xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
        # xmax = np.ceil(max(1, xmax))
        # ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
        # ymax = np.ceil(max(1, ymax))
        # xmin = -xmax
        # ymin = -ymax
        # (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        spatial_offset = (kernel_size-1)//2
        (y, x) = np.meshgrid(
            np.arange(-spatial_offset, spatial_offset+1),
            np.arange(-spatial_offset, spatial_offset+1))

        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / Lambda * x_theta)
        return gb

    print(gabor_fn(7, 0.5, 1, 5, 0.5).shape)
