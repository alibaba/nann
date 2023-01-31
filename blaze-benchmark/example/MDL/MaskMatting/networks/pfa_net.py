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
cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import VGG
from cqutils import print_network
from config import *

__all__ = ['PFANet',
        ]

Normal_Type = 'batch'
Momentum = 0.1
def _normal_layer(in_ch):
    if Normal_Type == 'caffe':
        return BatchNormCaffe(in_ch)
    elif Normal_Type == 'group':
        return nn.GroupNorm(in_ch, in_ch)
    elif Normal_Type == 'spectral':
        return PlaceHolderLayer()
    elif Normal_Type == 'dist_batch':
        return DistributedBatchNorm2d(in_ch, momentum=Momentum)
    else:
        return nn.BatchNorm2d(in_ch, momentum=Momentum)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        k = kernel_size
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, (1, k), padding=(0, 4)),
            _normal_layer(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, (k, 1), padding=(4, 0)),
            _normal_layer(1),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, (k, 1), padding=(4, 0)),
            _normal_layer(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, (1, k), padding=(0, 4)),
            _normal_layer(1),
            nn.ReLU(inplace=True))

    def forward(self, input):
        br1 = self.branch1(input)
        br2 = self.branch2(input)
        attention = br1 + br2
        attention = F.sigmoid(attention)
        attention = attention.repeat(1, self.in_channels, 1, 1)
        return attention

class ChannelWiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelWiseAttention, self).__init__()

        self.in_channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(in_channels, in_channels // 2)
        self.dense2 = nn.Linear(in_channels // 2, in_channels)

    def forward(self, input):
        H, W = input.size()[2:]
        attention = self.gap(input)
        attention = attention.view(attention.size(0), -1)
        attention = F.relu(self.dense1(attention), inplace=True)
        attention = F.sigmoid(self.dense2(attention))
        attention = attention.view(attention.size(0), self.in_channels, 1, 1)
        attention = attention.repeat(1, 1, H, W)
        attention = input * attention
        return attention

class CFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFE, self).__init__()
        rate = [3, 5, 7]
        self.conv0 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[0],
                              padding=rate[0], bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[1],
                              padding=rate[1], bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, dilation=rate[2],
                              padding=rate[2], bias=False)
        self.bn = _normal_layer(out_channels * 4)

    def forward(self, input):
        cfe0 = self.conv0(input)
        cfe1 = self.conv1(input)
        cfe2 = self.conv2(input)
        cfe3 = self.conv3(input)
        cfe_concat = torch.cat([cfe0, cfe1, cfe2, cfe3], 1)
        cfe_concat = F.relu(self.bn(cfe_concat), inplace=True)
        return cfe_concat

class PFANet(nn.Module):
    def __init__(self, hdfs_client=None):
        super(PFANet, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512]
        self.base = VGG(cfg, hdfs_client=hdfs_client,
                        pretrained=model_addr['vgg16'])
        self.en1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            _normal_layer(64),
            nn.ReLU(inplace=True))
        self.en2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            _normal_layer(64),
            nn.ReLU(inplace=True))

        self.C3_cfe = CFE(256, 32)
        self.C4_cfe = CFE(512, 32)
        self.C5_cfe = CFE(512, 32)

        self.C345 = nn.Sequential(
            ChannelWiseAttention(384),
            nn.Conv2d(384, 64, 1),
            _normal_layer(64),
            nn.ReLU(inplace=True)
        )

        self.SA = SpatialAttention(64)
        self.C12 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=1),
            _normal_layer(64),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Conv2d(128, 1, 3, padding=1)

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in PFANet: {}".format(num_params))
        self.init_weight()

    def forward(self, input):
        encoders = self.base(input)
        encoders[0] = self.en1(encoders[0])
        encoders[1] = self.en2(encoders[1])

        encoders[2] = self.C3_cfe(encoders[2])
        encoders[3] = self.C4_cfe(encoders[3])
        encoders[4] = self.C5_cfe(encoders[4])
        encoders[3] = F.upsample(encoders[3], encoders[2].size()[2:],
                                 mode='bilinear', align_corners=True)
        encoders[4] = F.upsample(encoders[4], encoders[2].size()[2:],
                                 mode='bilinear', align_corners=True)

        en_345 = torch.cat(encoders[2:], 1)
        en_345 = self.C345(en_345)
        en_345 = F.upsample(en_345, input.size()[2:],
                            mode='bilinear', align_corners=True)

        encoders[1] = F.upsample(encoders[1], encoders[0].size()[2:],
                                 mode='bilinear', align_corners=True)
        en_12 = torch.cat(encoders[:2], 1)
        en_12 = self.C12(en_12)
        en_12 = en_12 * self.SA(en_345)

        output = torch.cat([en_12, en_345], 1)
        output = F.sigmoid(self.output_layer(output))

        return output

    def init_weight(self):
        self.en1.apply(layer_weights_init)
        self.en2.apply(layer_weights_init)
        self.C3_cfe.apply(layer_weights_init)
        self.C4_cfe.apply(layer_weights_init)
        self.C5_cfe.apply(layer_weights_init)
        self.C345.apply(layer_weights_init)
        self.C12.apply(layer_weights_init)
        self.SA.apply(layer_weights_init)
        self.output_layer.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.en1.parameters()})
        lr_list.append({'params': self.en2.parameters()})
        lr_list.append({'params': self.C3_cfe.parameters()})
        lr_list.append({'params': self.C4_cfe.parameters()})
        lr_list.append({'params': self.C5_cfe.parameters()})
        lr_list.append({'params': self.C345.parameters()})
        lr_list.append({'params': self.C12.parameters()})
        lr_list.append({'params': self.SA.parameters()})
        lr_list.append({'params': self.output_layer.parameters()})
        return lr_list

if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)
    k = 9
    in_channels = 32
    branch1 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, (1, k), padding=(0, 4)),
        _normal_layer(in_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // 2, 1, (k, 1), padding=(4, 0)),
        _normal_layer(1),
        nn.ReLU(inplace=True),
    )
    pred = branch1(input)
    print(pred[0].size())