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

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import Res50BasePytorch
from cqutils import print_network
from config import *

__all__ = ['HCA',
        ]

global Normal_Type, Momentum
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        if pretrained:
            self.pretrained = model_addr['res101']
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        low_level_feats = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        low_level_feats.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feats.append(x)
        x = self.layer2(x)
        low_level_feats.append(x)
        x = self.layer3(x)
        low_level_feats.append(x)
        x = self.layer4(x)
        low_level_feats.append(x)
        return x, low_level_feats

    def _load_pretrained_model(self):
        pretrain_dict = torch.load(self.pretrained)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

class HBlock(nn.Module):
    def __init__(self, channels, o_ch=None, kernel_size=3):
        super(HBlock, self).__init__()
        if o_ch is None:
            o_ch = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, o_ch, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2),
            _normal_layer(o_ch))
        self.conv2 = nn.Sequential(
            nn.Conv2d(o_ch, o_ch, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2),
            _normal_layer(o_ch))
        self.conv3 = nn.Sequential(
            nn.Conv2d(o_ch, 25, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2),
            _normal_layer(25))

    def forward(self, x, size):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, size=size, mode='bilinear',
                          align_corners=True)
        return x

class DBlock(nn.Module):
    def __init__(self, in_c1, in_c2, out_channels, kernel_size=3):
        super(DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c1, out_channels, kernel_size=1, stride=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c2, out_channels, kernel_size=1, stride=1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2),
            _normal_layer(out_channels))
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2),
            _normal_layer(out_channels))

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
            y = F.interpolate(y, x.size()[2:], mode='bilinear',
                              align_corners=True)
        x = F.relu(self.conv3(x + y))
        x = F.relu(self.conv4(x))
        return x

class UDecoder(nn.Module):
    def __init__(self):
        super(UDecoder, self).__init__()
        self.up4 = DBlock(1024, 2048, 256, 5)
        self.up3 = DBlock(512, 256, 256, 5)
        self.up2 = DBlock(256, 256, 128, 3)
        self.up1 = DBlock(64, 128, 128, 3)

    def forward(self, feats):
        final_feats = [feats[4]]
        x = self.up4(feats[3], feats[4])
        final_feats.append(x)
        x = self.up3(feats[2], x)
        final_feats.append(x)
        x = self.up2(feats[1], x)
        final_feats.append(x)
        x = self.up1(feats[0], x)
        final_feats.append(x)
        return final_feats[::-1]

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 512, 1, bias=False)
        self.bn1 = BatchNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

class Base(Res50BasePytorch):
    def forward(self, x):
        encoder = []

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder

class HCA(nn.Module):
    def __init__(self, hdfs_client=None):
        super(HCA, self).__init__()
        self.hdfs_client = hdfs_client
        # self.hb6 = HBlock(512)
        self.hb5 = HBlock(2048, 512, kernel_size=7)
        self.hb4 = HBlock(256, kernel_size=5)
        self.hb3 = HBlock(256, kernel_size=5)
        self.hb2 = HBlock(128, kernel_size=3)
        self.hb1 = HBlock(128, kernel_size=3)
        self.sconv1 = nn.Conv2d(25, 1, kernel_size=1, stride=1)
        self.sconv2 = nn.Conv2d(25, 1, kernel_size=1, stride=1)
        self.sconv3 = nn.Conv2d(25, 1, kernel_size=1, stride=1)
        self.sconv4 = nn.Conv2d(25, 1, kernel_size=1, stride=1)
        self.sconv5 = nn.Conv2d(25, 1, kernel_size=1, stride=1)
        # self.sconv6 = nn.Conv2d(25, 2, kernel_size=1, stride=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(125, 128, kernel_size=3, stride=1, padding=1),
            _normal_layer(128))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            _normal_layer(128))
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1)

        BatchNorm = _normal_layer
        output_stride = 16

        self.pretrained = model_addr['res101']
        self.backbone = Base(use_own_bn=False,
                             strides=[1, 2, 2, 2],
                             dilations=[1, 1, 1, 1],
                             layers=[3, 4, 23, 3])

        # self.aspp = ASPP(output_stride, BatchNorm)
        self.decoder = UDecoder()
        self.init_weight()

    def forward(self, x):
        size = x.size()[2:]
        low_level_feats = self.backbone(x)
        feats = self.decoder(low_level_feats)

        o1 = self.hb1(feats[0], size)
        o2 = self.hb2(feats[1], size)
        o3 = self.hb3(feats[2], size)
        o4 = self.hb4(feats[3], size)
        o5 = self.hb5(feats[4], size)

        out = torch.cat([o1, o2, o3, o4, o5], dim=1)

        o1 = F.sigmoid(self.sconv1(o1))
        o2 = F.sigmoid(self.sconv2(o2))
        o3 = F.sigmoid(self.sconv3(o3))
        o4 = F.sigmoid(self.sconv4(o4))
        o5 = F.sigmoid(self.sconv5(o5))

        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.sigmoid(self.conv3(out))
        return [out, o1, o2, o3, o4, o5]

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            self.backbone.load_state_dict(torch.load(self.pretrained), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            self.backbone.load_state_dict(torch.load(model_tmp_path), strict=False)

        self.hb5.apply(layer_weights_init)
        self.hb4.apply(layer_weights_init)
        self.hb3.apply(layer_weights_init)
        self.hb2.apply(layer_weights_init)
        self.hb1.apply(layer_weights_init)
        self.sconv1.apply(layer_weights_init)
        self.sconv2.apply(layer_weights_init)
        self.sconv3.apply(layer_weights_init)
        self.sconv4.apply(layer_weights_init)
        self.sconv5.apply(layer_weights_init)

        self.conv1.apply(layer_weights_init)
        self.conv2.apply(layer_weights_init)
        self.conv3.apply(layer_weights_init)

        self.decoder.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.backbone.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.hb1.parameters()})
        lr_list.append({'params': self.hb2.parameters()})
        lr_list.append({'params': self.hb3.parameters()})
        lr_list.append({'params': self.hb4.parameters()})
        lr_list.append({'params': self.hb5.parameters()})
        lr_list.append({'params': self.sconv1.parameters()})
        lr_list.append({'params': self.sconv2.parameters()})
        lr_list.append({'params': self.sconv3.parameters()})
        lr_list.append({'params': self.sconv4.parameters()})
        lr_list.append({'params': self.sconv5.parameters()})
        lr_list.append({'params': self.conv1.parameters()})
        lr_list.append({'params': self.conv2.parameters()})
        lr_list.append({'params': self.conv3.parameters()})
        lr_list.append({'params': self.decoder.parameters()})

        return lr_list