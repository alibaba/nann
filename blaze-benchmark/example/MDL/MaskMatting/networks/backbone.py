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
from collections import OrderedDict
cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from cqutils import initialize_weights
from networks.Layers import BatchNormCaffe, PlaceHolderLayer, DistributedBatchNorm2d, _normal_layer
from config import *

__all__ = ['Res50BasePytorch',
           'Res50BaseCaffe',
           'VGG',
           'ResNet',
        ]

class ResNet(nn.Module):
    def __init__(self, use_bn=True, bn_type='normal',
                 strides=[1, 2, 2, 2],
                 layers=[3, 4, 6, 3],
                 dilations=[1, 1, 1, 1],
                 momentum=0.1,
                 pretrained=None, hdfs_client=None):
        super(ResNet, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.bn_type = bn_type
        self.momentum = momentum
        self.pretrained = pretrained
        self.hdfs_client = hdfs_client
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = self.normal_layer(64, momentum=self.momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], dilation=dilations[3])
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in ResNet: {}".format(num_params))
        if self.pretrained is not None:
            self.load_pretrained()

    def forward(self, x):
        encoder = []
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        encoder.append(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder

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

    def load_pretrained(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            checkpoint = torch.load(self.pretrained)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)

        try:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint[k]
            self.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = k[len('module'):]
                    new_dict[k] = checkpoint[nk]
            self.load_state_dict(new_dict)
        print("=> loaded checkpoint '{}'".format(self.pretrained))

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class Res50BasePytorch(nn.Module):
    def __init__(self, use_bn=True, use_own_bn=False,
                 strides=[1, 2, 2, 2],
                 layers=[3, 4, 6, 3],
                 dilations=[1, 1, 1, 1],
                 pretrained=None, hdfs_client=None):
        super(Res50BasePytorch, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.use_own_bn = use_own_bn
        if self.use_own_bn:
            self.bn_type = 'caffe'
        else:
            self.bn_type = 'normal'
        self.pretrained = pretrained
        self.hdfs_client = hdfs_client
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = BatchNormCaffe(64) if self.use_own_bn else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], dilation=dilations[3])
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in ResNet: {}".format(num_params))
        if self.pretrained is not None:
            self.load_pretrained()

    def forward(self, x):
        encoder = []
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        encoder.append(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, dilation=dilation,
                              stride=stride, bias=False),
                    BatchNormCaffe(planes * self.expansion) if self.use_own_bn else \
                        nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride,
                              dilation=dilation, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, dilation, downsample,
                                 use_bn=self.use_bn, bn_type=self.bn_type))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes,
                                     use_bn=self.use_bn, bn_type=self.bn_type))

        return nn.Sequential(*layers)

    def load_pretrained(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            checkpoint = torch.load(self.pretrained)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)

        try:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint[k]
            self.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = k[len('module'):]
                    new_dict[k] = checkpoint[nk]
            self.load_state_dict(new_dict)
        print("=> loaded checkpoint '{}'".format(self.pretrained))


class Res50BaseCaffe(nn.Module):
    def __init__(self, use_bn=True, use_own_bn=False,
                 layers=[3, 4, 6, 3]):
        super(Res50BaseCaffe, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.use_own_bn = use_own_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=True)
        if self.use_bn:
            self.bn1 = BatchNormCaffe(64) if self.use_own_bn else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=1)
        self.layer4 = self._make_layer(512, layers[3], stride=1)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BatchNormCaffe(planes * self.expansion) if self.use_own_bn else \
                        nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, downsample,
                                 use_bn=self.use_bn, use_own_bn=self.use_own_bn))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes, use_bn=self.use_bn,
                                     use_own_bn=self.use_own_bn))

        return nn.Sequential(*layers)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, use_bn=True, momentum=0.1,
                 bn_type='normal'):
        super(BottleNeck, self).__init__()
        self.use_bn = use_bn
        self.bn_type = bn_type
        self.momentum = momentum
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                               dilation=dilation, bias=False)
        if self.use_bn:
            self.bn1 = self.normal_layer(planes, momentum=self.momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        if self.use_bn:
            self.bn2 = self.normal_layer(planes, momentum=self.momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               dilation=dilation, bias=False)
        if self.use_bn:
            self.bn3 = self.normal_layer(planes*4, momentum=self.momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

# vgg_cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
class VGG(nn.Module):
    def __init__(self, cfg, use_own_bn=True, hdfs_client=None,
                 pretrained=None):
        super(VGG, self).__init__()
        self.use_own_bn = use_own_bn
        self.hdfs_client = hdfs_client
        self.pretrained = pretrained
        self.features = nn.ModuleList(self.make_layers(cfg))

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in VGG: {}".format(num_params))
        if self.pretrained is not None:
            self.weight_init()

    def forward(self, x):
        encoder = []
        for layer in self.features:
            x = layer(x)
            encoder.append(x)
        return encoder

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        sub_layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.Sequential(*sub_layers))
                sub_layers = []
                sub_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if self.use_own_bn:
                        sub_layers += [conv2d, BatchNormCaffe(v), nn.ReLU(inplace=True)]
                    else:
                        sub_layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    sub_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        if len(sub_layers) > 0:
            layers.append(nn.Sequential(*sub_layers))

        return layers

    def weight_init(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            checkpoint = torch.load(self.pretrained)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)
        names = list(self.state_dict().keys())
        ckpt_names = list(checkpoint.keys())
        vgg_dict = OrderedDict()

        load_names = []
        for name in names:
            if 'num_batches_tracked' in name:
                # vgg_dict[name] = torch.zeros()
                pass
            else:
                load_names.append(name)
        for index, name in enumerate(load_names):
            vgg_dict[name] = checkpoint[ckpt_names[index]]
        self.load_state_dict(vgg_dict)
        print('VGG loaded {}'.format(self.pretrained))

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(cfg, use_own_bn=False,
                pretrained='D:/project/models/vgg16_bn-6c64b313.pth')
    input = torch.randn(1, 3, 128, 128)
    en = model(input)
    print(en[-1].sum())

    # names = list(model.state_dict().keys())
    # print(names)
    # print(model)
    # print(torchvision.models.vgg16_bn())
    # checkpoint = torch.load('C:/Users/shuixi.wb/Downloads/vgg16_bn-6c64b313.pth')
    # names = list(checkpoint.keys())
    # print(names)
    # num_params = 0
    # for p in model.parameters():
    #     num_params += p.numel()
