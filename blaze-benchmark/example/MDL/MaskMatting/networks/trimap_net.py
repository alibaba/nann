from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models

import math
from collections import OrderedDict
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))
from cqutils import initialize_weights

from config import *
import logging
import pyhdfs
import contextlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cqutils.oss2_client import OSSCTD

__all__ = [
        'PSPNet50',
        'PSPNet101',
        ]

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

# class PSPRefineNet50(nn.Module):
#     def __init__(self, num_classes, frozen = False, pretrained=True,
#             use_aux=True, hdfs_client=None):
#         num_classes = int(num_classes)
#         super(PSPRefineNet50, self).__init__()
#         self.use_aux = use_aux
#         resnet = models.resnet50()
#         logging.info('resnet50:{}'.format(resnet))
#         import os
#         if pretrained:
#             if hdfs_client is None:
#                 if not os.path.exists(model_addr['res50']):
#                     raise RuntimeError('Please ensure {} exists.'.format(
#                         model_addr['res50']))
#                 resnet.load_state_dict(torch.load(model_addr['res50']))
#             else:
#                 self.hdfs_client = hdfs_client
#                 model_tmp_path = '/tmp/resnet50.pth'
#                 self.hdfs_client.copy_to_local(model_addr['res50'],
#                         model_tmp_path)
#                 resnet.load_state_dict(torch.load(model_tmp_path))
#         self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
#         self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
# 
#         for n, m in self.layer3.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
#         for n, m in self.layer4.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
# 
#         self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
#         self.final = nn.Sequential(
#             nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512, momentum=.95),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Conv2d(512, num_classes, kernel_size=1)
#         )
# 
#         self.refine = nn.Sequential(
#                 nn.Conv2d(4, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1)
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
#                 )
# 
#         if use_aux:
#             self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
#             initialize_weights(self.aux_logits)
# 
#         initialize_weights(self.ppm, self.final)
#     
#         if frozen:
#             print("----froze tnet")
#             for param in self.parameters():
#                 param.requires_grad = False
#                 
#     def forward(self, x):
#         x_size = x.size()
#         x = self.layer0(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         if self.training and self.use_aux:
#             aux = self.aux_logits(x)
#         x = self.layer4(x)
#         x = self.ppm(x)
#         x = self.final(x)
#         if self.training and self.use_aux:
#             return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:],
#                             mode='bilinear')
#         return F.upsample(x, x_size[2:], mode='bilinear')


class PSPNet50(nn.Module):
    def __init__(self, num_classes, frozen=False, pretrained=True,
            use_aux=True, oss_client=None):
        num_classes = int(num_classes)
        super(PSPNet50, self).__init__()
        self.use_aux = use_aux
        resnet = models.resnet50()
        logging.info('resnet50:{}'.format(resnet))
        import os
        self.oss_client = oss_client
        if pretrained:
            if oss_client is None:
                if not os.path.exists(model_addr['res50']):
                    raise RuntimeError('Please ensure {} exists.'.format(
                        model_addr['res50']))
                resnet.load_state_dict(torch.load(model_addr['res50']))
            else:
                model_tmp_path = '/tmp/resnet50.pth'
                model_tmp_wei = self.oss_client.read_file(model_addr2['res50'])
                with open(model_tmp_path, 'wb') as fout:
                    fout.write(model_tmp_wei)
                resnet.load_state_dict(torch.load(model_tmp_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)
    
        if frozen:
            print("----froze tnet")
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:],
                            mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')

class PSPNet101(nn.Module):
    def __init__(self, num_classes, frozen = False, pretrained=True,
            use_aux=True, hdfs_client=None):
        num_classes = int(num_classes)
        super(PSPNet101, self).__init__()
        self.use_aux = use_aux
        resnet = models.resnet101()
        logging.info('resnet101:{}'.format(resnet))
        import os
        if pretrained:
            if hdfs_client is None:

                if not os.path.exists(model_addr['res101']):
                    raise RuntimeError('Please ensure {} exists.'.format(
                        model_addr['res101']))
                resnet.load_state_dict(torch.load(model_addr['res101']))
            else:
                self.hdfs_client = hdfs_client
                model_tmp_path = '/tmp/resnet101.pth'
                self.hdfs_client.copy_to_local(model_addr['res101'],
                        model_tmp_path)
                resnet.load_state_dict(torch.load(model_tmp_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)
    
        if frozen:
            print("----froze tnet")
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:],
                            mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')
    
def test(path):
    model = PSPNet50(num_classes=2)
    model.cuda()
    model.eval()
    print(model)
    x = Variable(torch.randn(1, 3, 224, 224)).cuda()
    y = model.forward(x)
    y = F.softmax(y)    
    print('y.size:')
    print(y.size())


if __name__ == '__main__':
    test('')
