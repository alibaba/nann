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
from cqutils import initialize_weights
from networks.trimap_net import PSPNet50, PSPNet101
from networks.backbone import Res50BasePytorch
from networks.Loss import SaliencyLoss
from networks.Layers import BatchNormCaffe
from config import *

__all__ = ['PSPNetAddVN',
        ]

class PSPNetAddVN(nn.Module):
    def __init__(self, hdfs_client=None, is_trian=True):
        super(PSPNetAddVN, self).__init__()
        self.pspnet = PSPNet(hdfs_client=hdfs_client)
        if is_trian:
            self.vinet = VariationalNet(hdfs_client=hdfs_client)
            self.loss = SaliencyLoss(weight=[1.0, 0.6])
            self.vinet_loss = torch.nn.MSELoss()

    def forward(self, input, label=None):
        if label is not None:
            pred, img_encoder = self.pspnet(input)
            label_encoder = self.vinet(label)
            return [pred, img_encoder, label_encoder]
        else:
            pred, img_encoder = self.pspnet(input)
            return [pred, img_encoder]

    def model_loss(self, inputs, label):
        if len(inputs) > 2:
            pred, img_encoder = inputs[0], inputs[1]
            label_encoder = inputs[2]
            main_loss = self.loss(pred, label)
            loss = main_loss + self.vinet_loss(img_encoder, label_encoder)
            return loss, main_loss
        else:
            pred, img_encoder = inputs[0], inputs[1]
            main_loss = self.loss(pred, label)
            return main_loss, main_loss

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.pspnet.base.parameters(), 'lr': 0.1*lr})
        lr_list.append({'params': self.get_parameters(self.pspnet.ppm, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.pspnet.final, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.vinet, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.pspnet.ppm)})
        lr_list.append({'params': self.get_parameters(self.pspnet.final)})
        lr_list.append({'params': self.get_parameters(self.vinet)})

        return lr_list

    def get_parameters(self, model, bias=False):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.BatchNorm2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if bias and 'bias' in name:
                        yield param
                    elif not bias and 'weight' in name:
                        yield param

class PSPNet(nn.Module):
    def __init__(self, hdfs_client=None):
        super(PSPNet, self).__init__()
        self.hdfs_client = hdfs_client
        self.base = Res50BasePytorch(use_own_bn=True)
        for n, m in self.base.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample' in n:
                m.stride = (1, 1)
        for n, m in self.base.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        self.aux_logits = nn.Conv2d(1024, 1, kernel_size=1)

        self.init_weight()

    def forward(self, input):
        x_size = input.size()
        encoder = self.base(input)
        aux = self.aux_logits(encoder[3])
        x = self.ppm(encoder[4])
        x = self.final(x)

        pred = []
        pred.append(F.sigmoid(F.upsample(x, x_size[2:], mode='bilinear')))
        pred.append(F.sigmoid(F.upsample(aux, x_size[2:], mode='bilinear')))
        return pred, encoder[4]

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(model_addr['res50']):
                raise RuntimeError('Please ensure {} exists.'.format(
                    model_addr['res50']))
            self.base.load_state_dict(torch.load(model_addr['res50']), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(model_addr['res50'],
                                           model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

        self.ppm.apply(self.layer_weights_init)
        self.final.apply(self.layer_weights_init)
        self.aux_logits.apply(self.layer_weights_init)

    def layer_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            # init.normal(m.weight.data, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    length = len(param)
                    nn.init.constant(param, 0.0)
                    nn.init.constant(param[length//4:length//2], 1.0)
                elif 'weight' in name:
                    nn.init.uniform(param, -0.2, 0.2)
                    # nn.init.xavier_normal(param)

class VariationalNet(nn.Module):
    def __init__(self, hdfs_client=None):
        super(VariationalNet, self).__init__()
        self.hdfs_client = hdfs_client
        self.base = Res50BasePytorch(use_own_bn=True)
        for n, m in self.base.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.base.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)
        self.base.apply(self.layer_weights_init)

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(model_addr['res50']):
                raise RuntimeError('Please ensure {} exists.'.format(
                    model_addr['res50']))
            self.base.load_state_dict(torch.load(model_addr['res50']), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(model_addr['res50'],
                                           model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

    def layer_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            # init.normal(m.weight.data, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    length = len(param)
                    nn.init.constant(param, 0.0)
                    nn.init.constant(param[length//4:length//2], 1.0)
                elif 'weight' in name:
                    nn.init.uniform(param, -0.2, 0.2)
                    # nn.init.xavier_normal(param)

    def forward(self, input):
        encoder = self.base(input)
        return encoder[4]

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