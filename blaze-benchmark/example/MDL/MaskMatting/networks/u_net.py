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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cqutils import initialize_weights

from config import *
import logging
import pyhdfs
import contextlib

__all__ = [
        'UNet',
        'EncoderDecoderVGG16',
        ]


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class EncoderDecoderVGGBase16(nn.Module):
    def __init__(self, num_classes, pretrained=False, hdfs_client=None):
        super(EncoderDecoderVGGBase16, self).__init__()
        in_plane = 3
        # conv1 and pool1
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_plane, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv2 and pool2
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv3 and pool3
        self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv4 and pool4
        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv5 and pool5
        self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                )
        # deconv5
        self.deconv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=5, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
                )
        # deconv4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=5, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                )
        # deconv3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
                )
        # deconv2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        # deconv1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )

        self.final = nn.Conv2d(64, num_classes, kernel_size=5, padding=2)
        # self._initialize_weights()
        self._initialize_weights_with_xavier()
        logging.info('Init weight from xavier random done')
        frozen = False
        if frozen:
            self._frozen_state1_params()
            print('*********************************************************')
            print('\nFrozen MattingNet params done.\n')
            print('*********************************************************')
    
    def _frozen_state1_params(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
            # print('conv1 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv2.parameters():
            param.requires_grad = False
            # print('conv2 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv3.parameters():
            param.requires_grad = False
            # print('conv3 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv4.parameters():
            param.requires_grad = False
            # print('conv4 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv5.parameters():
            param.requires_grad = False
        # for param in self.conv6.parameters():
        #     param.requires_grad = False
        # for param in self.deconv6.parameters():
        #     param.requires_grad = False
        # for param in self.unpool5.parameters():
        #     param.requires_grad = False
        for param in self.deconv5.parameters():
            param.requires_grad = False
        for param in self.unpool4.parameters():
            param.requires_grad = False
        for param in self.deconv4.parameters():
            param.requires_grad = False
        for param in self.unpool3.parameters():
            param.requires_grad = False
        for param in self.deconv3.parameters():
            param.requires_grad = False
        for param in self.unpool2.parameters():
            param.requires_grad = False
        for param in self.deconv2.parameters():
            param.requires_grad = False
        for param in self.unpool1.parameters():
            param.requires_grad = False
        for param in self.deconv1.parameters():
            param.requires_grad = False
        for param in self.raw_alpha_pred.parameters():
            param.requires_grad = False

    def forward(self, x, gt_comp = None, fgs=None, bgs=None):
        size1 = x.size()
        x, idx1 = self.conv1(x)
        size2 = x.size()
        x, idx2 = self.conv2(x)
        size3 = x.size()
        x, idx3 = self.conv3(x)
        size4 = x.size()
        x, idx4 = self.conv4(x)
        size5 = x.size()
        # x, idx5 = self.conv5(x)
        x = self.conv5(x)
        # x       = self.conv6(x)
        # x       = self.deconv6(x)

        # x       = self.unpool5(x, idx5, output_size=size5)
        x       = self.deconv5(x)
        
        x       = self.unpool4(x, idx4, output_size=size4)
        x       = self.deconv4(x)
        
        x       = self.unpool3(x, idx3, output_size=size3)
        x       = self.deconv3(x)
        
        x       = self.unpool2(x, idx2, output_size=size2)
        x       = self.deconv2(x)
        
        x       = self.unpool1(x, idx1, output_size=size1)
        x       = self.deconv1(x)
        # x       = F.sigmoid(self.raw_alpha_pred(x))
        #####################################################
        x       = self.final(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print(m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print(m)
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def _initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniform(m.weight.data)
                # print('init nn.Conv2d:{}'.format(m))
                # print('m.weight.data.shape:{}'.format(m.weight.data.size()))
                # print('m.bias.data.shape:{}'.format(m.bias.data.size()))
                if m.bias is not None:
                    m.bias.data.zero_()
                else:
                    print('m:{} has no bias'.format(m))
            elif isinstance(m, nn.BatchNorm2d):
                print('nn.BatchNorm2d:{}'.format(m))
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print('nn.BatchNorm2d:{}'.format(m))
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def EncoderDecoderVGG16(num_classes, pretrained=False, hdfs_client=None):
    ed_model = EncoderDecoderVGGBase16(num_classes=num_classes)
    ed_dict = ed_model.state_dict() 
    import os
    if pretrained:
        pretrained_dict = None
        if hdfs_client is None:
            if not os.path.exists(model_addr['vgg16_bn']):
                raise RuntimeError('Please ensure {} exists.'.format(
                    model_addr['vgg16_bn']))
            pretrained_dict = torch.load(model_addr['vgg16_bn'])   
        else:
            model_tmp_path = '/tmp/vgg16_bn.pth'
            hdfs_client.copy_to_local(model_addr['vgg16_bn'],
                    model_tmp_path)
            pretrained_dict = torch.load(model_tmp_path)

        print('\n\n****************************************************')
        print('Reinit weight from {}'.format(model_addr['vgg16_bn']))
        print('****************************************************\n\n')

        pretrained_keys = list(pretrained_dict.keys())
        ed_keys = list(ed_dict.keys())
        print('type:{}'.format(type(ed_keys)))
        
        print('pretrained vgg16_bn model dict keys:\n{}'.format(pretrained_dict.keys()))
        print('ed model dict keys:\n{}'.format(ed_dict.keys()))
        param_dit = OrderedDict()
        for i, p_k in enumerate(pretrained_keys):
            if i < 28 + 4 * 13 - 2:
                m_k = ed_keys[i]
                # if i == 0:
                #     # the first conv, copy the first 3 chanels, and set the last one
                #     # to zero
                #     v_np = pretrained_dict[p_k].numpy()
                #     m_v_np = np.zeros((v_np.shape[0], in_plane, v_np.shape[2],
                #         v_np.shape[3]), v_np.dtype)
                #     m_v_np[:, :3, :, :] = v_np
                #     m_v = torch.from_numpy(m_v_np)
                #     param_dit[m_k] = m_v
                #     print('pretrained features first conv: idx={}, key={}'.format(i, p_k))
                #     print('ed key={}\n'.format(m_k))
                # elif i < 26 + 4 * 13:
                if i < 26 + 4 * 13:
                    # the rest convs and all bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained features the rest convs and all bias: idx={}, key={}'.format(i, p_k))
                    print('ed key={}\n'.format(m_k))
                elif i == 26 + 4 * 13:
                    # the pretrain first linear classifer is converted to conv
                    m_v_np = pretrained_dict[p_k].view(4096, 512, 7, 7).numpy()
                    m_v = torch.from_numpy(m_v_np)
                    param_dit[m_k] = m_v
                    print('pretrained first classifer conv: idx={}, key={}'.format(i, p_k))
                    print('ed key={}\n'.format(m_k))
                else:
                    # the pretrained first linear classifer bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained first classifer bias: idx={}, key={}'.format(i, p_k))
                    print('ed key={}\n'.format(m_k))
            else:
                print('pretrained else: idx={}, key={}'.format(i, p_k))
        ed_dict.update(param_dit)

        ed_model.load_state_dict(ed_dict)
    return ed_model




class UNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, hdfs_client=None):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')

 
def test(path):
    # model = UNet(num_classes=2)
    model = EncoderDecoderVGG16(num_classes=2, pretrained=True, hdfs_client=None)
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
