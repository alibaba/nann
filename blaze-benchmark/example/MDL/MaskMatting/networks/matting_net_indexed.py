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
from networks.Layers import layer_weights_init
from networks.Layers import _normal_layer
from networks.backbone import Res50BasePytorch
from networks.efficient_net import build_efficient_net
from networks.shm_net import MattingVgg16, TMFusion
from networks.trimap_net import PSPNet50
from cqutils import print_network
from networks.index_pooling_upsample import IndexedPooling, IndexedUpsamlping
from networks.index_pooling_upsample import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from config import *

__all__ = ['MattingNetIndexed',
           'UNetIndexed',
           'MobileNetV2UNetIndex',
        ]

class MattingNetIndexed(nn.Module):
    def __init__(self, hdfs_client=None, in_plane=4, bn_type='normal', is_pretrained=False):
        super(MattingNetIndexed, self).__init__()
        self.hdfs_client = hdfs_client
        self.is_pretrained = is_pretrained
        self.bn_type = bn_type
        # conv1 and pool1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, 64, kernel_size=3, padding=1),
            self.normal_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.normal_layer(64),
            nn.ReLU(inplace=True),
        )

        # conv2 and pool2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.normal_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.normal_layer(128),
            nn.ReLU(inplace=True),
        )
        # self.pool2 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        # conv3 and pool3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
        )
        # self.pool3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        # conv4 and pool4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
        )

        # conv5 and pool5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        self.deconv5 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            self.normal_layer(512),
            nn.ReLU(inplace=True)
        )
        # deconv4
        self.deconv4 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            self.normal_layer(256),
            nn.ReLU(inplace=True)
        )
        # deconv3
        self.deconv3 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            self.normal_layer(128),
            nn.ReLU(inplace=True)
        )
        # deconv2
        self.deconv2 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            self.normal_layer(64),
            nn.ReLU(inplace=True)
        )
        # deconv1
        self.deconv1 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            self.normal_layer(64),
            nn.ReLU(inplace=True)
        )

        self.raw_alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self._initialize_weights_with_xavier()
        print('\nInit weight from xavier random done')
        if is_pretrained:
            self.load_pretrained(in_plane)

        self.pool1 = IndexedPooling(64)
        self.pool2 = IndexedPooling(128)
        self.pool3 = IndexedPooling(256)
        self.pool4 = IndexedPooling(512)

        self.pool1.apply(layer_weights_init)
        self.pool2.apply(layer_weights_init)
        self.pool3.apply(layer_weights_init)
        self.pool4.apply(layer_weights_init)

        self.unpool4 = IndexedUpsamlping()
        self.unpool3 = IndexedUpsamlping()
        self.unpool2 = IndexedUpsamlping()
        self.unpool1 = IndexedUpsamlping()

    def _initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
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

    def forward(self, x, gt_comp=None, fgs=None, bgs=None):
        # trimaps = x.cpu().data.cpu().numpy()[:, 3, :, :]
        # if gt_comp is not None:
        #     print('gt_comp in forward size: {}'.format(gt_comp.size()))
        # if fgs is not None:
        #     print('fgs size: {}'.format(fgs.size()))
        # if bgs is not None:
        #     print('bgs size: {}'.format(bgs.size()))
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
        # x, idx5 = self.conv5(x)
        x = self.conv5(x)
        # x       = self.conv6(x)
        # x       = self.deconv6(x)

        # x       = self.unpool5(x, idx5, output_size=size5)
        x = self.deconv5(x)

        x = self.unpool4(x, size4[2:], indices=idx4)
        x = self.deconv4(x)

        x = self.unpool3(x, size3[2:], indices=idx3)
        x = self.deconv3(x)

        x = self.unpool2(x, size2[2:], indices=idx2)
        x = self.deconv2(x)

        x = self.unpool1(x, size1[2:], indices=idx1)
        x = self.deconv1(x)
        # x       = F.sigmoid(self.raw_alpha_pred(x))
        #####################################################
        x = self.raw_alpha_pred(x)
        # ones = Variable(torch.ones(x.size()).cuda())
        # zeros = Variable(torch.zeros(x.size()).cuda())
        # x       = torch.min(torch.max(x, zeros), ones)
        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)
        #####################################################
        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        # if pred_comp is not None:
        #     print('pred_comp.size: {}'.format(pred_comp.size()))
        return x, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

    def load_pretrained(self, in_plane=4):
        pretrained_model = model_addr['vgg16']
        print('\n\n****************************************************')
        print('Reinit weight from {}'.format(pretrained_model))
        print('****************************************************\n\n')
        pretrained_dict = self.load_dict(pretrained_model)
        matting_dict = self.state_dict()

        pretrained_keys = pretrained_dict.keys()
        matting_keys = list(matting_dict.keys())

        print('pretrained vgg16 model dict keys:\n{}'.format(pretrained_dict.keys()))
        print('matting model dict keys:\n{}'.format(matting_dict.keys()))
        param_dit = OrderedDict()
        for i, p_k in enumerate(pretrained_keys):
            if i < 26 + 5 * 13:
                m_k = matting_keys[i]
                if i == 0:
                    # the first conv, copy the first 3 chanels, and set the last one
                    # to zero
                    v_np = pretrained_dict[p_k].numpy()
                    m_v_np = np.zeros((v_np.shape[0], in_plane, v_np.shape[2],
                                       v_np.shape[3]), v_np.dtype)
                    m_v_np[:, :3, :, :] = v_np
                    m_v = torch.from_numpy(m_v_np)
                    param_dit[m_k] = m_v
                    print('pretrained features first conv: idx={}, key={}'.format(i, p_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                elif i < 26 + 5 * 13:
                    # the rest convs and all bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained features the rest convs and all bias: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                elif i == 26 + 5 * 13:
                    # the pretrain first linear classifer is converted to conv
                    m_v_np = pretrained_dict[p_k].view(4096, 512, 7, 7).numpy()
                    m_v = torch.from_numpy(m_v_np)
                    param_dit[m_k] = m_v
                    print('pretrained first classifer conv: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                else:
                    # the pretrained first linear classifer bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained first classifer bias: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
            else:
                print('pretrained else: idx={}, key={}'.format(i, p_k))
        matting_dict.update(param_dit)

        self.load_state_dict(matting_dict)

    def load_dict(self, pretrained_model, num_batches_tracked=True):
        if self.hdfs_client is None:
            if not os.path.exists(pretrained_model):
                raise RuntimeError('Please ensure {} exists.'.format(
                    pretrained_model))
            checkpoint = torch.load(pretrained_model)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(pretrained_model,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)
        if num_batches_tracked:
            mapped_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                mapped_key = key
                mapped_state_dict[mapped_key] = value
                if 'running_var' in key:
                    mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1)
            return mapped_state_dict
        else:
            return checkpoint

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

    def train(self, mode=True):
        super(MattingNetIndexed, self).train(mode)
        for m in self.conv1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.conv2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.conv3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.conv4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.conv5.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class UNetIndexed(nn.Module):
    def __init__(self, hdfs_client=None, in_plane=4, bn_type='normal', is_pretrained=False):
        super(UNetIndexed, self).__init__()
        self.hdfs_client = hdfs_client
        self.is_pretrained = is_pretrained
        self.bn_type = bn_type
        # conv1 and pool1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, 64, kernel_size=3, padding=1),
            self.normal_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.normal_layer(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = IndexedPooling(64)

        # conv2 and pool2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.normal_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.normal_layer(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = IndexedPooling(128)

        # conv3 and pool3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.normal_layer(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = IndexedPooling(256)

        # conv4 and pool4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = IndexedPooling(512)

        # conv5 and pool5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.normal_layer(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        self.deconv5 = ASPP(512, 512, output_stride=16, bn_type=bn_type)

        # deconv4
        self.unpool4 = IndexedUpsamlping()
        self.deconv4 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512*2, 256, kernel_size=5, padding=2),
            self.normal_layer(256),
            nn.ReLU(inplace=True)
        )
        # deconv3
        self.unpool3 = IndexedUpsamlping()
        self.deconv3 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(256*2, 128, kernel_size=5, padding=2),
            self.normal_layer(128),
            nn.ReLU(inplace=True)
        )
        # deconv2
        self.unpool2 = IndexedUpsamlping()
        self.deconv2 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128*2, 64, kernel_size=5, padding=2),
            self.normal_layer(64),
            nn.ReLU(inplace=True)
        )
        # deconv1
        self.unpool1 = IndexedUpsamlping()
        self.deconv1 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(64*2, 64, kernel_size=5, padding=2),
            self.normal_layer(64),
            nn.ReLU(inplace=True)
        )

        self.raw_alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self._initialize_weights_with_xavier()
        print('\nInit weight from xavier random done')

    def _initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
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

    def forward(self, x, gt_comp=None, fgs=None, bgs=None):
        size1 = x.size()
        x1 = self.conv1(x)
        x2, idx1 = self.pool1(x1)

        size2 = x2.size()
        x2 = self.conv2(x2)
        x3, idx2 = self.pool2(x2)

        size3 = x3.size()
        x3 = self.conv3(x3)
        x4, idx3 = self.pool3(x3)

        size4 = x4.size()
        x4 = self.conv4(x4)
        x5, idx4 = self.pool4(x4)

        size5 = x5.size()
        out = self.conv5(x5)

        out = self.deconv5(out)

        out = self.unpool4(out, size4[2:], indices=idx4)
        out = self.deconv4(torch.cat((out, x4), 1))

        out = self.unpool3(out, size3[2:], indices=idx3)
        out = self.deconv3(torch.cat((out, x3), 1))

        out = self.unpool2(out, size2[2:], indices=idx2)
        out = self.deconv2(torch.cat((out, x2), 1))

        out = self.unpool1(out, size1[2:], indices=idx1)
        out = self.deconv1(torch.cat((out, x1), 1))

        #####################################################
        out = self.raw_alpha_pred(out)
        out.masked_fill_(out > 1, 1)
        out.masked_fill_(out < 0, 0)
        #####################################################
        pred_comp = self._composite4(fgs, bgs, out, trimaps=None)
        return out, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

"""
    IndexNet Matting
    Indices Matter: Learning to Index for Deep Image Matting
    https://github.com/poppinace/indexnet_matting/blob/master/scripts/hlmobilenetv2.py
"""

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, bn_type='normal'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.bn_type = bn_type

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                self.normal_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                self.normal_layer(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                self.normal_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                self.normal_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                self.normal_layer(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


def depth_sep_dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, padding=padding, dilation=dilation, groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, padding=padding, dilation=dilation, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class _ASPPModule(nn.Module):
    def __init__(self, inp, planes, kernel_size, padding, dilation, batch_norm):
        super(_ASPPModule, self).__init__()
        BatchNorm2d = batch_norm
        if kernel_size == 1:
            self.atrous_conv = nn.Sequential(
                nn.Conv2d(inp, planes, kernel_size=1, stride=1, padding=padding, dilation=dilation, bias=False),
                BatchNorm2d(planes),
                nn.ReLU6(inplace=True)
            )
        elif kernel_size == 3:
            # we use depth-wise separable convolution to save the number of parameters
            self.atrous_conv = depth_sep_dilated_conv_3x3_bn(inp, planes, padding, dilation, BatchNorm2d)

    def forward(self, x):
        x = self.atrous_conv(x)

        return x


class ASPP(nn.Module):
    def __init__(self, inp, oup, output_stride=32, bn_type='normal', width_mult=1.):
        super(ASPP, self).__init__()
        self.bn_type = bn_type

        if output_stride == 32:
            dilations = [1, 2, 4, 8]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        BatchNorm2d = self.normal_layer

        self.aspp1 = _ASPPModule(inp, int(256 * width_mult), 1, padding=0, dilation=dilations[0],
                                 batch_norm=BatchNorm2d)
        self.aspp2 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[1], dilation=dilations[1],
                                 batch_norm=BatchNorm2d)
        self.aspp3 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[2], dilation=dilations[2],
                                 batch_norm=BatchNorm2d)
        self.aspp4 = _ASPPModule(inp, int(256 * width_mult), 3, padding=dilations[3], dilation=dilations[3],
                                 batch_norm=BatchNorm2d)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inp, int(256 * width_mult), 1, stride=1, padding=0, bias=False),
            BatchNorm2d(int(256 * width_mult)),
            nn.ReLU6(inplace=True)
        )

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(int(256 * width_mult) * 5, oup, 1, stride=1, padding=0, bias=False),
            BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.bottleneck_conv(x)

        return self.dropout(x)

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class UNetDecoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=5, bn_type='normal'):
        super(UNetDecoder, self).__init__()
        self.oup = oup
        self.bn_type = bn_type

        self.dconv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, 1, padding=kernel_size//2, bias=False),
            self.normal_layer(oup),
            nn.ReLU6(inplace=True),
        )

    def forward(self, l_encode, l_low, indices=None):
        _, c, _, _ = l_encode.size()
        if indices is not None:
            l_encode = indices * F.interpolate(l_encode, size=l_low.size()[2:], mode='nearest')
        l_cat = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_cat)

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)


class MobileNetV2UNetIndex(nn.Module):
    def __init__(
            self,
            hdfs_client=None,
            in_plane=4,
            output_stride=32,
            width_mult=1.,
            decoder_kernel_size=5,
            apply_aspp=True,
            freeze_bn=False,
            use_nonlinear=False,
            indexnet='depthwise',
            index_mode='m2o',
            bn_type='normal'
    ):
        super(MobileNetV2UNetIndex, self).__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride
        self.index_mode = index_mode
        self.bn_type = bn_type

        block = InvertedResidual
        aspp = ASPP
        decoder_block = UNetDecoder

        if indexnet == 'holistic':
            index_block = HolisticIndexBlock
        elif indexnet == 'depthwise':
            if 'o2o' in index_mode:
                index_block = DepthwiseO2OIndexBlock
            elif 'm2o' in index_mode:
                index_block = DepthwiseM2OIndexBlock
            else:
                raise NameError
        else:
            raise NameError

        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # assert input_size % output_stride == 0
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_plane, initial_channel, 3, 1, padding=1, bias=False),
            self.normal_layer(initial_channel),
            nn.ReLU6(inplace=True),
        )

        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            inverted_residual_setting[i][4] = 1  # change stride
            if current_stride == output_stride:
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0])
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4])
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6])

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        # define index blocks
        if output_stride == 32:
            self.index0 = index_block(32, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index2 = index_block(24, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index3 = index_block(32, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index4 = index_block(64, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index6 = index_block(160, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
        elif output_stride == 16:
            self.index0 = index_block(32, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index2 = index_block(24, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index3 = index_block(32, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
            self.index4 = index_block(64, kernel_size=4, padding=1, use_nonlinear=use_nonlinear, bn_type=bn_type)
        else:
            raise NotImplementedError

        ### context aggregation ###
        if apply_aspp:
            self.dconv_pp = aspp(320, 160, output_stride=output_stride, bn_type=bn_type)
        else:
            self.dconv_pp = nn.Sequential(
                nn.Conv2d(320, 160, 1, 1, padding=0, bias=False),
                self.normal_layer(160),
                nn.ReLU6(inplace=True),
            )

        ### decoder ###
        self.decoder_layer6 = decoder_block(160 * 2, 96, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer5 = decoder_block(96 * 2, 64, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer4 = decoder_block(64 * 2, 32, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer3 = decoder_block(32 * 2, 24, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer2 = decoder_block(24 * 2, 16, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer1 = decoder_block(16 * 2, 32, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)
        self.decoder_layer0 = decoder_block(32 * 2, 32, kernel_size=decoder_kernel_size,
                                            bn_type=bn_type)

        self.pred = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(32, 1, decoder_kernel_size, 1, padding=decoder_kernel_size // 2, bias=False),
                self.normal_layer(1),
                nn.ReLU6(inplace=True),
            ),
            nn.Conv2d(1, 1, decoder_kernel_size, 1, padding=decoder_kernel_size // 2, bias=False)
        )

        self.apply(layer_weights_init)

    def _build_layer(self, block, layer_setting, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, bn_type=self.bn_type))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, bn_type=self.bn_type))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _stride(self, m, stride):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.stride = stride
                return

    def forward(self, x):
        # encode
        l0 = self.layer0(x)  # 4x320x320
        idx0_en, idx0_de = self.index0(l0)
        l0 = idx0_en * l0
        l0p = 4 * F.avg_pool2d(l0, (2, 2), stride=2)  # 32x160x160

        l1 = self.layer1(l0p)  # 16x160x160
        l2 = self.layer2(l1)  # 24x160x160
        idx2_en, idx2_de = self.index2(l2)
        l2 = idx2_en * l2
        l2p = 4 * F.avg_pool2d(l2, (2, 2), stride=2)  # 24x80x80

        l3 = self.layer3(l2p)  # 32x80x80
        idx3_en, idx3_de = self.index3(l3)
        l3 = idx3_en * l3
        l3p = 4 * F.avg_pool2d(l3, (2, 2), stride=2)  # 32x40x40

        l4 = self.layer4(l3p)  # 64x40x40
        idx4_en, idx4_de = self.index4(l4)
        l4 = idx4_en * l4
        l4p = 4 * F.avg_pool2d(l4, (2, 2), stride=2)  # 64x20x20

        l5 = self.layer5(l4p)  # 96x20x20
        l6 = self.layer6(l5)  # 160x20x20
        if self.output_stride == 32:
            idx6_en, idx6_de = self.index6(l6)
            l6 = idx6_en * l6
            l6p = 4 * F.avg_pool2d(l6, (2, 2), stride=2)  # 160x10x10
        elif self.output_stride == 16:
            l6p, idx6_de = l6, None

        l7 = self.layer7(l6p)  # 320x10x10

        # pyramid pooling
        l = self.dconv_pp(l7)  # 160x10x10

        # decode
        l = self.decoder_layer6(l, l6, idx6_de)
        l = self.decoder_layer5(l, l5)
        l = self.decoder_layer4(l, l4, idx4_de)
        l = self.decoder_layer3(l, l3, idx3_de)
        l = self.decoder_layer2(l, l2, idx2_de)
        l = self.decoder_layer1(l, l1)
        l = self.decoder_layer0(l, l0, idx0_de)

        l = self.pred(l)

        return l

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)