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
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import Res50BasePytorch
from cqutils import print_network
from config import *


__all__ = ['HRNet',
        ]

global Normal_Type, Conv_Type, Momentum
Normal_Type = 'batch'
Conv_Type = 'normal'
Momentum = 0.1
CFG = {'MODEL':
           {'INIT_WEIGHTS': True,
            'NAME': 'pose_hrnet',
            'NUM_JOINTS': 1,
            'PRETRAINED': '/user/chenquan.cq/models/pretrained/hrnet_w32-36af842e.pth',
            'TARGET_TYPE': 'gaussian',
            'IMAGE_SIZE': [256, 256],
            'HEATMAP_SIZE': [64, 64],
            'SIGMA': 2,
            'EXTRA':
                {'PRETRAINED_LAYERS':['conv1',
                                      'bn1',
                                      'conv2',
                                      'bn2',
                                      'layer1',
                                      'transition1',
                                      'stage2',
                                      'transition2',
                                      'stage3',
                                      'transition3',
                                      'stage4'],
                 'FINAL_CONV_KERNEL': 1,
                 'STAGE2':
                     {'NUM_MODULES': 1,
                      'NUM_BRANCHES': 2,
                      'BLOCK': 'BASIC',
                      'NUM_BLOCKS': [4, 4],
                      'NUM_CHANNELS': [32, 64],
                      'FUSE_METHOD': 'SUM'},
                 'STAGE3':
                     {'NUM_MODULES': 4,
                      'NUM_BRANCHES': 3,
                      'BLOCK': 'BASIC',
                      'NUM_BLOCKS': [4, 4, 4],
                      'NUM_CHANNELS': [32, 64, 128],
                      'FUSE_METHOD': 'SUM'},
                 'STAGE4':
                     {'NUM_MODULES': 3,
                      'NUM_BRANCHES': 4,
                      'BLOCK': 'BASIC',
                      'NUM_BLOCKS': [4, 4, 4, 4],
                      'NUM_CHANNELS': [32, 64, 128, 256],
                      'FUSE_METHOD': 'SUM'},
                 },
            },
       'LOSS': {'USE_TARGET_WEIGHT': True},
       }


def _normal_layer(in_ch):
    global Normal_Type, Conv_Type, Momentum
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


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.master = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            _normal_layer(out_channels))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            _normal_layer(out_channels))

    def forward(self, input):
        master = F.relu(self.master(input))
        gate = self.gate(input)
        output = master * F.sigmoid(gate)
        return output


def _conv2d(*args, **kwargs):
    global Normal_Type, Conv_Type, Momentum
    if Conv_Type == 'normal':
        return nn.Conv2d(*args, **kwargs)
    else:
        return GatedConv2d(*args, **kwargs)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return _conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = _normal_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = _normal_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = _conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = _normal_layer(planes)
        self.conv2 = _conv2d(planes, planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)
        self.bn2 = _normal_layer(planes)
        self.conv3 = _conv2d(planes, planes * self.expansion, kernel_size=1,
                             bias=False)
        self.bn3 = _normal_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                _conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                _normal_layer(
                    num_channels[branch_index] * block.expansion
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            _conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            _normal_layer(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    _conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    _normal_layer(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    _conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    _normal_layer(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRBase(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        cfg = CFG
        extra = cfg['MODEL']['EXTRA']
        super(HRBase, self).__init__()

        # stem net
        self.conv1 = _conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                             bias=False)
        self.bn1 = _normal_layer(64)
        self.conv2 = _conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                             bias=False)
        self.bn2 = _normal_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # self.final_layer = _conv2d(
        #     in_channels=pre_stage_channels[0],
        #     out_channels=cfg['MODEL']['NUM_JOINTS'],
        #     kernel_size=extra['FINAL_CONV_KERNEL'],
        #     stride=1,
        #     padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        # )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            _conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            _normal_layer(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            _conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            _normal_layer(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                _normal_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        encoder = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        encoder.extend(y_list)
        # x = self.final_layer(y_list[0])

        return encoder

    def init_weights(self, pretrained=''):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        # if os.path.isfile(pretrained):
        #     pretrained_state_dict = torch.load(pretrained)
        #     print('=> loading pretrained model {}'.format(pretrained))
        #
        #     need_init_state_dict = {}
        #     for name, m in pretrained_state_dict.items():
        #         if name.split('.')[0] in self.pretrained_layers \
        #            or self.pretrained_layers[0] is '*':
        #             need_init_state_dict[name] = m
        #     self.load_state_dict(need_init_state_dict, strict=False)
        # elif pretrained:
        #     print('=> please download pre-trained models first!')
        #     raise ValueError('{} is not exist!'.format(pretrained))


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, att_ch=[1024],
                 is_upsample=True, upsample_type='upsampleConv',
                 normal_type='batch', batchnorm_momentum=0.1):
        super(Decoder, self).__init__()
        self.sppa = SPPA(in_ch, normal_type=normal_type,
                         upsample_type=upsample_type,
                         batchnorm_momentum=batchnorm_momentum)
        self.in_ch = in_ch
        self.is_upsample = is_upsample
        self.upsample_type = upsample_type
        self.normal_type = normal_type
        self.att_ch = 1
        self.batchnorm_momentum = batchnorm_momentum

        if self.is_upsample:
            if self.upsample_type == 'ConvTranspose':
                self.dec_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2,
                                                   padding=1, bias=False, groups=in_ch)
            elif self.upsample_type == 'upsampleConv':
                self.dec_conv = nn.Conv2d(in_ch, in_ch,
                                          kernel_size=3,
                                          padding=1, bias=False)

        # Master branch
        self.conv_master = self.conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn_master = self.normal_layer(in_ch)

        # Global pooling branch
        self.conv_gpb = self.conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn_gpb = self.normal_layer(in_ch)

        # skip connect
        conv_list = []
        bn_list = []
        for ch in att_ch:
            conv_list.append(self.conv2d(ch, self.att_ch, kernel_size=5, padding=2))
            bn_list.append(self.normal_layer(self.att_ch))
        self.att_conv_list = nn.ModuleList(conv_list)
        self.att_bn_list = nn.ModuleList(bn_list)
        self.att_conv = self.conv2d((len(att_ch)+1)*self.att_ch, self.att_ch,
                                    kernel_size=1, padding=0)
        self.att_bn = self.normal_layer(self.att_ch)

        self.conv1 = _conv2d(2 * in_ch, in_ch, kernel_size=1, padding=0)
        self.conv2 = self.conv2d(2 * in_ch, out_ch, kernel_size=1, padding=0)
        self.bn_feature = self.normal_layer(out_ch)
        self.conv3 = _conv2d(out_ch, 1, kernel_size=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2=None, att_ups=None):
        # multi-stage FPA
        if input2 is None:
            en = input1
        else:
            en = input1
            dec = input2
            if self.is_upsample:
                if self.upsample_type == 'ConvTranspose':
                    dec = self.dec_conv(dec)
                elif self.upsample_type == 'upsampleConv':
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')
                    dec = self.dec_conv(dec)
                else:
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')
            en = self.relu(self.conv1(torch.cat((en, dec), 1)))

        att = self.sppa(en)
        fmap_att = att
        if att_ups is not None:
            size = input1.size()
            for idx, att_up in enumerate(att_ups):
                att_ups[idx] = F.upsample(att_ups[idx], size[2:], mode='bilinear')
                att_ups[idx] = self.att_bn_list[idx](self.att_conv_list[idx](att_ups[idx]))
                att_ups[idx] = self.relu(att_ups[idx])
                # fmap_att *= att_ups[idx]
                fmap_att = torch.cat((fmap_att, att_ups[idx]), 1)
            fmap_att = self.relu(self.att_bn(self.att_conv(fmap_att)))

        # Master branch
        x_master = self.conv_master(en)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(en.shape[2:])(en) \
            .view(en.shape[0], self.in_ch, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # merge master/global/attention
        x_master = x_master * fmap_att
        x_master = self.relu(x_master + x_gpb)

        # concatenate encoder
        x = torch.cat((en, x_master), 1)
        x = self.conv2(x)
        x = self.bn_feature(x)
        dec_out = F.relu(x)
        side_output = F.sigmoid(self.conv3(dec_out))

        return dec_out, att, side_output

    def normal_layer(self, in_ch):
        if self.normal_type == 'group':
            return nn.GroupNorm(in_ch, in_ch)
        elif self.normal_type == 'spectral':
            return PlaceHolderLayer()
        elif self.normal_type == 'dist_batch':
            momentum = self.batchnorm_momentum
            return DistributedBatchNorm2d(in_ch, momentum=momentum)
        else:
            momentum = self.batchnorm_momentum
            return nn.BatchNorm2d(in_ch, momentum=momentum)

    def conv2d(self, *args, **kwargs):
        if self.normal_type == 'spectral':
            return nn.utils.spectral_norm(_conv2d(*args, **kwargs))
        else:
            return _conv2d(*args, **kwargs)


class SPPA(nn.Module):
    def __init__(self, channels=2048,
                 upsample_type='upsampleConv',
                 normal_type='group',
                 batchnorm_momentum=0.001):
        super(SPPA, self).__init__()
        channels_mid = int(channels/4)
        channels_mid_2 = int(channels/8)
        channels_out = 1
        self.upsample_type = upsample_type
        self.channels_cond = channels
        self.normal_type = normal_type
        self.batchnorm_momentum = batchnorm_momentum

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = self.conv2d(self.channels_cond, channels_mid,
                                     kernel_size=(3, 3), stride=2, dilation=5,
                                     padding=5, bias=False)
        self.conv5x5_1 = self.conv2d(channels_mid, channels_mid,
                                     kernel_size=(3, 3), stride=2, dilation=3,
                                     padding=3, bias=False)
        self.conv3x3_1 = self.conv2d(channels_mid, channels_mid,
                                     kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn1_1 = self.normal_layer(channels_mid)
        self.bn2_1 = self.normal_layer(channels_mid)
        self.bn3_1 = self.normal_layer(channels_mid)

        self.conv7x7_2 = self.conv2d(channels_mid, channels_mid_2,
                                     kernel_size=(3, 3), stride=1, dilation=5,
                                     padding=5, bias=False)
        self.conv5x5_2 = self.conv2d(channels_mid, channels_mid_2,
                                     kernel_size=(3, 3), stride=1, dilation=3,
                                     padding=3, bias=False)
        self.conv3x3_2 = self.conv2d(channels_mid, channels_mid_2,
                                     kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1_2 = self.normal_layer(channels_mid_2)
        self.bn2_2 = self.normal_layer(channels_mid_2)
        self.bn3_2 = self.normal_layer(channels_mid_2)

        # Convolution Upsample
        if self.upsample_type == 'ConvTranspose':
            self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid_2, channels_mid_2,
                                                      kernel_size=4, stride=2,
                                                      padding=1, bias=False)
            self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid_2, channels_mid_2,
                                                      kernel_size=4, stride=2,
                                                      padding=1, bias=False)
            self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid_2, channels_out,
                                                      kernel_size=4, stride=2,
                                                      padding=1, bias=False)
            self.bn_upsample_3 = self.normal_layer(channels_mid_2)
            self.bn_upsample_2 = self.normal_layer(channels_mid_2)
            self.bn_upsample_1 = self.normal_layer(channels_out)
        elif self.upsample_type == 'upsampleConv':
            self.conv_upsample_3 = _conv2d(channels_mid_2, channels_mid_2,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=False)
            self.conv_upsample_2 = _conv2d(channels_mid_2, channels_mid_2,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=False)
            self.conv_upsample_1 = _conv2d(channels_mid_2, channels_out,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=False)
            self.bn_upsample_3 = self.normal_layer(channels_mid_2)
            self.bn_upsample_2 = self.normal_layer(channels_mid_2)
            self.bn_upsample_1 = self.normal_layer(channels_out)
        else:
            self.conv_upsample_1 = _conv2d(channels_mid_2, channels_out,
                                           kernel_size=1, stride=1,
                                           padding=0, bias=False)
            self.bn_upsample_1 = self.normal_layer(channels_out)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_size = x.size()
        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        if self.upsample_type == 'ConvTranspose':
            x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
            x2_merge = self.relu(x2_2 + x3_upsample)
            x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
            x1_merge = self.relu(x1_2 + x2_upsample)

            att = self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))
        elif self.upsample_type == 'upsampleConv':
            x3_upsample = F.upsample(x3_2, x2_2.size()[2:], mode='bilinear')
            x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_upsample)))
            x2_merge = self.relu(x2_2 + x3_upsample)
            x2_upsample = F.upsample(x2_merge, x1_2.size()[2:], mode='bilinear')
            x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_upsample)))
            x1_merge = self.relu(x1_2 + x2_upsample)

            x1_upsample = F.upsample(x1_merge, input_size[2:], mode='bilinear')
            att = self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_upsample)))
        else:
            x3_upsample = F.upsample(x3_2, x2_2.size()[2:], mode='bilinear')
            x2_merge = self.relu(x2_2 + x3_upsample)
            x2_upsample = F.upsample(x2_merge, x1_2.size()[2:], mode='bilinear')
            x1_merge = self.relu(x1_2 + x2_upsample)

            x1_upsample = F.upsample(x1_merge, input_size[2:], mode='bilinear')
            att = self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_upsample)))

        return att

    def normal_layer(self, in_ch):
        if self.normal_type == 'group':
            return nn.GroupNorm(in_ch, in_ch)
        elif self.normal_type == 'spectral':
            return PlaceHolderLayer()
        elif self.normal_type == 'dist_batch':
            momentum = self.batchnorm_momentum
            return DistributedBatchNorm2d(in_ch, momentum=momentum)
        else:
            momentum = self.batchnorm_momentum
            return nn.BatchNorm2d(in_ch, momentum=momentum)

    def conv2d(self, *args, **kwargs):
        if self.normal_type == 'spectral':
            return nn.utils.spectral_norm(_conv2d(*args, **kwargs))
        else:
            return _conv2d(*args, **kwargs)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = _conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = _normal_layer(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             _conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             _normal_layer(256),
                                             nn.ReLU())
        self.conv1 = _conv2d(1280, 256, 1, bias=False)
        self.bn1 = _normal_layer(256)
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


class DeeplabDecoder(nn.Module):
    def __init__(self, high_level_inplanes, low_level_inplanes):
        super(DeeplabDecoder, self).__init__()
        self.conv1 = _conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = _normal_layer(48)
        self.relu = nn.ReLU()
        last_conv_inplanes = high_level_inplanes + 48
        self.last_conv = nn.Sequential(_conv2d(last_conv_inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       _normal_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       _conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       _normal_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       _conv2d(256, 1, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class HRNet(nn.Module):
    def __init__(self, hdfs_client=None, upsample_type='upsampleConv'):
        super(HRNet, self).__init__()
        num_channels = CFG['MODEL']['EXTRA']['STAGE4']['NUM_CHANNELS']
        self.hdfs_client =hdfs_client
        self.upsample_type = upsample_type
        Normal_Type = 'caffe'
        self.base = HRBase()

        Normal_Type = 'batch'
        self.aspp = ASPP(num_channels[0], 16)
        self.decoder = DeeplabDecoder(256, 64)

        self.pretrained_layers = CFG['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        self.pretrained_path = CFG['MODEL']['PRETRAINED']
        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)

        output = self.aspp(encoder[1])
        output = self.decoder(output, encoder[0])

        return F.sigmoid(output)

    def init_weight(self):
        self.load_pretained(self.base, self.pretrained_path)
        self.aspp.apply(layer_weights_init)
        self.decoder.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.aspp.parameters()})
        lr_list.append({'params': self.decoder.parameters()})

        return lr_list

    def load_pretained(self, netowrks, pretrained):
        if self.hdfs_client is None:
            if not os.path.exists(pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    pretrained))
            checkpoint = torch.load(pretrained)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(pretrained,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)

        try:
            new_dict = OrderedDict()
            for k, _ in netowrks.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint[k]
            netowrks.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in netowrks.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint[nk]
            netowrks.load_state_dict(new_dict)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torchvision.models.resnet50()
    model = HRBase()
    # print([name for name, param in model.named_parameters()])
    model = model.to(device)
    input = torch.randn(1, 3, 512, 512)
    pred = model(input)
    for i in range(4):
        print(pred[i].size())