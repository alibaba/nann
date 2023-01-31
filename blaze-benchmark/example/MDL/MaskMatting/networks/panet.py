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
from networks.Layers import layer_weights_init, PlaceHolderLayer, DistributedBatchNorm2d
from networks.backbone import Res50BasePytorch
from cqutils import print_network
from config import *

__all__ = ['PANet',
           'MSPANet',
        ]

class MSPANet(nn.Module):
    def __init__(self, hdfs_client=None, upsample_type='upsampleConv'):
        super(MSPANet, self).__init__()
        self.hdfs_client =hdfs_client
        self.upsample_type = upsample_type
        self.pretrained = model_addr['res101']
        self.base = Base(use_own_bn=True,
                         strides=[1, 2, 1, 1],
                         layers=[3, 4, 23, 3])
        self.decoder1 = MSDecoder(64, 64,
                                  att_ch=[1, 1, 1, 1],
                                  is_upsample=True,
                                  upsample_type=self.upsample_type)
        self.decoder2 = MSDecoder(256, 64,
                                  att_ch=[1, 1, 1],
                                  is_upsample=True,
                                  upsample_type=self.upsample_type)
        self.decoder3 = MSDecoder(512, 256,
                                  att_ch=[1, 1],
                                  is_upsample=False,
                                  upsample_type=self.upsample_type)
        self.decoder4 = MSDecoder(1024, 512,
                                  att_ch=[1],
                                  is_upsample=False,
                                  upsample_type=self.upsample_type)
        self.decoder5 = MSDecoder(2048, 1024,
                                  att_ch=[],
                                  is_upsample=False,
                                  upsample_type=self.upsample_type)

        self.hca = HCA(in_ch_list=[64, 64, 256, 512, 1024],
                       out_ch_list=[128, 128, 256, 256, 512],
                       kenel_list=[3, 3, 5, 5, 5])

        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)
        dec5, att5, side_output5 = self.decoder5(encoder[4])
        dec4, att4, side_output4 = self.decoder4(encoder[3], dec5, [att5])
        dec3, att3, side_output3 = self.decoder3(encoder[2], dec4, [att5, att4])
        dec2, att2, side_output2 = self.decoder2(encoder[1], dec3, [att5, att4, att3])
        dec1, att1, side_output1 = self.decoder1(encoder[0], dec2, [att5, att4, att3, att2])

        # output = [side_output1,
        #           side_output2,
        #           side_output3,
        #           side_output4,
        #           side_output5,
        #           ]

        output = self.hca(input, dec1, dec2, dec3, dec4, dec5)

        return output

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            self.base.load_state_dict(torch.load(self.pretrained), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

        self.decoder1.apply(layer_weights_init)
        self.decoder2.apply(layer_weights_init)
        self.decoder3.apply(layer_weights_init)
        self.decoder4.apply(layer_weights_init)
        self.decoder5.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1*lr})
        lr_list.append({'params': self.get_parameters(self.decoder1, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder2, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder3, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder4, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder5, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder1)})
        lr_list.append({'params': self.get_parameters(self.decoder2)})
        lr_list.append({'params': self.get_parameters(self.decoder3)})
        lr_list.append({'params': self.get_parameters(self.decoder4)})
        lr_list.append({'params': self.get_parameters(self.decoder5)})

        # lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        # lr_list.append({'params': self.decoder1.parameters()})
        # lr_list.append({'params': self.decoder2.parameters()})
        # lr_list.append({'params': self.decoder3.parameters()})
        # lr_list.append({'params': self.decoder4.parameters()})
        # lr_list.append({'params': self.decoder5.parameters()})
        # lr_list.append({'params': self.hca.parameters()})

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

class MSDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, att_ch=[1024],
                 is_upsample=True, upsample_type='upsampleConv',
                 normal_type='batch', batchnorm_momentum=0.1):
        super(MSDecoder, self).__init__()
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

        self.conv1 = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1, padding=0)
        self.conv2 = self.conv2d(2 * in_ch, out_ch, kernel_size=1, padding=0)
        self.bn_feature = self.normal_layer(out_ch)
        self.conv3 = nn.Conv2d(out_ch, 1, kernel_size=1, padding=0)

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
            return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
        else:
            return nn.Conv2d(*args, **kwargs)

class HCA(nn.Module):
    def __init__(self, in_ch_list, out_ch_list, kenel_list):
        super(HCA, self).__init__()
        self.hca1 = HCAModule(in_ch_list[0], out_ch_list[0],
                              kenel_list[0])
        self.hca2 = HCAModule(in_ch_list[1], out_ch_list[1],
                              kenel_list[1])
        self.hca3 = HCAModule(in_ch_list[2], out_ch_list[2],
                              kenel_list[2])
        self.hca4 = HCAModule(in_ch_list[3], out_ch_list[3],
                              kenel_list[3])
        self.hca5 = HCAModule(in_ch_list[4], out_ch_list[4],
                              kenel_list[4])
        self.side_conv1 = nn.Conv2d(25, 1, kernel_size=1)
        self.side_conv2 = nn.Conv2d(25, 1, kernel_size=1)
        self.side_conv3 = nn.Conv2d(25, 1, kernel_size=1)
        self.side_conv4 = nn.Conv2d(25, 1, kernel_size=1)
        self.side_conv5 = nn.Conv2d(25, 1, kernel_size=1)

        self.conv1 = nn.Conv2d(125, 125, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(125)
        self.conv2 = nn.Conv2d(125, 125, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(125)
        self.conv3 = nn.Conv2d(125, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        x1 = self.hca1(inputs[1])
        x2 = self.hca2(inputs[2])
        x3 = self.hca3(inputs[3])
        x4 = self.hca4(inputs[4])
        x5 = self.hca5(inputs[5])

        x1 = F.upsample(x1, inputs[0].size()[2:])
        x2 = F.upsample(x2, inputs[0].size()[2:])
        x3 = F.upsample(x3, inputs[0].size()[2:])
        x4 = F.upsample(x4, inputs[0].size()[2:])
        x5 = F.upsample(x5, inputs[0].size()[2:])

        x1_output = self.side_conv1(x1)
        x2_output = self.side_conv2(x2)
        x3_output = self.side_conv3(x3)
        x4_output = self.side_conv4(x4)
        x5_output = self.side_conv5(x5)

        x = torch.cat([x1, x2, x3, x4, x5], 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        x_output = F.sigmoid(x)
        x1_output = F.sigmoid(x1_output)
        x2_output = F.sigmoid(x2_output)
        x3_output = F.sigmoid(x3_output)
        x4_output = F.sigmoid(x4_output)
        x5_output = F.sigmoid(x5_output)
        output = [x_output,
                  x1_output,
                  x2_output,
                  x3_output,
                  x4_output,
                  x5_output]
        return output

class HCAModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(HCAModule, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                               padding=int((kernel - 1) / 2))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel,
                               padding=int((kernel - 1) / 2))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, 25, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)

        return x

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
            self.conv_upsample_3 = nn.Conv2d(channels_mid_2, channels_mid_2,
                                             kernel_size=3, stride=1,
                                             padding=1, bias=False)
            self.conv_upsample_2 = nn.Conv2d(channels_mid_2, channels_mid_2,
                                             kernel_size=3, stride=1,
                                             padding=1, bias=False)
            self.conv_upsample_1 = nn.Conv2d(channels_mid_2, channels_out,
                                             kernel_size=3, stride=1,
                                             padding=1, bias=False)
            self.bn_upsample_3 = self.normal_layer(channels_mid_2)
            self.bn_upsample_2 = self.normal_layer(channels_mid_2)
            self.bn_upsample_1 = self.normal_layer(channels_out)
        else:
            self.conv_upsample_1 = nn.Conv2d(channels_mid_2, channels_out,
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
            return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
        else:
            return nn.Conv2d(*args, **kwargs)

class PANet(nn.Module):
    def __init__(self, hdfs_client=None, upsample_type='upsampleConv'):
        super(PANet, self).__init__()
        self.hdfs_client =hdfs_client
        self.upsample_type = upsample_type
        self.base = Base(use_own_bn=True,
                         strides=[1, 2, 1, 1])
        self.decoder1 = Decoder(64, 64, is_upsample=True,
                                upsample_type=self.upsample_type)
        self.decoder2 = Decoder(256, 64, is_upsample=True,
                                upsample_type=self.upsample_type)
        self.decoder3 = Decoder(512, 256, is_upsample=False,
                                upsample_type=self.upsample_type)
        self.decoder4 = Decoder(1024, 512, is_upsample=False,
                                upsample_type=self.upsample_type)
        self.decoder5 = Decoder(2048, 1024, is_upsample=False,
                                upsample_type=self.upsample_type)

        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)
        dec5, side_output5 = self.decoder5(encoder[4])
        dec4, side_output4 = self.decoder4(encoder[3], dec5)
        dec3, side_output3 = self.decoder3(encoder[2], dec4)
        dec2, side_output2 = self.decoder2(encoder[1], dec3)
        _, side_output1 = self.decoder1(encoder[0], dec2)

        output = [side_output1,
                  side_output2,
                  side_output3,
                  side_output4,
                  side_output5,
                  ]

        return output

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

        self.decoder1.apply(layer_weights_init)
        self.decoder2.apply(layer_weights_init)
        self.decoder3.apply(layer_weights_init)
        self.decoder4.apply(layer_weights_init)
        self.decoder5.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1*lr})
        lr_list.append({'params': self.get_parameters(self.decoder1, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder2, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder3, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder4, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder5, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder1)})
        lr_list.append({'params': self.get_parameters(self.decoder2)})
        lr_list.append({'params': self.get_parameters(self.decoder3)})
        lr_list.append({'params': self.get_parameters(self.decoder4)})
        lr_list.append({'params': self.get_parameters(self.decoder5)})

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

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, is_upsample=True, upsample_type='upsampleConv'):
        super(Decoder, self).__init__()
        self.fpa = FPA(in_ch, upsample_type)
        self.is_upsample = is_upsample
        self.upsample_type = upsample_type
        if self.is_upsample:
            if self.upsample_type == 'ConvTranspose':
                self.deconv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2,
                                                 padding=1, bias=False, groups=in_ch)
            elif self.upsample_type == 'upsampleConv':
                self.upsample_conv = nn.Conv2d(in_ch, in_ch,
                                               kernel_size=5,
                                               padding=2, bias=False)
        self.conv1 = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=1, padding=0)
        self.bn_feature = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, 1, kernel_size=1, padding=0)

    def forward(self, input1, input2=None):
        if input2 is None:
            fmap = input1
        else:
            en = input1
            dec = input2
            if self.is_upsample:
                if self.upsample_type == 'ConvTranspose':
                    dec = self.deconv(dec)
                elif self.upsample_type == 'upsampleConv':
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')
                    dec = self.upsample_conv(dec)
                else:
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')

            fmap = torch.cat((en, dec), dim=1)
            fmap = self.conv1(fmap)
            fmap = F.relu(fmap)

        fmap_att = self.fpa(fmap)
        x = torch.cat((fmap, fmap_att), 1)
        x = self.conv2(x)
        x = self.bn_feature(x)
        dec_out = F.relu(x)
        side_output = F.sigmoid(self.conv3(dec_out))

        return dec_out, side_output

class PyramidAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PyramidAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3),
                               stride=(1, 1), padding=(3, 3),
                               dilation=(3, 3), bias=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3),
                               stride=(1, 1), padding=(5, 5),
                               dilation=(5, 5), bias=True)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3),
                               stride=(1, 1), padding=(7, 7),
                               dilation=(7, 7), bias=True)

    def forward(self, input):
        b1 = self.conv1(input)
        b2 = self.conv2(input)
        b3 = self.conv3(input)
        x = b1 + b2 + b3
        return F.sigmoid(x)

class FPA(nn.Module):
    def __init__(self, channels=2048, upsample_type='upsampleConv'):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)
        self.upsample_type = upsample_type
        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        if self.upsample_type == 'ConvTranspose':
            self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample_1 = nn.BatchNorm2d(channels)
        else:
            self.conv_upsample_3 = nn.Conv2d(channels_mid, channels_mid, kernel_size=5,
                                             stride=1, padding=2, bias=False)
            self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=5,
                                             stride=1, padding=2, bias=False)
            self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

            self.conv_upsample_1 = nn.Conv2d(channels_mid, channels, kernel_size=5,
                                             stride=1, padding=2, bias=False)
            self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

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

            x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))
        else:
            x3_upsample = F.upsample(x3_2, x2_2.size()[2:], mode='bilinear')
            x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_upsample)))
            x2_merge = self.relu(x2_2 + x3_upsample)
            x2_upsample = F.upsample(x2_merge, x1_2.size()[2:], mode='bilinear')
            x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_upsample)))
            x1_merge = self.relu(x1_2 + x2_upsample)

            x1_upsample = F.upsample(x1_merge, x_master.size()[2:], mode='bilinear')
            x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_upsample)))

        #
        out = self.relu(x_master + x_gpb)

        return out

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torchvision.models.resnet50()
    model = MSPANet()
    # print([name for name, param in model.named_parameters()])
    model = model.to(device)
    print_network(model)