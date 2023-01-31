"""
    https://github.com/YudeWang/deeplabv3plus-pytorch/blob/master/lib/net/deeplabv3plus.py
"""
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from networks.backbone import ResNet, ResNet
from networks.Layers import layer_weights_init, _normal_layer
from networks.config import *

__all__ = ['DeeplabV3Plus',
        ]

class DeeplabV3Plus(nn.Module):
    def __init__(self, num_classes, hdfs_client=None,
                 pretrained=False, bn_type='dist_batch'):
        super(DeeplabV3Plus, self).__init__()
        self.num_classes = num_classes
        self.bn_type = bn_type
        self.backbone = ResNet(
            strides=[1, 2, 2, 1],
            dilations=[1, 1, 1, 2],
            layers=[3, 4, 23, 3],
            bn_type='dist_batch',
            pretrained=None)
        input_channel = 2048        
        self.aspp = ASPP(
            dim_in=input_channel,
            dim_out=256,
            rate=1,
            bn_mom =0.0003,
            bn_type=bn_type,
        )
        self.dropout1 = nn.Dropout(0.5)
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, 48, 1, 1, padding=1//2, bias=True),
            self.normal_layer(48, momentum=0.0003),
            nn.ReLU(inplace=True),
        )       
        self.cat_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, 1, padding=1,bias=True),
            self.normal_layer(256, momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
            self.normal_layer(256, momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, 1, padding=0)

        if pretrained:
            self.load_pretrained(model_addr['deeplabv3plus'])

    def forward(self, x):
        x_bottom = self.backbone(x)
        feature_aspp = self.aspp(x_bottom[-1])
        feature_aspp = self.dropout1(feature_aspp)
        # feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(x_bottom[1])
        feature_aspp = F.upsample(feature_aspp, feature_shallow.shape[2:],                          
                mode='bilinear', align_corners=True)
        feature_cat = torch.cat((feature_aspp, feature_shallow), 1)
        result = self.cat_conv(feature_cat) 
        result = self.cls_conv(result)
        # result = self.upsample4(result)
        result = F.upsample(result, x.shape[2:],                          
                mode='bilinear', align_corners=True)
        return result

    def load_pretrained(self, model_path):
        print('===> loading {}'.format(model_path))
        checkpoint = torch.load(model_path)
        new_dict = OrderedDict()
        layer_weights_init(self.modules())
        try:
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                elif 'cls_conv' in k:
                    v_np = checkpoint[k].cpu().numpy()
                    if 'bias' in k:
                        m_v_np = np.zeros(
                            (self.num_classes, *v_np.shape[1:])).astype(v_np.dtype)
                    else:
                        m_v_np = np.random.normal(
                            scale=0.01,
                            size=(self.num_classes, *v_np.shape[1:])).astype(v_np.dtype)
                    # m_v_np[:21] = v_np
                    m_v = torch.from_numpy(m_v_np)
                    new_dict[k] = m_v
                    print('===> {} mean {}'.format(k,
                        self.state_dict()[k].mean()))
                else:
                    new_dict[k] = checkpoint[k]
        except Exception as e:
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                elif 'cls_conv' in k:
                    nk = 'module.' + k
                    v_np = checkpoint[nk].cpu().numpy()
                    if 'bias' in k:
                        m_v_np = np.zeros(
                            (self.num_classes, *v_np.shape[1:])).astype(v_np.dtype)
                    else:
                        m_v_np = np.random.normal(
                            scale=0.01,
                            size=(self.num_classes, *v_np.shape[1:])).astype(v_np.dtype)
                    # m_v_np[:21] = v_np
                    m_v = torch.from_numpy(m_v_np)
                    new_dict[k] = m_v
                    print('===> {} mean {}'.format(k,
                        self.state_dict()[k].mean()))
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint[nk]
        state_dict = self.state_dict()
        state_dict.update(new_dict)
        self.load_state_dict(state_dict)

        print('===> loaded {}'.format(model_path))

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, bn_type='normal'):
        super(ASPP, self).__init__()
        self.bn_type = bn_type
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
            self.normal_layer(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
            self.normal_layer(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
            self.normal_layer(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
            self.normal_layer(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = self.normal_layer(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
            self.normal_layer(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
#       self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#             nn.BatchNorm2d(dim_out),
#             nn.ReLU(inplace=True),
#       )
    def forward(self, x):
        [b,c,row,col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#       feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)
