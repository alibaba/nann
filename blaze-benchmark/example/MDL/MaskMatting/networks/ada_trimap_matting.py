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
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import Res50BasePytorch
from networks.efficient_net import build_efficient_net
from networks.shm_net import MattingVgg16, TMFusion
from networks.trimap_net import PSPNet50
from networks.pos_hrnet import get_pose_net, model_cfg
from networks.deeplabv3plus import DeeplabV3Plus
from cqutils import print_network
from config import *

__all__ = ['HRMNet',
        ]


class HRMNet(nn.Module):
    def __init__(self, hdfs_client=None, pretrained=False):
        super(HRMNet, self).__init__()
        self.hdfs_client = hdfs_client
        # cfg = model_cfg
        # cfg['MODEL']['NUM_JOINTS'] = 22
        # cfg['MODEL']['PRETRAINED'] = model_addr['HRNet_w48']
        # self.base = get_pose_net(
        #     cfg, True,
        #     hdfs_client=hdfs_client
        # )

        self.base = DeeplabV3Plus(
            num_classes=22,
            hdfs_client=hdfs_client,
            pretrained=True,
        )

        # for p in self.parameters():
        #     p.requires_grad = False

        # self.mnet = MattingVgg16(in_plane=6, frozen=False)

        if pretrained:
            self.init_weight()

    def forward(self, input):
        # mask = input[:, 3:, :, :]
        # input = input[:, :3, :, :]
        # output_mask = self.base(input)
        # output_mask = F.softmax(output_mask / 0.1, dim=1)
        # output_mask[:, 2:] = output_mask[:, 2:] * mask
        # output_mask[:, 1:2] = output_mask[:, 1:2] * mask
        # output_mask[:, :1] = output_mask[:, :1]
        output_mask = self.base(input)
        output_mask = F.softmax(output_mask, dim=1)
        
        return output_mask

        in_mnet = torch.cat(
            (input, output_mask[:, 2:],
             output_mask[:, 1:2],
             output_mask[:, :1]), 1)
        raw_alpha_patch, _ = self.mnet(in_mnet)

        if self.training:
            return raw_alpha_patch, output_mask

        # fusion
        alpha = output_mask.argmax(1, keepdim=True).float()
        alpha.masked_scatter_(
            alpha == 2, raw_alpha_patch[alpha == 2])
        # fg = output_mask[:, 1, :, :]
        # unk = output_mask[:, 2, :, :]
        # alpha = unk * raw_alpha_patch + fg

        return alpha, raw_alpha_patch, output_mask

    def init_weight(self):
        # load pretrained model for base
        checkpoint = torch.load(model_addr['PoseHighResolutionNet'])
        try:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                elif 'final_layer.weight' in k:
                    v_np = checkpoint['state_dict'][k].numpy()
                    m_v_np = np.zeros((3, v_np.shape[1], v_np.shape[2],
                                       v_np.shape[3]), v_np.dtype)
                    m_v_np[:2, :, :, :] = v_np
                    m_v = torch.from_numpy(m_v_np)
                    new_dict[k] = m_v
                else:
                    new_dict[k] = checkpoint['state_dict'][k]
            self.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                elif 'final_layer.weight' in k:
                    nk = 'module.' + k
                    v_np = checkpoint['state_dict'][nk].numpy()
                    m_v_np = np.zeros((3, v_np.shape[1], v_np.shape[2],
                                       v_np.shape[3]), v_np.dtype)
                    m_v_np[:2, :, :, :] = v_np
                    m_v = torch.from_numpy(m_v_np)
                    new_dict[k] = m_v
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint['state_dict'][nk]
            self.load_state_dict(new_dict)

        # load pretrained model for mnet
        # checkpoint = torch.load(model_addr['MattingVgg16'])
        # new_dict = OrderedDict()
        # for k, _ in self.mnet.state_dict().items():
        #     if 'num_batches_tracked' in k:
        #         new_dict[k] = torch.zeros(1)
        #     else:
        #         new_dict[k] = checkpoint['mnet.' + k]
        # self.mnet.load_state_dict(new_dict)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters()})
        # lr_list.append({'params': self.transition.parameters()})

        return lr_list

    def train(self, mode=True):
        super(HRMNet, self).train(mode)
        # for m in self.base.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()

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
                    new_dict[k] = checkpoint['state_dict'][k]
            netowrks.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in netowrks.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint['state_dict'][nk]
            netowrks.load_state_dict(new_dict)
