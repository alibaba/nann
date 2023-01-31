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
import logging
import pyhdfs
import contextlib

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cqutils import initialize_weights
from config import model_addr

__all__ = [
        'ClsRes101_affinity',
        ]

'''
    sigma belongs to [0.1, 50] in the paper
    larger sigma indicates larger variance for a cluster
'''
class ClsRes101_affinity(nn.Module):
    def __init__(self, num_classes, num_centers, sigma,
            frozen=False, pretrained=True, hdfs_client=None):
        super(ClsRes101_affinity, self).__init__()
        self.num_classes = int(num_classes)
        self.num_centers = int(num_centers)
        self.sigma = sigma
        self.miu, self.rw = None, None
        self.resnet = models.resnet101()
        if pretrained:
            if hdfs_client is None:
                if not os.path.exists(model_addr['res101']):
                    raise RuntimeError('Please ensure {} exists.'.format(
                        model_addr['res101']))
                self.resnet.load_state_dict(torch.load(model_addr['res101']))
            else:
                self.hdfs_client = hdfs_client
                model_tmp_path = '/tmp/resnet101.pth'
                self.hdfs_client.copy_to_local(model_addr['res101'],
                        model_tmp_path)
                self.resnet.load_state_dict(torch.load(model_tmp_path))

        self.resnet_without_fc = nn.Sequential(*list(self.resnet.children())[:-1])
        
        num_ftrs = self.resnet.fc.in_features
        self.last_bn = nn.BatchNorm1d(num_ftrs)
        self.cluster_centers = nn.Parameter(
                torch.rand(num_classes, self.num_centers, num_ftrs))
        self.register_parameter("cluster_centers", self.cluster_centers)

        if frozen:
            print("----froze net")
            for param in self.parameters():
                param.requires_grad = False

    def dist(self, features):
        batch_size = len(features)
        f_expand = features.view(batch_size, 1, 1, -1)
        #print(f_expand.size())
        w_expand = self.cluster_centers.view(1, self.num_classes, self.num_centers, -1)
        #print(w_expand.size())
        fw_norm = f_expand - w_expand
        #print(fw_norm.size())
        fw_norm = (fw_norm ** 2).sum(dim=-1)
        #print(fw_norm.size())
        #print(fw_norm)
        distance = (- fw_norm / self.sigma).exp()
        distance = distance.max(dim=-1)[0]
        #print(distance.size())
        return distance

    def calc_rw(self):
        num = self.num_classes * self.num_centers
        w_reshape_expand1 = self.cluster_centers.view(1, num, -1)
        w_reshape_expand2 = self.cluster_centers.view(num, 1, -1)
        w_norm_mat = w_reshape_expand2 - w_reshape_expand1
        #print(w_norm_mat.size())
        w_norm_mat = (w_norm_mat **2).sum(dim=-1)
        #print(w_norm_mat.size())
        w_norm_upper = torch.triu(w_norm_mat)
        miu = 2.0 / (num ** 2 - num) * w_norm_upper.sum()
        #print("miu:", miu)
        residuals = torch.triu((w_norm_upper - miu) ** 2)
        rw = 2.0 / (num ** 2 - num) * residuals.sum()
        #rw = 1.0 - (- rw / 50000.0).exp() # newly added
        rw = torch.log(rw / self.sigma + 1) / 10 # newly added
        self.miu, self.rw = miu, rw
        return rw

    def forward(self, x):
        features = self.resnet_without_fc(x)
        features = self.last_bn(features)
        d = self.dist(features)
        rw = self.calc_rw()
        #print("cluster_centers:", self.cluster_centers)
        #print("d:", d)
        #print("rw:", rw)
        return d, rw

    def print_info(self):
        print("miu:", self.miu.data.cpu().numpy()[0],
                "rw:", self.rw.data.cpu().numpy()[0])
        #print("cluster_centers:", self.cluster_centers)

def test():
    model = ClsRes101_affinity(num_classes=5, num_centers=2, sigma=50)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    #print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #print(model.parameters())
    # the valid input size of images for resnet is 224X224
    x = Variable(torch.randn(1, 3, 224, 224))
    if torch.cuda.is_available():
        x = x.cuda()
    y, rw = model.forward(x)
    print('y:', y)
    print('y.size:', y.size())
    print('rw:', rw)


if __name__ == '__main__':
    test()
