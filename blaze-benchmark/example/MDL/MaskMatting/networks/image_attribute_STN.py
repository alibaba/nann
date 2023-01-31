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
        'ClsRes101_STN',
        ]

class ClsRes101_STN(nn.Module):
    def __init__(self, num_classes, frozen = False, pretrained=True,
            hdfs_client=None):
        num_classes = int(num_classes)
        super(ClsRes101_STN, self).__init__()
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

        self.resnet.fc = nn.Linear(2048, num_classes)
        '''
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes),
                                        nn.LogSoftmax(dim=1))
        '''
        initialize_weights(self.resnet.fc)
    
        # Spatial transformer localization-network
        self.localization = models.resnet18(pretrained=True)

        # Regressor for the 3 * 2 affine matrix
        num_ftrs = self.localization.fc.in_features
        self.localization.fc = nn.Linear(num_ftrs,6)
        # Initialize the weights/bias with identity transformation
        self.localization.fc.weight.data.zero_()
        self.localization.fc.bias.data.copy_(torch.FloatTensor([1, 0, 0, 0, 1, 0]))
        #initialize_weights(self.localization.fc)

        if frozen:
            print("----froze tnet")
            for param in self.parameters():
                param.requires_grad = False

    # Spatial transformer network forward function
    def stn(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x = self.stn(x)
        y = self.resnet(x)
        return y

    
def test(path):
    model = ClsRes101_STN(num_classes=2)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print(model)
    # the valid input size of images for resnet is 224X224
    x = Variable(torch.randn(1, 3, 224, 224))
    if torch.cuda.is_available():
        x = x.cuda()
    y = model.forward(x)
    print('y.size:')
    print(y.size())


if __name__ == '__main__':
    test('')
