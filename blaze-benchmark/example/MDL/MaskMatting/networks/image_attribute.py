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
        'ClsRes101',
        ]

class ClsRes101(nn.Module):
    def __init__(self, num_classes, frozen = False, pretrained=True,
            hdfs_client=None):
        num_classes = int(num_classes)
        super(ClsRes101, self).__init__()
        self.resnet = models.resnet101()
        if pretrained:
            if hdfs_client is None:
                if not os.path.exists(model_addr['res101']):
                    raise RuntimeError('Please ensure {} exists.'.format(
                        model_addr['res101']))
                self.resnet.load_state_dict(torch.load(model_addr['res101']))
                print('\n\nResume model with pretrained model:{}\n\n'.format(model_addr['res101']))
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
    
        if frozen:
            print("----froze tnet")
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        y = self.resnet(x)
        return y

    
def test(path):
    model = ClsRes101(num_classes=2)
    model.cuda()
    model.eval()
    print(model)
    # the valid input size of images for resnet101 is 224X224
    x = Variable(torch.randn(1, 3, 224, 224)).cuda()
    y = model.forward(x)
    print('y.size:')
    print(y.size())


if __name__ == '__main__':
    test('')
