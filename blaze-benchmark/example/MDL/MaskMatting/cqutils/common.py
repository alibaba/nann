#!/usr/bin/python
#****************************************************************#
# ScriptName: cqutils/common.py
# Author: jinluyang.jly@alibaba-inc.com
# Create Date: 2022-01-14 15:19
# Modify Author: @alibaba-inc.com
# Modify Date: 2022-01-14 15:19
# Function: 
#***************************************************************#
import torch

# set mean and std value from ImageNet dataset
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
imagenet_mean = torch.Tensor(mean).reshape(1,3,1,1)
imagenet_std = torch.Tensor(std).reshape(1,3,1,1)


def transpose_cuda_tensor(pic):
    if pic.dim() == 3:
        img = pic.permute((2, 0, 1)) # HWC -> CHW
    else:
        img = pic.permute((0, 3, 1, 2)) #NHWC -> NCHW
    return img.float().div(255)
