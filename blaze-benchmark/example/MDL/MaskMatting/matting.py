# -*- coding: utf-8 -*-
'''
    pytorch 1.0
'''
from __future__ import print_function
import argparse
import os
import os.path as osp
import sys
import math
import numpy as np
import warnings
import shutil
import traceback

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel
import random
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as tv_transforms
import torchvision
from networks.inductive_guided_filter import MNetMA

# working path
cur_path = osp.abspath(osp.dirname(__file__))
working_dir = osp.join(cur_path, '../')
sys.path.append(working_dir)

from cqutils import  dss_net_output_non_binary
from cqutils import extend_transforms
import networks

import cv2

network_names = sorted(name for name in networks.__dict__
                       if not name.startswith("__")
                       and callable(networks.__dict__[name]))

# set mean and std value from ImageNet dataset
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Testing settings
parser = argparse.ArgumentParser(description='Testing a Saliency network')

parser.add_argument('--max-size', dest='max_size',
                    help='max scale size (default: 256)', default=512, type=int)

parser.add_argument('--set-gpu', dest='set_gpu',
                    help='set gpu device (default: "0")', default='0', type=str)

parser.add_argument('--crop-size', dest='crop_size',
                    help='rand crop size (default: 256)', default=512, type=int)

parser.add_argument('--dataset-dir', dest='dataset_dir',
                    help='dataset dir (default: None)', default=None, type=str)

parser.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='dataset names to merge')

parser.add_argument('--data-type', dest='data_type',
                    help='loading data by tfs, http or local', default='local', type=str)

parser.add_argument('-j', '--workers', dest='workers',
                    help='dataloader workers', default=4, type=int)

parser.add_argument('--input-normalize', dest='input_normalize', action='store_true',
                    help='normalizing input data')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume-dir', dest='resume_dir',
                    help='dir to latest checkpoint (default: None)', default=None, type=str)

parser.add_argument('--arch', '-a', metavar='ARCH', default=None, choices=network_names,
                    help='model architecture: ' + ' | '.join(network_names) + ' (default: None)')

parser.add_argument('--loss-func', dest='loss_func',
                    help='model loss function')

parser.add_argument('--side-output', dest='side_output', action='store_true')

parser.add_argument('--additional-output', dest='additional_output', action='store_true')

parser.add_argument('--vis', dest='vis', action='store_true',
                    help='visualize validation dataset')

parser.add_argument('--vis-dir', dest='vis_dir',
                    help='visualization dir (default: None)', default=None, type=str)

class MaskMattingModel(object):
    def __init__(self):
        self.model = MNetMA(is_pretrained=False)
        self.model.cuda()
        self.model.eval()
        self.mean, self.std = torch.Tensor(mean).cuda(), torch.Tensor(std).cuda()
        self.cuda_image_tf = tv_transforms.Compose([extend_transforms.ImageFormatToTensor()])
        self.cuda_mask_tf = tv_transforms.Compose([extend_transforms.GrayImageFormatToTensor()])


    def predict(self, img, mask):
        img = self.cuda_image_tf(img)
        img = tv_transforms.Normalize(self.mean, self.std)(img)
        img = img.unsqueeze(0)

        mask = self.cuda_mask_tf(mask)
        mask = mask.unsqueeze(0)
        inputs = torch.cat((img, mask), 1)
        return self.model(inputs)


#class MaskMatting(object):
#    def __init__(self, use_cuda=True):
#        super(MaskMatting, self).__init__()
#        self.use_cuda = use_cuda
#        self.image_tf = tv_transforms.Compose(
#            [extend_transforms.SingleResize(512)
#             #extend_transforms.ImageToCudaTensor(),
#             #tv_transforms.Normalize(mean=mean, std=std)
#             ])
#
#        self.mask_tf = tv_transforms.Compose(
#            [extend_transforms.SingleResize(512, cv2.INTER_NEAREST)
#             #extend_transforms.GrayImageToCudaTensor(),
#             ])
#
#    def predict(self, img, mask):
#      with nvtx.annotate("mask4",color="red"):
#        '''
#            img: rgb image
#            mask: 0,255
#        '''
#        # compute output
#        #img = torch.from_numpy(img)
#        #mask = torch.from_numpy(mask)
#        h, w = img.shape[:2]
#        with nvtx.annotate("MaskMatting.image_tf"):
#            img = self.image_tf(img.copy())
#            img = torch.from_numpy(img)
#        with nvtx.annotate("MaskMatting.mask_tf"):
#            mask = self.mask_tf(mask.copy())
#            mask = torch.from_numpy(mask)
#        with nvtx.annotate("compute output"):
#            pred_alpha = hs.CPUClient.get('CudaMaskMatting').run("predict", [img, mask], 0)
#
#        # non_binary_output = dss_net_output_non_binary(pred_alpha)[0]
#        # non_binary_output = cv2.resize(non_binary_output, (w, h))
#        non_binary_output = pred_alpha[0, 0, :, :].numpy() * 255
#        non_binary_output = cv2.resize(non_binary_output, (w, h))
#        non_binary_output = non_binary_output.astype(np.uint8)
#        cv2.imwrite('/home/admin/alimama_interactivematting/h.png', non_binary_output)
#        return non_binary_output
#
#
#def predict():
#    args = parser.parse_args()
#    net = MaskMatting(model_path='')
#    dataset_names = args.datasets
#    for dataset_name in dataset_names:
#        save_path = os.path.join(args.vis_dir, dataset_name)
#        if args.vis and not os.path.exists(save_path):
#            print('Testing result path:{}'.format(save_path))
#            os.makedirs(save_path)
#
#        predict_dataset = saliency.DatasetIter(
#            root=args.dataset_dir,
#            dataset_names=[dataset_name],
#            hdfs_client=None,
#            img_transform=None,
#            data_type=args.data_type)
#
#        for idx, data in enumerate(predict_dataset):
#            try:
#                img = data
#                img_name = predict_dataset.file_list[idx].split('\t')[0].split('/')[-1]
#                alpha, mask, alpha_raw = net.predict(img)
#            except Exception as e:
#                print(traceback.format_exc())
#                continue
#            print('idx:{}, name{}'.format(idx, img_name))
#
#            img_path = os.path.join(save_path, '{}.jpg'.format(img_name))
#            alpha_raw_path = os.path.join(save_path, '{}_raw.png'.format(img_name))
#            mask_path = os.path.join(save_path, '{}_mask.png'.format(img_name))
#            pred_path = os.path.join(save_path, '{}.png'.format(img_name))
#
#            cv2.imwrite(img_path, img)
#            cv2.imwrite(alpha_raw_path, alpha_raw)
#            cv2.imwrite(mask_path, mask)
#            cv2.imwrite(pred_path, alpha)
#
#if __name__ == '__main__':
#    predict()
