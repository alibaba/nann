from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
from collections import OrderedDict
import numpy as np

import os, sys
sys.path.append(os.path.dirname(__file__))

from trimap_net import PSPNet50 

import logging
from cqutils.common import imagenet_mean, imagenet_std, transpose_cuda_tensor
import nvtx

__all__ = [
        'MattingNet',
        'AutoHumanMatting'
        ]



class MattingNet(nn.Module):
    def __init__(self, in_plane=4, frozen=False):
        super(MattingNet, self).__init__()
        assert(in_plane >= 3)
        # conv1 and pool1
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_plane, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv2 and pool2
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv3 and pool3
        self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv4 and pool4
        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # conv5 and pool5
        self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                )
        # self.pool5 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        
        # conv6
        # self.conv6 = nn.Sequential(
        #         nn.Conv2d(512, 4096, kernel_size=7, padding=3),
        #         nn.ReLU(inplace=True),
        #         )
        
        # deconv6
        # self.deconv6 = nn.Sequential(
        #         nn.Conv2d(4096, 512, kernel_size=1, padding=0),
        #         nn.ReLU(inplace=True),
        #         )
        
        # deconv5
        # self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=5, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
                )
        # deconv4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=5, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                )
        # deconv3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
                )
        # deconv2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        # deconv1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )

        self.raw_alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        # self._initialize_weights()
        self._initialize_weights_with_xavier()
        logging.info('Init weight from xavier random done')
        if frozen:
            self._frozen_state1_params()
            print('*********************************************************')
            print('\nFrozen MattingNet params done.\n')
            print('*********************************************************')
        self.register_buffer('mean', imagenet_mean, persistent = False)
        self.register_buffer('std', imagenet_std, persistent = False)
    
    def _frozen_state1_params(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
            # print('conv1 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv2.parameters():
            param.requires_grad = False
            # print('conv2 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv3.parameters():
            param.requires_grad = False
            # print('conv3 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv4.parameters():
            param.requires_grad = False
            # print('conv4 param.requires_grad={}'.format(param.requires_grad))
        for param in self.conv5.parameters():
            param.requires_grad = False
        # for param in self.conv6.parameters():
        #     param.requires_grad = False
        # for param in self.deconv6.parameters():
        #     param.requires_grad = False
        # for param in self.unpool5.parameters():
        #     param.requires_grad = False
        for param in self.deconv5.parameters():
            param.requires_grad = False
        for param in self.unpool4.parameters():
            param.requires_grad = False
        for param in self.deconv4.parameters():
            param.requires_grad = False
        for param in self.unpool3.parameters():
            param.requires_grad = False
        for param in self.deconv3.parameters():
            param.requires_grad = False
        for param in self.unpool2.parameters():
            param.requires_grad = False
        for param in self.deconv2.parameters():
            param.requires_grad = False
        for param in self.unpool1.parameters():
            param.requires_grad = False
        for param in self.deconv1.parameters():
            param.requires_grad = False
        for param in self.raw_alpha_pred.parameters():
            param.requires_grad = False
    def transpose_normalize_concat(self, rescaled_img, rescaled_trimap):
        with nvtx.annotate("expand dims, transpose, dim_preprocess "):
            rescaled_trimap = rescaled_trimap.unsqueeze(2)
            rescaled_trimap = transpose_cuda_tensor(rescaled_trimap).unsqueeze(0)
            rescaled_img = rescaled_img.unsqueeze(0)
            inputs = transpose_cuda_tensor(rescaled_img)
        with nvtx.annotate("normalize and concat"):
            inputs = (inputs - self.mean) / self.std
            inputs = torch.cat((inputs, rescaled_trimap), 1)
        return inputs

    def forward(self, rescaled_img, rescaled_trimap, gt_comp = None, fgs=None, bgs=None):
        x = self.transpose_normalize_concat(rescaled_img, rescaled_trimap)
        size1 = x.size()
        x, idx1 = self.conv1(x)
        size2 = x.size()
        x, idx2 = self.conv2(x)
        size3 = x.size()
        x, idx3 = self.conv3(x)
        size4 = x.size()
        x, idx4 = self.conv4(x)
        size5 = x.size()
        # x, idx5 = self.conv5(x)
        x = self.conv5(x)
        # x       = self.conv6(x)
        # x       = self.deconv6(x)

        # x       = self.unpool5(x, idx5, output_size=size5)
        x       = self.deconv5(x)
        
        x       = self.unpool4(x, idx4, output_size=size4)
        x       = self.deconv4(x)
        
        x       = self.unpool3(x, idx3, output_size=size3)
        x       = self.deconv3(x)
        
        x       = self.unpool2(x, idx2, output_size=size2)
        x       = self.deconv2(x)
        
        x       = self.unpool1(x, idx1, output_size=size1)
        x       = self.deconv1(x)
        # x       = F.sigmoid(self.raw_alpha_pred(x))
        #####################################################
        x       = self.raw_alpha_pred(x)
        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)
        #####################################################
        return x  # pred_comp is not used
        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        return x, pred_comp
    
    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print(m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print(m)
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def _initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniform(m.weight.data)
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


def MattingVgg16(in_plane=4, pretrained_model=None, frozen=False):
    matting_model = MattingNet(in_plane=in_plane, frozen=frozen)
    if pretrained_model is not None:
        print('\n\n****************************************************')
        print('Reinit weight from {}'.format(pretrained_model))
        print('****************************************************\n\n')
        pretrained_dict = torch.load(pretrained_model)
        matting_dict = matting_model.state_dict()

        pretrained_keys = pretrained_dict.keys()
        matting_keys = matting_dict.keys()
        
        print('pretrained vgg16 model dict keys:\n{}'.format(pretrained_dict.keys()))
        print('matting model dict keys:\n{}'.format(matting_dict.keys()))
        param_dit = OrderedDict()
        for i, p_k in enumerate(pretrained_keys):
            if i < 28 + 4 * 13 - 2:
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
                    print('matting key={}\n'.format(m_k))
                elif i < 26 + 4 * 13:
                    # the rest convs and all bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained features the rest convs and all bias: idx={}, key={}'.format(i, p_k))
                    print('matting key={}\n'.format(m_k))
                elif i == 26 + 4 * 13:
                    # the pretrain first linear classifer is converted to conv
                    m_v_np = pretrained_dict[p_k].view(4096, 512, 7, 7).numpy()
                    m_v = torch.from_numpy(m_v_np)
                    param_dit[m_k] = m_v
                    print('pretrained first classifer conv: idx={}, key={}'.format(i, p_k))
                    print('matting key={}\n'.format(m_k))
                else:
                    # the pretrained first linear classifer bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained first classifer bias: idx={}, key={}'.format(i, p_k))
                    print('matting key={}\n'.format(m_k))
            else:
                print('pretrained else: idx={}, key={}'.format(i, p_k))
        matting_dict.update(param_dit)

        matting_model.load_state_dict(matting_dict)
    return matting_model

class TMFusion(nn.Module):
    def __init__(self, fg_thresh, bg_thresh, gt=None, trimap_channel=4):
        super(TMFusion, self).__init__()
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.trimap_channel = trimap_channel
        return
    def forward(self, trimap, alpha):
        # fg = trimap.clone()
        # fg.masked_fill_(fg < self.fg_thresh, 0)
        # bg = trimap.clone()
        # bg.masked_fill_(bg > self.bg_thresh, 1)

        #alpha.masked_scatter_(trimap > self.fg_thresh, trimap[trimap > self.fg_thresh])
        #alpha.masked_scatter_(trimap < self.bg_thresh, trimap[trimap < self.bg_thresh])

        # alpha_fusion = torch.max(fg, torch.min(bg, alpha))

        if self.trimap_channel == 4:
            alpha.masked_scatter_(trimap > self.fg_thresh, trimap[trimap > self.fg_thresh])
            alpha.masked_scatter_(trimap < self.bg_thresh, trimap[trimap < self.bg_thresh])
        else:
            fg = trimap[:, 1:2, :, :]
            # bg = 1. - trimap[:, 2:3, :, :]
            # alpha.masked_scatter_(fg > self.fg_thresh, fg[fg > self.fg_thresh])
            # alpha.masked_scatter_(bg < self.bg_thresh, bg[bg < self.bg_thresh])
            # alpha.masked_fill_(fg > self.fg_thresh, 1)
            # alpha.masked_fill_(bg < self.bg_thresh, 0)

            unk = trimap[:, 0:1, :, :]
            alpha = unk * alpha + fg

        return alpha

class AutoHumanMatting(nn.Module):
    """
    Semantic Human Matting end to end
    """
    def __init__(self, input_size, fg_thresh, bg_thresh, mnet_in_plane=4,
            mnet_frozen=False, tnet_frozen=False):
        super(AutoHumanMatting, self).__init__()
        self.input_size = input_size
        self.use_aux = True
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh

        self.mnet_in_plane = mnet_in_plane
        self.tnet = PSPNet50(num_classes=mnet_in_plane/2, pretrained=False,
                             use_aux=self.use_aux)
        self.mnet = MattingVgg16(in_plane=mnet_in_plane, frozen=mnet_frozen)
        self.tm_fusion = TMFusion(
            fg_thresh=self.fg_thresh, bg_thresh=self.bg_thresh,
            trimap_channel=mnet_in_plane)

    def forward(self, scale_img, ori_patch, info=None, gt_comp=None, fgs=None, bgs=None):
        """
        info: h, w, y1, x1, y2, x2, win_scale_y, win_scale_x, flip
        """
        crop_size = 320

        ### down-sample
        if self.training:
            scale_img = F.upsample(
                scale_img, (400, 400), mode='bilinear')
        else:
            ori_h = scale_img.size()[2]
            ori_w = scale_img.size()[3]
            #print("ori_h", ori_h, "ori_w", ori_w)
            need_rescale = False
            if math.sqrt(ori_h * ori_w) >= 700: 
                need_rescale = True
                scale_img = F.upsample(
                    scale_img, (round(ori_h/2), round(ori_w/2)), mode='bilinear') 

        if self.training and self.use_aux:
            out_tnet, out_tnet_aux = self.tnet(scale_img)
        else:
            out_tnet = self.tnet(scale_img)
            
        ### up-sample back
        if self.training:
            out_tnet = F.upsample(
                out_tnet, (800, 800), mode='bilinear')
        else:
            if need_rescale:
                out_tnet = F.upsample(
                    out_tnet, (ori_h, ori_w), mode='bilinear')

        out_tnet = F.softmax(out_tnet / 0.1, dim=1)
        if self.mnet_in_plane == 4:
            out_tnet = out_tnet[:, 1:, :, :]

        trimap_patch_ori = out_tnet
        # out_tnet = out_tnet.detach()
        # out_tnet.requires_grad = True

        if self.training:
            trimap_patch = []
            # crop to the original image patch
            for i in range(info.size(0)):
                y1, x1, y2, x2 = int(info.data[i,2]), int(info.data[i,3]), \
                        int(info.data[i,4]), int(info.data[i,5])
                flip = info.data[i, 8]

                raw_trimap_patch = out_tnet[i:i+1, :, y1:y2, x1:x2]
                # resize to target image patch
                raw_trimap_patch = F.upsample(
                    raw_trimap_patch, (crop_size, crop_size),
                    mode='bilinear')
                # flip
                if flip > 0:
                    inv_idx = torch.arange(raw_trimap_patch.size(3)-1, -1, -1).long().cuda()
                    inv_idx = Variable(inv_idx)
                    raw_trimap_patch_inv = raw_trimap_patch.index_select(3, inv_idx)
                    raw_trimap_patch = raw_trimap_patch_inv
                trimap_patch.append(raw_trimap_patch)
            trimap_patch = torch.cat(trimap_patch, 0)
        else:
            trimap_patch = out_tnet

        # in_mnet = torch.cat((ori_patch, trimap_patch.detach()), 1)
        in_mnet = torch.cat((ori_patch, trimap_patch), 1)
        raw_alpha_patch, raw_comp_patch = self.mnet(in_mnet, gt_comp=gt_comp, fgs=fgs, bgs=bgs)

        # final_alpha_patch = self.tm_fusion(trimap_patch.detach(), raw_alpha_patch)
        final_alpha_patch = self.tm_fusion(trimap_patch, raw_alpha_patch)

        final_comp_patch = self._composite4(fgs, bgs, final_alpha_patch, trimaps=None)
        return final_alpha_patch, final_comp_patch, trimap_patch, raw_alpha_patch, trimap_patch_ori

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

if __name__ == '__main__':
    # model = MattingVgg16(in_plane=4, pretrained_model='model/vgg16_bn-6c64b313.pth')
    # # model = MattingNet()
    # model.cuda()
    # print(model)

    # x = Variable(torch.randn(1, 4, 224, 224)).cuda()
    # y, _ = model.forward(x)
    # print('y.size:')
    # print(y.size())

    model = AutoHumanMatting(400, 0.9, 0.1)
    model.cuda()
    model.train()
    print(model)
    scale_img = Variable(torch.randn(1, 3, 224, 224)).cuda()
    ori_patch = Variable(torch.randn(1, 3, 320, 320)).cuda()
    info = torch.zeros(1, 9) 
    info[0,0] = 800
    info[0,1] = 800
    info[0,2] = 10
    info[0,3] = 10
    info[0,4] = 330
    info[0,5] = 330
    info[0,6] = 1
    info[0,7] = 1
    info[0,8] = 1
    info = Variable(info).cuda()
    alpha, comp, _, _, _ = model.forward(scale_img, ori_patch, info)
    print('alpha.shape:{}'.format(alpha.size()))
    # print('comp.shape:{}'.format(comp.size()))
    # model = PSPNet50(num_classes=2)
    # model.cuda()
    # model.eval()
    # print(model)
    # x = Variable(torch.randn(1, 3, 224, 224)).cuda()
    # y = model.forward(x)
    # print('y.size:')
    # print(y.size())
    
