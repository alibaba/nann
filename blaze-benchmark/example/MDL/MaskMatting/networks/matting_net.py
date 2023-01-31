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
from cqutils import print_network
from config import *

__all__ = ['MattingNet',
           'MattingRaw',
           'MattingRefining',
           'Matting',
           'MattingETE',
           'MattingETEAlpha',
           'AdaTimapDIM'
        ]

def _normal_layer(in_ch, use_own_bn=False):
    if use_own_bn:
        return BatchNormCaffe(in_ch)
    else:
        return nn.BatchNorm2d(in_ch)

'''
    SHM MattingNet
'''
class MattingNet(nn.Module):
    def __init__(self, hdfs_client=None, in_plane=4, is_pretrained=True):
        super(MattingNet, self).__init__()
        self.hdfs_client = hdfs_client
        self.is_pretrained = is_pretrained
        # conv1 and pool1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, 64, kernel_size=3, padding=1),
            _normal_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            _normal_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        # conv2 and pool2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            _normal_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            _normal_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        # self.pool2 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        # conv3 and pool3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            _normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            _normal_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            _normal_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        # self.pool3 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        # conv4 and pool4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        # self.pool4 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )
        # conv5 and pool5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            _normal_layer(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        # self.pool5 = nn.Sequential(
        #         nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #         )

        # # conv6
        # self.conv6 = nn.Sequential(
        #         nn.Conv2d(512, 4096, kernel_size=7, padding=3),
        #         nn.BatchNorm2d(4096),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.5)
        #         )
        #
        # # deconv6
        # self.deconv6 = nn.Sequential(
        #         nn.Conv2d(4096, 512, kernel_size=1, padding=0),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.5)
        #         )
        #
        # # deconv5
        # self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # deconv4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # deconv3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # deconv2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # deconv1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.raw_alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        self._initialize_weights_with_xavier()
        print('\nInit weight from xavier random done')
        if self.is_pretrained:
            self.load_pretrained(in_plane=in_plane)

    def _initialize_weights_with_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
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

    def forward(self, x, gt_comp=None, fgs=None, bgs=None):
        # trimaps = x.cpu().data.cpu().numpy()[:, 3, :, :]
        # if gt_comp is not None:
        #     print('gt_comp in forward size: {}'.format(gt_comp.size()))
        # if fgs is not None:
        #     print('fgs size: {}'.format(fgs.size()))
        # if bgs is not None:
        #     print('bgs size: {}'.format(bgs.size()))
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
        x = self.deconv5(x)

        x = self.unpool4(x, idx4, output_size=size4)
        x = self.deconv4(x)

        x = self.unpool3(x, idx3, output_size=size3)
        x = self.deconv3(x)

        x = self.unpool2(x, idx2, output_size=size2)
        x = self.deconv2(x)

        x = self.unpool1(x, idx1, output_size=size1)
        x = self.deconv1(x)
        # x       = F.sigmoid(self.raw_alpha_pred(x))
        #####################################################
        x = self.raw_alpha_pred(x)
        # ones = Variable(torch.ones(x.size()).cuda())
        # zeros = Variable(torch.zeros(x.size()).cuda())
        # x       = torch.min(torch.max(x, zeros), ones)
        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)
        #####################################################
        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        # if pred_comp is not None:
        #     print('pred_comp.size: {}'.format(pred_comp.size()))
        return x, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        ###########CHECK#####################
        # import cv2
        # print('fgs.size: {}'.format(fgs.cpu().data.size()))
        # print('fgs.shape:{}'.format(fgs.cpu().data.numpy().shape))
        # fgs_cv2 = fgs.cpu().data.numpy().transpose(0, 2, 3, 1)
        # bgs_cv2 = bgs.cpu().data.numpy().transpose(0, 2, 3, 1)
        # alphas_cv2 = alphas.data.cpu().numpy().transpose(0, 2, 3, 1)
        # comps_cv2 = comp.data.cpu().numpy().transpose(0, 2, 3, 1)
        # for bid in xrange(fgs_cv2.shape[0]):
        #     fg_img = fgs_cv2[bid, :, :, :]
        #     r, g, b = cv2.split(fg_img)
        #     fg_img = cv2.merge([b, g, r]) * 255
        #     fg_img = fg_img.astype(np.uint8)
        #     cv2.imwrite('data/test/batch_{}_fg.png'.format(bid), fg_img)
        #
        #     bg_img = bgs_cv2[bid, :, :, :]
        #     r, g, b = cv2.split(bg_img)
        #     bg_img = cv2.merge([b, g, r]) * 255
        #     bg_img = bg_img.astype(np.uint8)
        #     cv2.imwrite('data/test/batch_{}_bg.png'.format(bid), bg_img)
        #
        #     alpha_img = alphas_cv2[bid, :, :, 0] * 255
        #     print('alphas_cv2.shape:{}'.format(alphas_cv2.shape))
        #     print('alpha_img.shape:{}'.format(alpha_img.shape))
        #     alpha_img = alpha_img.astype(np.uint8)
        #     trimap_img = (trimaps[bid, :, :]*255).astype(np.uint8)
        #     cv2.imwrite('data/test/batch_{}_trimap.png'.format(bid), trimap_img)
        #     cv2.imwrite('data/test/batch_{}_alpha.png'.format(bid), alpha_img)
        #     alpha_img[np.where(trimap_img == 0)]  = 0
        #     alpha_img[np.where(trimap_img == 255)] = 255
        #     cv2.imwrite('data/test/batch_{}_alpha_merge.png'.format(bid), alpha_img)
        #     comp_img = comps_cv2[bid, :, :, :]
        #     print('comps_cv2.shape:{}'.format(comps_cv2.shape))
        #     print('comp_img.shape:{}'.format(comp_img.shape))
        #     r, g, b = cv2.split(comp_img)
        #     comp_img = cv2.merge([b, g, r]) * 255
        #     comp_img = comp_img.astype(np.uint8)
        #     cv2.imwrite('data/test/batch_{}_comp.png'.format(bid), comp_img)
        #####################################

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

    def load_pretrained(self, in_plane=4):
        pretrained_model = model_addr['vgg16']
        print('\n\n****************************************************')
        print('Reinit weight from {}'.format(pretrained_model))
        print('****************************************************\n\n')
        pretrained_dict = self.load_dict(pretrained_model)
        matting_dict = self.state_dict()

        pretrained_keys = pretrained_dict.keys()
        matting_keys = list(matting_dict.keys())

        print('pretrained vgg16 model dict keys:\n{}'.format(pretrained_dict.keys()))
        print('matting model dict keys:\n{}'.format(matting_dict.keys()))
        param_dit = OrderedDict()
        for i, p_k in enumerate(pretrained_keys):
            if i < 26 + 5 * 13:
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
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                elif i < 26 + 5 * 13:
                    # the rest convs and all bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained features the rest convs and all bias: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                elif i == 26 + 5 * 13:
                    # the pretrain first linear classifer is converted to conv
                    m_v_np = pretrained_dict[p_k].view(4096, 512, 7, 7).numpy()
                    m_v = torch.from_numpy(m_v_np)
                    param_dit[m_k] = m_v
                    print('pretrained first classifer conv: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
                else:
                    # the pretrained first linear classifer bias just copy
                    param_dit[m_k] = pretrained_dict[p_k]
                    print('pretrained first classifer bias: idx={}, key={}'.format(i, p_k))
                    # print('matting key={}\n'.format(m_k))
                    print('matting key={}, pretrained key={}\n'.format(m_k, p_k))
            else:
                print('pretrained else: idx={}, key={}'.format(i, p_k))
        matting_dict.update(param_dit)

        self.load_state_dict(matting_dict)

    def load_dict(self, pretrained_model, num_batches_tracked=True):
        if self.hdfs_client is None:
            if not os.path.exists(pretrained_model):
                raise RuntimeError('Please ensure {} exists.'.format(
                    pretrained_model))
            checkpoint = torch.load(pretrained_model)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(pretrained_model,
                                           model_tmp_path)
            checkpoint = torch.load(model_tmp_path)
        if num_batches_tracked:
            mapped_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                mapped_key = key
                mapped_state_dict[mapped_key] = value
                if 'running_var' in key:
                    mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1)
            return mapped_state_dict
        else:
            return checkpoint

'''
    mask2alpha:
        MattingRaw - PSPNet(mask)+MattingNet([input, mask]->raw alpha)
        MattingRefining - PSPNet(mask)+MattingNet(raw alpha)+fusion
        Matting - PSPNet(mask)+MattingNet(input*mask->raw alpha)
'''
class MattingRaw(nn.Module):
    def __init__(self, hdfs_client=None):
        super(MattingRaw, self).__init__()
        self.hdfs_client = hdfs_client
        self.pspnet_pretrained = model_addr['PSPNet50']
        self.pspnet = PSPNet50(2, frozen=False, pretrained=False,
                               use_aux=False, hdfs_client=hdfs_client)
        for p in self.parameters():
            p.requires_grad = False
        self.matting_net = MattingNet(hdfs_client=hdfs_client,
                                      in_plane=4, is_pretrained=True)

        self.weight_init()

    def forward(self, input, gt_comp=None, fgs=None, bgs=None):
        mask = self.pspnet(input)
        _, mask = torch.max(F.softmax(mask, 1), 1)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 1)

        x, _ = self.matting_net(torch.cat([input, mask], 1))

        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)

        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        if not self.training:
            return x, pred_comp, mask
        return x, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

    def weight_init(self):
        self.load_pretained(self.pspnet, self.pspnet_pretrained)

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

    def train(self, mode=True):
        super(MattingRaw, self).train(mode)
        for m in self.matting_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MattingRefining(nn.Module):
    def __init__(self, hdfs_client=None, in_plane=5):
        super(MattingRefining, self).__init__()
        self.hdfs_client = hdfs_client
        self.pretrained = model_addr['MattingNet']
        self.pspnet_pretrained = model_addr['PSPNet50']
        self.matting_net = MattingNet(hdfs_client=hdfs_client,
                                      in_plane=4, is_pretrained=False)
        self.pspnet = PSPNet50(2, frozen=False, pretrained=False,
                               use_aux=False, hdfs_client=hdfs_client)
        for p in self.parameters():
            p.requires_grad = False

        self.conv1 = nn.Conv2d(in_plane, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.weight_init()

    def forward(self, input, gt_comp=None, fgs=None, bgs=None):
        # with torch.no_grad():
        mask = self.pspnet(input)
        _, mask = torch.max(F.softmax(mask, 1), 1)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 1)

        alpha, _ = self.matting_net(torch.cat([input, mask], 1))

        x = torch.cat([input, mask, alpha], 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x + mask

        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)

        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        if not self.training:
            return x, pred_comp, mask, alpha
        return x, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

    def weight_init(self):
        # if self.hdfs_client is None:
        #     if not os.path.exists(self.pretrained):
        #         raise RuntimeError('Please ensure {} exists.'.format(
        #             self.pretrained))
        #     checkpoint = torch.load(self.pretrained)
        # else:
        #     model_tmp_path = '/tmp/resnet50.pth'
        #     self.hdfs_client.copy_to_local(self.pretrained,
        #                                    model_tmp_path)
        #     checkpoint = torch.load(model_tmp_path)
        #
        # try:
        #     self.matting_net.load_state_dict(checkpoint['state_dict'])
        # except Exception as e:
        #     from collections import OrderedDict
        #     mdict = OrderedDict()
        #     for k, v in checkpoint['state_dict'].items():
        #         assert (k.startswith('module.'))
        #         nk = k[len('module.'):]
        #         mdict[nk] = v
        #     self.matting_net.load_state_dict(mdict)
        self.load_pretained(self.matting_net, self.pretrained)
        self.load_pretained(self.pspnet, self.pspnet_pretrained)

        self.conv1.apply(layer_weights_init)
        self.conv2.apply(layer_weights_init)
        self.conv3.apply(layer_weights_init)
        self.conv4.apply(layer_weights_init)

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

    def train(self, mode=True):
        super(MattingRefining, self).train(mode)
        for m in self.matting_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.pspnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class Matting(nn.Module):
    def __init__(self, hdfs_client=None):
        super(Matting, self).__init__()
        self.hdfs_client = hdfs_client
        self.pspnet_pretrained = model_addr['PSPNet50']
        self.pspnet = PSPNet50(2, frozen=False, pretrained=False,
                               use_aux=False, hdfs_client=hdfs_client)
        for p in self.parameters():
            p.requires_grad = False
        self.matting_net = MattingNet(hdfs_client=hdfs_client,
                                      in_plane=3, is_pretrained=True)

        self.weight_init()

    def forward(self, input, gt_comp=None, fgs=None, bgs=None):
        mask = self.pspnet(input)
        _, mask = torch.max(F.softmax(mask, 1), 1)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 1)
        mask = mask.repeat(1, 3, 1, 1)

        input = input * mask
        x, _ = self.matting_net(input)

        x.masked_fill_(x > 1, 1)
        x.masked_fill_(x < 0, 0)

        pred_comp = self._composite4(fgs, bgs, x, trimaps=None)
        if not self.training:
            return x, pred_comp, mask
        return x, pred_comp

    def _composite4(self, fgs, bgs, alphas, trimaps=None):
        comp = None
        if fgs is not None and bgs is not None:
            comp = fgs * alphas + (1 - alphas) * bgs
        return comp

    def weight_init(self):
        self.load_pretained(self.pspnet, self.pspnet_pretrained)

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
        print("=> loaded checkpoint '{}'".format(pretrained))

    def train(self, mode=True):
        super(Matting, self).train(mode)
        for m in self.pspnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

'''
    multi-head:
        MattingETE - backbone-->mask_head-
                             \            \
                              -------------> alpha_head                
        MattingETEAlpha - backbone-->alpha_head
        AdaTimapDIM - backbone->trimap_head->raw_alpha_head->fusion
'''

class MattingETE(nn.Module):
    def __init__(self, hdfs_client=None):
        super(MattingETE, self).__init__()
        self.hdfs_client = hdfs_client
        # self.base = Res50BasePytorch(
        #     use_own_bn=True, strides=[1, 2, 1, 1],
        #     layers=[3, 4, 6, 3], dilations=[1, 1, 2, 4],
        #     pretrained=model_addr['res50']
        # )
        # encoder_planes = [64, 256, 512, 1024, 2048]

        self.base = build_efficient_net(
            'efficientnet-b3', pretrained=model_addr['efficientnet-b3'],
            hdfs_client=hdfs_client)
        encoder_planes = [24, 32, 48, 96, 136, 232, 384]
        encoder_planes = encoder_planes[::-1]

        self.head_mask = nn.Sequential(
            _PyramidPoolingModule(
                encoder_planes[0], encoder_planes[0]//4, (1, 2, 3, 6)),
            nn.Conv2d(
                encoder_planes[0]*2, encoder_planes[0]//4,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_planes[0]//4, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(encoder_planes[0]//4, 3, kernel_size=1),
        )

        self.deconv = []
        for idx, en_ch in enumerate(encoder_planes[:-1]):
            self.deconv.append(nn.Sequential(
                nn.Conv2d(en_ch, encoder_planes[idx+1], kernel_size=5, padding=2),
                nn.BatchNorm2d(encoder_planes[idx+1]),
                nn.ReLU(inplace=True)))
        self.deconv.append(nn.Sequential(
            nn.Conv2d(encoder_planes[-1], encoder_planes[-1], kernel_size=5, padding=2),
            nn.BatchNorm2d(encoder_planes[-1]),
            nn.ReLU(inplace=True)))
        self.deconv = nn.ModuleList(self.deconv)
        self.alpha_output = nn.Conv2d(encoder_planes[-1], 1, kernel_size=5, padding=2)
        # self.head_alpha = nn.Sequential(
        #     _PyramidPoolingModule(2048, 512, (1, 2, 3, 6)),
        #     nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512, momentum=.95),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Conv2d(512, 1, kernel_size=1),
        # )

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in MattingETE: {}".format(num_params))
        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)

        output_mask = self.head_mask(encoder[-1])
        # output_alpha = self.head_alpha(encoder[-1])
        output_alpha = torch.cat([encoder[-1].detach(), output_mask.detach()], 1)
        for idx, deconv_ in enumerate(self.deconv[:-1]):
            output_alpha = deconv_(output_alpha)
            output_alpha = F.upsample(
                output_alpha, encoder[-idx-2].size()[2:], mode='bilinear')
        output_alpha = self.deconv[-1](output_alpha)
        output_alpha = self.alpha_output(output_alpha)

        output_mask = F.upsample(output_mask, input.size()[2:], mode='bilinear')
        output_alpha = F.upsample(output_alpha, input.size()[2:], mode='bilinear')
        output_mask = F.softmax(output_mask, dim=1)
        # output_alpha = F.softmax(output_alpha, dim=1)
        output_alpha.masked_fill_(output_alpha > 1, 1)
        output_alpha.masked_fill_(output_alpha < 0, 0)

        return output_alpha, output_mask

    def init_weight(self):
        self.head_mask.apply(layer_weights_init)
        # self.head_alpha.apply(layer_weights_init)
        self.deconv.apply(layer_weights_init)
        self.alpha_output.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.head_mask.parameters()})
        # lr_list.append({'params': self.head_alpha.parameters()})
        lr_list.append({'params': self.deconv.parameters()})
        lr_list.append({'params': self.alpha_output.parameters()})
        return lr_list

    def train(self, mode=True):
        super(MattingETE, self).train(mode)

class MattingETEAlpha(nn.Module):
    def __init__(self, hdfs_client=None):
        super(MattingETEAlpha, self).__init__()
        self.hdfs_client = hdfs_client
        self.base = Res50BasePytorch(
            use_own_bn=True, strides=[1, 2, 1, 1],
            layers=[3, 4, 6, 3], dilations=[1, 1, 2, 4],
            pretrained=model_addr['res50']
        )

        self.head_alpha = nn.Sequential(
            _PyramidPoolingModule(2048, 512, (1, 2, 3, 6)),
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 2, kernel_size=1),
        )

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in MattingETEAlpha: {}".format(num_params))
        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)

        output_alpha = self.head_alpha(encoder[-1])

        output_alpha = F.upsample(output_alpha, input.size()[2:], mode='bilinear')
        output_alpha = F.softmax(output_alpha, dim=1)

        return None, output_alpha

    def init_weight(self):
        self.head_alpha.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.head_alpha.parameters()})
        return lr_list

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out

class AdaTimapDIM(nn.Module):
    def __init__(self, hdfs_client=None):
        super(AdaTimapDIM, self).__init__()
        self.hdfs_client = hdfs_client
        self.base = Res50BasePytorch(
            use_own_bn=True, strides=[1, 2, 1, 1],
            layers=[3, 4, 6, 3], dilations=[1, 1, 2, 4],
            pretrained=model_addr['res50']
        )
        encoder_planes = [64, 256, 512, 1024, 2048]
        encoder_planes = encoder_planes[::-1]

        self.head_mask = nn.Sequential(
            _PyramidPoolingModule(
                encoder_planes[0], encoder_planes[0] // 4, (1, 2, 3, 6)),
            nn.Conv2d(
                encoder_planes[0] * 2, encoder_planes[0] // 4,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_planes[0] // 4, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(encoder_planes[0] // 4, 3, kernel_size=1),
        )
        checkpoint = torch.load(model_addr['MattingETE'])
        try:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint['state_dict'][k]
            self.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint['state_dict'][nk]
            self.load_state_dict(new_dict)

        for p in self.parameters():
            p.requires_grad = False

        self.mnet = MattingVgg16(in_plane=6, frozen=False)
        checkpoint = torch.load(model_addr['MattingVgg16'])
        new_dict = OrderedDict()
        for k, _ in self.mnet.state_dict().items():
            if 'num_batches_tracked' in k:
                new_dict[k] = torch.zeros(1)
            else:
                new_dict[k] = checkpoint['mnet.'+k]
        self.mnet.load_state_dict(new_dict)

    def forward(self, input):
        encoder = self.base(input)
        output_mask = self.head_mask(encoder[-1])

        output_mask = F.upsample(
            output_mask, input.size()[2:], mode='bilinear')
        output_mask = F.softmax(output_mask, dim=1)
        in_mnet = torch.cat(
            (input, output_mask[:, 2:],
             output_mask[:, 1:2], output_mask[:, :1]), 1)

        raw_alpha_patch, _ = self.mnet(in_mnet)

        if self.training:
            return raw_alpha_patch, output_mask

        alpha = output_mask.argmax(1, keepdim=True).float()
        alpha.masked_scatter_(
            alpha == 2, raw_alpha_patch[alpha == 2])
        # fg = output_mask[:, 1, :, :]
        # unk = output_mask[:, 2, :, :]
        # alpha = unk * raw_alpha_patch + fg

        return alpha, raw_alpha_patch, output_mask

    def train(self, mode=True):
        super(AdaTimapDIM, self).train(mode)
        for m in self.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.head_mask.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

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