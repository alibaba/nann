import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import numpy as np
import random
sys.path.append(os.path.dirname(__file__))
from networks.affinity_boundary_loss import CBDLoss

__all__ = ['SaliencyLoss',
           'PFANetLoss',
           'VNetLoss',
           'WeightedMSELoss',
           'WeightedL1Loss',
           'MattingLoss',
           'WeightedMattingLoss',
           'CombinedMattingLoss',
           'FantasyLoss',
           'AFNetLoss',
           'SemanticSegmentationLoss',
        ]

class SaliencyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(SaliencyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if isinstance(output, list):
            if self.weight is not None:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.weight[0] * F.binary_cross_entropy(output[0][label!=self.ignore_index],
                                                               label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.weight[i] * F.binary_cross_entropy(x[label!=self.ignore_index],
                                                                    label[label!=self.ignore_index])
            else:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = F.binary_cross_entropy(output[0][label!=self.ignore_index],
                                              label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += F.binary_cross_entropy(x[label!=self.ignore_index],
                                                   label[label!=self.ignore_index])
            return loss, main_loss
        else:
            label = F.upsample(label, size=output.size()[2:])
            loss = F.binary_cross_entropy(output[label!=self.ignore_index],
                                          label[label!=self.ignore_index])
            main_loss = loss
        return loss, main_loss

class PFANetLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(PFANetLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_loss = SaliencyLoss(weight, ignore_index)
        self.laplace_loss = LaplaceLoss(weight, ignore_index)

    def forward(self, output, label):
        loss_bce, main_loss_bce = self.bce_loss(output, label)
        loss_lap, main_loss_lap = self.laplace_loss(output, label)

        loss = 0.7 * loss_bce + 0.3 * loss_lap
        main_loss = 0.7 * main_loss_bce + 0.3 * main_loss_lap
        return loss, main_loss

class AFNetLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(AFNetLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_loss = SaliencyLoss(weight, ignore_index)
        self.bel_loss = BEL([1,0.8,0,0,0,0,0,0],
                            ignore_index,
                            kernels_size=[5,5,3,3,3,3,3])

    def forward(self, output, label):
        loss_bce, main_loss_bce = self.bce_loss(output, label)
        loss_lap, main_loss_lap = self.bel_loss(output, label)

        loss = loss_bce + 0.1 * loss_lap
        main_loss = main_loss_bce + 0.1 * main_loss_lap
        return loss, main_loss

class VNetLoss(nn.Module):
    def __init__(self, weight=None):
        super(VNetLoss, self).__init__()
        self.bce_loss = SaliencyLoss(weight)
        self.alpha =1.0

    def forward(self, outputs, label):
        if len(outputs) > 2:
            pred, img_encoder = outputs[0], outputs[1]
            label_encoder = outputs[2]
            loss, main_loss = self.bce_loss(pred, label)

            mean_img = img_encoder.mean()
            std_img = img_encoder.std()
            mean_label = label_encoder.mean()
            std_label = label_encoder.std()
            loss += self.alpha*((mean_img-mean_label)**2+(std_img-std_label)**2)
            return loss, main_loss
        else:
            pred, img_encoder = outputs[0], outputs[1]
            loss, main_loss = self.bce_loss(pred, label)
            return loss, main_loss

class L1(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(L1, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if isinstance(output, list):
            if self.weight is not None:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.weight[0] * F.l1_loss(output[0][label!=self.ignore_index],
                                                  label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.weight[i] * F.l1_loss(x[label!=self.ignore_index],
                                                       label[label!=self.ignore_index])
            else:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = F.l1_loss(output[0][label!=self.ignore_index],
                                 label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += F.l1_loss(x[label!=self.ignore_index],
                                      label[label!=self.ignore_index])
            return loss, main_loss
        else:
            label = F.upsample(label, size=output.size()[2:])
            loss = F.l1_loss(output[label!=self.ignore_index],
                             label[label!=self.ignore_index])
            main_loss = loss
        return loss, main_loss

class LaplaceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(LaplaceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.laplace_conv = nn.Conv2d(1, 1, 3, 1, bias=False)
        self.laplace_conv.weight.data = self.laplace_kernel(1, 1)
        for p in self.laplace_conv.parameters():
            p.requires_grad = False

    def laplace_kernel(self, in_channels, out_channels, kernel_size=3):
        # factor = (kernel_size + 1) // 2
        # if kernel_size % 2 == 1:
        #     center = factor - 1
        # else:
        #     center = factor - 0.5
        # og = np.ogrid[:kernel_size, :kernel_size]
        # filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        for i in range(in_channels):
            for j in range(out_channels):
                weight[i, j, :, :] = filt
        return torch.from_numpy(weight)

    def laplace(self, input):
        input = self.laplace_conv(input)
        # input = F.relu(F.tanh(input))
        # input = input.masked_fill_(input < 1e-8, 1e-8)
        # input = input.masked_fill_(input > 1 - 1e-8, 1 - 1e-8)
        # input = torch.log(input / (1 - input))

        return input

    def forward(self, output, label):
        if isinstance(output, list):
            if self.weight is not None:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.weight[0] *\
                       torch.mean(torch.sqrt((self.laplace(output[0])
                                              -self.laplace(label)) ** 2 + 1e-12))
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.weight[i] * \
                            torch.mean(torch.sqrt((self.laplace(x)
                                                   - self.laplace(label)) ** 2 + 1e-12))
            else:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = torch.mean(torch.sqrt((self.laplace(output[0])
                                              - self.laplace(label)) ** 2 + 1e-12))
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += torch.mean(torch.sqrt((self.laplace(x)
                                                   - self.laplace(label)) ** 2 + 1e-12))
            return loss, main_loss
        else:
            label = F.upsample(label, size=output.size()[2:])
            loss = torch.mean(torch.sqrt((self.laplace(output)
                                          - self.laplace(label)) ** 2 + 1e-12))
            main_loss = loss
        return loss, main_loss

class BEL(nn.Module):
    '''
    CVPR 2019:
        Attentive Feedback Network for Boundary-Aware Salient Object Detection
    Boundary Enhanced Loss
    '''
    def __init__(self, weight=None,
                 ignore_index=255,
                 kernels_size=[]):
        super(BEL, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kernels_size = kernels_size

    def contour(self, input, kernel_size):
        x = F.avg_pool2d(
            input,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2)
        return (input - x).abs()

    def euclidean(self, input, label, kernel_size):
        loss = torch.mean(torch.sqrt(
            (self.contour(input, kernel_size)
             -self.contour(label, kernel_size)) ** 2 + 1e-12))
        return loss

    def forward(self, output, label):
        if isinstance(output, list):
            if self.weight is not None:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.weight[0] * \
                       self.euclidean(output[0], label, self.kernels_size[0])

                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.weight[i+1] * \
                            self.euclidean(x, label, self.kernels_size[i+1])
            else:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.euclidean(output[0], label, self.kernels_size[0])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.euclidean(x, label, self.kernels_size[i+1])
            return loss, main_loss
        else:
            label = F.upsample(label, size=output.size()[2:])
            loss = torch.mean(torch.sqrt(
                (self.contour(output, self.kernels_size[0])
                 - self.contour(label, self.kernels_size[0])) ** 2 + 1e-12))
            main_loss = loss
        return loss, main_loss


'''
    SHM Matting Loss
'''

class WeightedMSELoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(WeightedMSELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, source, target, weight=None, act_num=None):
        elenum = None
        if act_num is None:
            elenum = 1
            for i in range(source.dim()):
                elenum *= source.size(i)
        else:
            assert(act_num.dim() == 2)
            elenum = torch.sum(act_num)
        if weight is None:
            return torch.sum((source - target) ** 2) / elenum, elenum
        else:
            return torch.sum(weight * ((source - target) ** 2)) / elenum, elenum

class WeightedL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(WeightedL1Loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, source, target, weight=None, act_num=None):
        elenum = None
        if act_num is None:
            elenum = 1
            for i in range(source.dim()):
                elenum *= source.size(i)
        else:
            assert(act_num.dim() == 2)
            elenum = torch.sum(act_num)
        if weight is None:
            return torch.mean(torch.sqrt((source - target) ** 2 + 1e-12)), elenum
        else:
            return torch.sum(torch.sqrt(weight * (source - target) ** 2 + 1e-12)) \
                    / elenum, elenum

class WeightedLaplaceL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(WeightedLaplaceL1Loss, self).__init__()
        self.ignore_index = ignore_index
        self.laplace_conv = nn.Conv2d(1, 1, 3, 1, bias=False)
        self.laplace_conv.weight.data = self.laplace_kernel(1, 1)
        for p in self.laplace_conv.parameters():
            p.requires_grad = False

    def forward(self, source, target, weight=None, act_num=None):
        elenum = None
        if act_num is None:
            elenum = 1
        else:
            assert(act_num.dim() == 2)
            elenum = torch.sum(act_num)
        if weight is None:
            return torch.mean(torch.sqrt(
                (self.laplace_conv(source) - self.laplace_conv(target)) ** 2 + 1e-12)), elenum
        else:
            return torch.sum(torch.sqrt(
                weight * (self.laplace_conv(source) - self.laplace_conv(target)) ** 2 + 1e-12))\
                   / elenum, elenum

    def laplace_kernel(self, in_channels, out_channels, kernel_size=3):
        filt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        for i in range(in_channels):
            for j in range(out_channels):
                weight[i, j, :, :] = filt
        return torch.from_numpy(weight)

class MattingLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(MattingLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.weighted_mse_loss = WeightedMSELoss(ignore_index=ignore_index)
        self.weighted_l1_loss = WeightedL1Loss(ignore_index=ignore_index)
        self.weighted_laplace_l1_loss = WeightedLaplaceL1Loss(ignore_index=ignore_index)

    def forward(self, pred_alphas, gt_alphas, pred_comps, gt_comps,
                weight=None, act_num=None):
        alpha_loss, _ = self.weighted_l1_loss(pred_alphas, gt_alphas,
                                              weight=weight, act_num=act_num)
        comp_loss, _ = self.weighted_l1_loss(pred_comps, gt_comps,
                                             weight=weight, act_num=act_num)
        loss = (alpha_loss + comp_loss) * 0.5
        return loss, alpha_loss, comp_loss

class WeightedMattingLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(WeightedMattingLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.weighted_mse_loss = WeightedMSELoss(ignore_index=ignore_index)
        self.weighted_l1_loss = WeightedL1Loss(ignore_index=ignore_index)
        self.alpha_gm = GMWeight()
        self.comp_gm = GMWeight()

    def forward(self, pred_alphas, gt_alphas, pred_comps, gt_comps):
        alpha_loss, _ = self.weighted_l1_loss(
            pred_alphas, gt_alphas,
            weight=self.alpha_gm(pred_alphas, gt_alphas))
        comp_loss, _ = self.weighted_l1_loss(
            pred_comps, gt_comps,
            weight=self.comp_gm(pred_comps, gt_comps))
        loss = (alpha_loss + comp_loss) * 0.5
        return loss, alpha_loss, comp_loss

'''
    End-to-End Matting Loss
'''
class GMWeight(nn.Module):
    '''
        Gradient Harmonizing Loss
        AAAI 2019 - Gradient Harmonized Single-stage Detector
    '''
    def __init__(self, bins=10, momentum=0.1):
        super(GMWeight, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        self.acc_sum = [0.0 for _ in range(bins)]

    def forward(self, pred, target):
        target = target.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred - target)

        tot = target.shape[-2] * target.shape[-1]
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        return weights

class CombinedMattingLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(CombinedMattingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha_gm = GMWeight()
        self.mask_gm = GMWeight()

    def forward(self, pred_alphas, pred_mask, gt, flag):
        alpha_flag = (1 - flag).float()
        mask_flag = flag.float()
        alpha_weight = self.alpha_gm(pred_alphas, gt)
        mask_weight = self.alpha_gm(pred_mask, gt)
        alpha_loss = F.binary_cross_entropy(
            pred_alphas[:, :1] * alpha_flag, gt * alpha_flag,
            weight=alpha_weight)
        mask_loss = F.nll_loss(
            (pred_mask.log() * mask_flag), (gt * mask_flag).long().squeeze(1),
            weight=mask_weight)
        loss = alpha_loss + mask_loss
        return loss

class FantasyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(FantasyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_alpha, pred_mask, gt, flag=None):
        alpha_flag = (1 - flag).float()
        alpha_flag = (alpha_flag > 0)

        # mask head
        pred = pred_mask[:, :-1] + pred_mask[:, -1:]
        unk_pred = pred_mask[:, -1:]
        pred = torch.cat((pred, unk_pred), 1)
        ce_loss = F.nll_loss(
            pred.log(), gt.long().squeeze(1))
        penalty_loss = - 0.1 * (1 - unk_pred).log().mean()
        mask_loss = ce_loss + penalty_loss

        # alpha head
        # t = random.uniform(0.3, 0.5)
        # unk_mask = (alpha_flag * (unk_pred > t)).detach()
        mask = pred_mask.detach().argmax(1, keepdim=True)
        unk_mask = (alpha_flag * (mask == 2))
        if unk_mask.sum() == 0:
            l1_loss = unk_mask.sum().float()
        else:
            l1_loss = torch.sqrt(
                (pred_alpha - gt)[unk_mask] ** 2 + 1e-12).mean()
        fg_pred = pred_mask[:, 1:2]
        bg_pred = pred_mask[:, :1]
        # fg_mask = (alpha_flag * (unk_pred <= t) * (fg_pred > bg_pred)).detach()
        # bg_mask = (alpha_flag * (unk_pred <= t) * (fg_pred <= bg_pred)).detach()
        fg_mask = (alpha_flag * (mask == 1))
        bg_mask = (alpha_flag * (mask == 0))
        if fg_mask.sum() == 0:
            fg_penalty_loss = fg_mask.sum().float()
        else:
            fg_penalty_loss = 0.1 * ((1 - gt) * fg_pred)[fg_mask].mean()
        if bg_mask.sum() == 0:
            bg_penalty_loss = bg_mask.sum().float()
        else:
            bg_penalty_loss = 0.1 * (gt * bg_pred)[bg_mask].mean()
        alpha_loss = l1_loss + fg_penalty_loss + bg_penalty_loss

        # loss = mask_loss + alpha_loss
        loss = mask_loss
        # loss = l1_loss 
        return loss, ce_loss, penalty_loss,\
               l1_loss, fg_penalty_loss, bg_penalty_loss


'''
    Semantic Segmentation Loss
'''
class SemanticSegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SemanticSegmentationLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cbd = CBDLoss(512, radius=5, ignore=21)

    def forward(self, pred_mask, gt, pos_label, neg_label):
        # mask head
        pred = pred_mask[:, :-1] + pred_mask[:, -1:]
        unk_pred = pred_mask[:, -1:]
        pred = torch.cat((pred, unk_pred), 1)
        ce_loss = F.nll_loss(
            pred.log(), gt.long().squeeze(1))

        org_ce_loss = F.nll_loss(
            pred_mask[:, :-1].log(), gt.long().squeeze(1),
            ignore_index=21)
        cbd_loss = self.cbd(pred_mask[:, -1:], pos_label, neg_label)
        penalty_loss = - 0.1 * (1 - unk_pred).log().mean()
        mask_loss = 0.7 * ce_loss + 0.3*org_ce_loss + penalty_loss + cbd_loss

        loss = mask_loss
        return loss, ce_loss, penalty_loss, org_ce_loss, cbd_loss


if __name__ == '__main__':
    # def laplace_kernel(in_channels, out_channels, kernel_size=3):
    #     filt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    #     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    #     for i in range(in_channels):
    #         for j in range(out_channels):
    #             weight[i, j, :, :] = filt
    #     return torch.from_numpy(weight)
    #
    # import cv2
    # img = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/5025666458_576b974455_o_g0_24hHn3c_m_5b07a1544894462f85abd139270921ab.jpg_gt.png',
    #                  cv2.IMREAD_GRAYSCALE)
    # img = np.expand_dims(img, 2)
    # img = torch.from_numpy(img.transpose(2, 0, 1).copy())
    # img = img.float().div(255)
    # img = img.unsqueeze(0)
    #
    # img = img.repeat(1, 3, 1, 1)
    # print(img.size())
    # img = img[:, 0]
    # print(img.size())
    # img = img.unsqueeze(1)
    #
    # # laplace_conv = nn.Conv2d(1, 1, 3, 1, bias=False)
    # # laplace_conv.weight.data = laplace_kernel(1, 1)
    # # img = laplace_conv(img)
    # output = np.array(img.data[0,0])
    # print(output.shape)
    # cv2.imshow('1', output)
    # cv2.waitKey()

    np.random.seed(0)
    torch.manual_seed(0)
    cet = CombinedMattingLoss()
    alpha = torch.randn(2, 2, 64, 64)
    alpha = F.softmax(alpha, dim=1)
    mask = torch.randn((2, 2, 64, 64))
    mask = F.softmax(mask, dim=1)
    gt = torch.zeros((2, 1, 64, 64))
    flag = np.expand_dims(np.array([0,1]), 1)
    flag = np.expand_dims(flag, 2)
    flag = np.expand_dims(flag, 3)
    flag = torch.from_numpy(flag)
    print(flag.shape)
    loss = cet(alpha, mask, gt, flag)
