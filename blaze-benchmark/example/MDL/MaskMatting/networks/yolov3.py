import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

from cqutils.yolo_utils import bboxes_iou
from networks.Layers import layer_weights_init
from config import *

from collections import defaultdict

__all__ = ['YOLOv3',
        ]

class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8] # fixed
        self.anchors = config_model['ANCHORS']
        self.anch_mask = config_model['ANCH_MASK'][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for  , obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if labels is None:  # not training
            pred[..., :4] *= self.stride
            return pred.view(batchsize, -1, n_ch).data

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,
                             size_average=False)  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2]) / batchsize
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / (2*batchsize)
        loss_obj = self.bce_loss(output[..., 4], target[..., 4]) / batchsize
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:]) / batchsize
        loss_l2 = self.l2_loss(output, target) / batchsize

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2

def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def create_yolov3_modules(config_model, ignore_thre):
    """
    Build yolov3 layer modules.
    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """

    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    mlist.append(resblock(ch=64))
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    mlist.append(resblock(ch=128, nblocks=2))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
    mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
    mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
    mlist.append(resblock(ch=1024, nblocks=4))

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    mlist.append(
         YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(
        YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    mlist.append(resblock(ch=256, nblocks=2, shortcut=False))
    mlist.append(
         YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre))

    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, config_model=None, ignore_thre=0.7,
                 pretrained=None, hdfs_client=None):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()

        self.pretrained = model_addr['DarkNet54'] if pretrained is not None else None
        self.hdfs_client = hdfs_client
        if config_model is None:
            config_model = {'TYPE': 'YOLOv3',
                            'BACKBONE': 'darknet53',
                            'ANCHORS': [[10, 13], [16, 30], [33, 23],
                                        [30, 61], [62, 45], [59, 119],
                                        [116, 90], [156, 198], [373, 326]],
                            'ANCH_MASK': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                            'N_CLASSES': 2,
                            }
        self.module_list = create_yolov3_modules(config_model, ignore_thre)
        if self.pretrained is not None:
            self.init_weight()

    def forward(self, x, targets=None):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`
        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output), self.loss_dict['xy'], \
                   self.loss_dict['wh'], self.loss_dict['conf'], \
                   self.loss_dict['cls'], self.loss_dict['l2']
        else:
            return torch.cat(output, 1)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.module_list.parameters()})

        return lr_list

    def init_weight(self):
        # load Darknet
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            darknet_weight = self.pretrained
        else:
            model_tmp_path = '/tmp/tmp.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            darknet_weight = model_tmp_path

        with open(darknet_weight, "rb") as fp:
            # skip the header
            header = np.fromfile(fp, dtype=np.int32, count=5)  # not used
            # read weights
            weights = np.fromfile(fp, dtype=np.float32)
            offset = 0
            initflag = False  # whole yolo weights : False, darknet weights : True

            for m in self.module_list:

                if m._get_name() == 'Sequential':
                    # normal conv block
                    offset, weights = self.parse_conv_block(m, weights, offset, initflag)

                elif m._get_name() == 'resblock':
                    # residual block
                    for modu in m._modules['module_list']:
                        for blk in modu:
                            offset, weights = self.parse_conv_block(blk, weights, offset, initflag)

                elif m._get_name() == 'YOLOLayer':
                    # YOLO Layer (one conv with bias) Initialization
                    offset, weights = self.parse_yolo_block(m, weights, offset, initflag)

                initflag = (offset >= len(weights))
                if initflag:
                    break

        print('==> loaded pretrained {}'.format(self.pretrained))

        self.module_list[11:].apply(layer_weights_init)

    def parse_conv_block(self, m, weights, offset, initflag):
        """
        Initialization of conv layers with batchnorm
        Args:
            m (Sequential): sequence of layers
            weights (numpy.ndarray): pretrained weights data
            offset (int): current position in the weights file
            initflag (bool): if True, the layers are not covered by the weights file. \
                They are initialized using darknet-style initialization.
        Returns:
            offset (int): current position in the weights file
            weights (numpy.ndarray): pretrained weights data
        """
        conv_model = m[0]
        bn_model = m[1]
        param_length = m[1].bias.numel()

        # batchnorm
        for pname in ['bias', 'weight', 'running_mean', 'running_var']:
            layerparam = getattr(bn_model, pname)

            if initflag:  # yolo initialization - scale to one, bias to zero
                if pname == 'weight':
                    weights = np.append(weights, np.ones(param_length))
                else:
                    weights = np.append(weights, np.zeros(param_length))

            param = torch.from_numpy(weights[offset:offset + param_length]).view_as(layerparam)
            layerparam.data.copy_(param)
            offset += param_length

        param_length = conv_model.weight.numel()

        # conv
        if initflag:  # yolo initialization
            n, c, k, _ = conv_model.weight.shape
            scale = np.sqrt(2 / (k * k * c))
            weights = np.append(weights, scale * np.random.normal(size=param_length))

        param = torch.from_numpy(
            weights[offset:offset + param_length]).view_as(conv_model.weight)
        conv_model.weight.data.copy_(param)
        offset += param_length

        return offset, weights

    def parse_yolo_block(self, m, weights, offset, initflag):
        """
        YOLO Layer (one conv with bias) Initialization
        Args:
            m (Sequential): sequence of layers
            weights (numpy.ndarray): pretrained weights data
            offset (int): current position in the weights file
            initflag (bool): if True, the layers are not covered by the weights file. \
                They are initialized using darknet-style initialization.
        Returns:
            offset (int): current position in the weights file
            weights (numpy.ndarray): pretrained weights data
        """
        conv_model = m._modules['conv']
        param_length = conv_model.bias.numel()

        if initflag:  # yolo initialization - bias to zero
            weights = np.append(weights, np.zeros(param_length))

        param = torch.from_numpy(
            weights[offset:offset + param_length]).view_as(conv_model.bias)
        conv_model.bias.data.copy_(param)
        offset += param_length

        param_length = conv_model.weight.numel()

        if initflag:  # yolo initialization
            n, c, k, _ = conv_model.weight.shape
            scale = np.sqrt(2 / (k * k * c))
            weights = np.append(weights, scale * np.random.normal(size=param_length))

        param = torch.from_numpy(
            weights[offset:offset + param_length]).view_as(conv_model.weight)
        conv_model.weight.data.copy_(param)
        offset += param_length

        return offset, weights

if __name__ == '__main__':
    net = YOLOv3(pretrained='')