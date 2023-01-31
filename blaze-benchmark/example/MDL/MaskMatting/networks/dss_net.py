import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.dirname(__file__))
from cqutils import initialize_weights

from config import *

__all__ = ['DSS_VGG16',
        'DSSLoss',
        ]


# loss function: seven probability map --- 6 scale + 1 fuse
class DSSLoss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(DSSLoss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
        return loss





# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}


# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, scale = [], [], 1
    # print('cfg:{}'.format(cfg))
    for k, v in enumerate(cfg):
        # print('k:{}, v:{}'.format(k, v))
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        scale *= 2
    return vgg, feat_layers, concat_layers


# DSS network
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect):
        super(DSS, self).__init__()
        self.extract = [3, 8, 15, 22, 29]
        self.connect = connect
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, x, label=None):
        prob, back, y, num = list(), list(), list(), 0
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                y.append(self.feat[num](x))
                num += 1
        # side output
        y.append(self.feat[num](self.pool(x)))
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        # fusion map
        back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        # add sigmoid
        for i in back: prob.append(F.sigmoid(i))
        return prob


# build the whole network
def build_model():
    return DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])

def DSS_VGG16(distributed=False, hdfs_client=None):
    model = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'])
    model.apply(weights_init)
    print('Apply user defined weight init done.')
    if distributed:
        model_tmp_path = '/tmp/vgg16_backbone.pth'
        hdfs_client.copy_to_local(model_addr['vgg16_backbone'], model_tmp_path)
        model.base.load_state_dict(torch.load(model_tmp_path))
    else:
        model.base.load_state_dict(torch.load(model_addr['vgg16_backbone']))
    print('Load vgg16 backbone weight trained from ImageNet done.')
    return model
# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = build_model()
    img = torch.randn(1, 3, 64, 64)
    net = net.cuda()
    print('dss:{}'.format(net))
    img = Variable(img.cuda())
    out = net(img)
    k = [out[x] for x in [1, 2, 3, 6]]
    print(k[0])
    print(len(out))
