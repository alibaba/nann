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
cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from networks.backbone import Res50BasePytorch
from cqutils import print_network
from config import *

__all__ = ['PRNet',
        ]

Normal_Type = 'batch'
Momentum = 0.1
def _normal_layer(in_ch, use_own_bn=False):
    if use_own_bn:
        return BatchNormCaffe(in_ch)
    elif Normal_Type == 'group':
        return nn.GroupNorm(in_ch, in_ch)
    elif Normal_Type == 'spectral':
        return PlaceHolderLayer()
    elif Normal_Type == 'dist_batch':
        return DistributedBatchNorm2d(in_ch, momentum=Momentum)
    else:
        return nn.BatchNorm2d(in_ch, momentum=Momentum)


class PRNet(nn.Module):
    def __init__(self, hdfs_client=None, load_pretrained=True):
        super(PRNet, self).__init__()
        self.hdfs_client = hdfs_client
        self.pretrained = None
        if load_pretrained:
            self.pretrained = model_addr['res101']
        self.base = PRBase(use_own_bn=True,
                           strides=[1, 2, 2, 2],
                           dilations=[1, 1, 1, 1],
                           layers=[3, 4, 23, 3],
                           pretrained=self.pretrained)
        branch_planes = [256, 256, 256, 256, 256]
        self.transition = Transition(
            inplanes=[64, 256, 512, 1024, 2048],
            outplanes=branch_planes)
        self.prm1 = ProgressiveRefineModule(
            inplanes=branch_planes,
            outplanes=branch_planes,
            size_times=[1, 2, 4, 8, 16])
        self.prm2 = ProgressiveRefineModule(
            inplanes=branch_planes,
            outplanes=branch_planes,
            size_times=[1, 2, 4, 8, 16],
            multi_branch=True)
        self.transition2 = Transition(
            inplanes=branch_planes,
            outplanes=[32, 32, 32, 32, 32],
            is_scale=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(160, 160, 3, padding=1),
            _normal_layer(160),
            # nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, 3, padding=1),
            _normal_layer(160),
            # nn.ReLU(inplace=True)
        )
        # self.output_layer = OutputLayer(
        #     inplanes=[256, 256, 256, 256, 256])
        self.output_layer = OutputLayer(
            inplanes=[160]+branch_planes)

        if load_pretrained:
            self.init_weight()

    def forward(self, input):
        encoders = self.base(input)
        outputs, _ = self.transition(encoders)
        outputs = self.prm1(outputs)
        outputs = self.prm2(outputs)

        fuse_output, outputs = self.transition2(outputs, input.size()[2:])
        fuse_output = torch.cat(fuse_output, 1)
        fuse_output = self.conv(fuse_output)
        outputs = [fuse_output] + outputs

        outputs = self.output_layer(outputs)
        # outputs = outputs[::-1]

        return outputs[0]

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(self.pretrained):
                raise RuntimeError('Please ensure {} exists.'.format(
                    self.pretrained))
            self.base.load_state_dict(torch.load(self.pretrained), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(self.pretrained,
                                           model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

        self.transition.apply(layer_weights_init)
        self.prm1.apply(layer_weights_init)
        self.prm2.apply(layer_weights_init)
        self.transition2.apply(layer_weights_init)
        self.conv.apply(layer_weights_init)
        self.output_layer.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.transition.parameters()})
        lr_list.append({'params': self.prm1.parameters()})
        lr_list.append({'params': self.prm2.parameters()})
        lr_list.append({'params': self.transition2.parameters()})
        lr_list.append({'params': self.conv.parameters()})
        lr_list.append({'params': self.output_layer.parameters()})

        return lr_list


class PRBase(Res50BasePytorch):
    def __init__(self, *args, **kwargs):
        super(PRBase, self).__init__(*args, **kwargs)
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in PRBase: {}".format(num_params))
    def forward(self, x):
        encoder = []

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder


class Plain(nn.Module):
    def __init__(self, inplanes=[], outplanes=[], size_times=[],
                 mode='up', multi_branch=True):
        super(Plain, self).__init__()
        self.branches = len(inplanes)
        self.mode = mode
        self.multi_branch = multi_branch
        self.branch_merges = []
        for in_ch, out_ch in zip(inplanes, outplanes):
            self.branch_merges.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        self.branch_merges = nn.ModuleList(self.branch_merges)

    def forward(self, inputs):
        outputs = []
        if self.multi_branch:
            for i in range(self.branches):
                outputs.append(self.branch_merges[i](inputs[i]))
        else:
            outputs.append(self.branch_merges[0](inputs))
        return outputs


class PathAggregation(nn.Module):
    def __init__(self, inplanes=[], outplanes=[], size_times=[],
                 mode='up', multi_branch=True):
        super(PathAggregation, self).__init__()
        self.branches = len(inplanes)
        self.mode = mode
        self.multi_branch = multi_branch

        self.branch_up = []
        for in_ch, out_ch in zip(inplanes, outplanes):
            self.branch_up.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=False)
                )
            )
        self.branch_up = nn.ModuleList(self.branch_up)

        self.branch_down = []
        for in_ch, out_ch in zip(outplanes, outplanes):
            self.branch_down.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=False)
                )
            )
        self.branch_down = nn.ModuleList(self.branch_down)

    def forward(self, inputs):

        outputs_mid = []
        for i in range(self.branches):
            outputs_mid.append(self.branch_up[i](inputs[i]))
        for i in range(self.branches - 1):
            outputs_mid[i] += F.upsample(
                outputs_mid[i + 1],
                outputs_mid[i].shape[2:],
                mode='bilinear',
                align_corners=True)

        outputs = []
        for i in range(self.branches):
            outputs.append(self.branch_down[i](outputs_mid[i]))
        for i in range(1, self.branches):
            outputs[i] += F.upsample(
                outputs[i - 1],
                outputs[i].shape[2:],
                mode='bilinear',
                align_corners=True)

        return outputs


class ProgressiveRefineModule(nn.Module):
    def __init__(self, inplanes=[], outplanes=[], size_times=[],
                 mode='up', multi_branch=True):
        super(ProgressiveRefineModule, self).__init__()
        self.branches = len(inplanes)
        self.mode = mode
        self.multi_branch = multi_branch
        self.branch_merges = []
        if self.mode == 'down':
            if not multi_branch:
                self.branch_merges.append(
                    BranchConcat(
                        inplanes, outplanes[-1],
                        size_times=size_times,
                        mode=self.mode))
            else:
                for i in range(self.branches):
                    self.branch_merges.append(
                        BranchConcat(
                            inplanes[:i+1], outplanes[i],
                            size_times=size_times[:i+1],
                            mode=self.mode))
        else:
            if not multi_branch:
                self.branch_merges.append(
                    BranchConcat(
                        inplanes, outplanes[0],
                        size_times=size_times,
                        mode=self.mode))
            else:
                for i in range(self.branches):
                    self.branch_merges.append(
                        BranchConcat(
                            inplanes[i:], outplanes[i],
                            size_times=size_times[i:],
                            mode=self.mode))
        self.branch_merges = nn.ModuleList(self.branch_merges)

    def forward(self, inputs):
        outputs = []
        if self.multi_branch:
            if self.mode == 'down':
                for i in range(self.branches):
                    outputs.append(self.branch_merges[i](inputs[:i+1]))
            else:
                for i in range(self.branches):
                    outputs.append(self.branch_merges[i](inputs[i:]))
        else:
            outputs.append(self.branch_merges[0](inputs))
        return outputs


class BranchMerge(nn.Module):
    def __init__(self, inplanes=[], outplanes=256, size_times=[],
                 mode='down'):
        super(BranchMerge, self).__init__()
        self.in_branches = len(inplanes)
        if mode == 'down':
            is_downsample = [item < size_times[-1] for item in size_times]
            is_upsample = [item > size_times[-1] for item in size_times]
        else:
            is_downsample = [item < size_times[0] for item in size_times]
            is_upsample = [item > size_times[0] for item in size_times]
        self.conv_list = []
        for i, ch in enumerate(inplanes):
            if is_downsample[i]:
                self.conv_list.append(
                    nn.Sequential(DownsampleModule(ch, ch, size_times[i],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1),
                                  nn.Conv2d(ch, outplanes,
                                            kernel_size=3,
                                            padding=1),
                                  _normal_layer(outplanes)))
            elif is_upsample[i]:
                self.conv_list.append(
                    nn.Sequential(UpsampleModule(ch, int(size_times[i]/size_times[0])),
                                  nn.Conv2d(ch, outplanes,
                                            kernel_size=3,
                                            padding=1),
                                  _normal_layer(outplanes)))
            else:
                self.conv_list.append(
                    nn.Sequential(nn.Conv2d(ch, outplanes,
                                            kernel_size=3,
                                            padding=1),
                                  _normal_layer(outplanes)))
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, inputs):
        output = F.relu(self.conv_list[0](inputs[0]))
        for input, branch_conv in zip(inputs[1:], self.conv_list[1:]):
            output += F.relu(branch_conv(input))
        return output


class BranchConcat(nn.Module):
    def __init__(self, inplanes=[], outplanes=256, size_times=[],
                 mode='down'):
        super(BranchConcat, self).__init__()
        self.in_branches = len(inplanes)
        self.mode = mode
        if mode == 'down':
            is_downsample = [item < size_times[-1] for item in size_times]
            is_upsample = [item > size_times[-1] for item in size_times]
        else:
            is_downsample = [item < size_times[0] for item in size_times]
            is_upsample = [item > size_times[0] for item in size_times]
        self.conv_list = []
        for i, ch in enumerate(inplanes):
            if is_downsample[i]:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(ch, outplanes,
                                  kernel_size=3,
                                  padding=1),
                        _normal_layer(outplanes),
                        nn.ReLU(),
                        DownsampleModule(outplanes, outplanes, size_times[i],
                                         kernel_size=3, stride=2, padding=1)
                    ))
            elif is_upsample[i]:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(ch, outplanes, kernel_size=3, padding=1),
                        _normal_layer(outplanes),
                        nn.ReLU(),))
                        # UpsampleModule(outplanes, int(size_times[i] / size_times[0])),))
            else:
                self.conv_list.append(
                    nn.Sequential(nn.Conv2d(ch, outplanes,
                                            kernel_size=3,
                                            padding=1),
                                  _normal_layer(outplanes),
                                  nn.ReLU()))
        self.conv_list = nn.ModuleList(self.conv_list)
        self.final_conv = nn.Conv2d(outplanes*len(inplanes), outplanes,
                                    kernel_size=1)
        self.final_bn = _normal_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if self.mode == 'down':
            size = inputs[-1].size()[2:]
        else:
            size = inputs[0].size()[2:]

        output = []
        for input, branch_conv in zip(inputs, self.conv_list):
            output.append(F.upsample(branch_conv(input), size,
                          mode='bilinear', align_corners=True))
        output = torch.cat(output, 1)
        output = self.final_bn(self.final_conv(output))
        if self.mode == 'down':
            output += inputs[-1]
        else:
            output += inputs[0]
        output = self.relu(output)
        return output


class Transition(nn.Module):
    def __init__(self, inplanes=[], outplanes=[],
                 is_scale=False):
        super(Transition, self).__init__()
        assert len(inplanes) == len(outplanes)
        self.is_scale = is_scale
        self.conv_list = []
        for in_ch, out_ch in zip(inplanes, outplanes):
            self.conv_list.append(
                nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1),
                              _normal_layer(out_ch))
            )
        self.conv_list = nn.ModuleList(self.conv_list)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, size=None):
        if self.is_scale:
            for idx in range(len(input)):
                input[idx] = F.upsample(input[idx], size,
                                        mode='bilinear',align_corners=True)
        upsample = input
        output = []
        for idx, x in enumerate(input):
            output.append(self.conv_list[idx](x))

        return output, upsample


class UpsampleModule(nn.Module):
    def __init__(self, in_ch, size, upsample_type='upsample'):
        super(UpsampleModule, self).__init__()
        self.upsample_type = upsample_type
        self.size = size
        if self.upsample_type == 'ConvTranspose':
            self.dec_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2,
                                               padding=1, bias=False, groups=in_ch)
        elif self.upsample_type == 'upsampleConv':
            self.dec_conv = nn.Conv2d(in_ch, in_ch,
                                      kernel_size=3,
                                      padding=1, bias=False)

    def forward(self, input):
        size = input.size()[2:]
        size = [item * self.size for item in size]
        if self.upsample_type == 'ConvTranspose':
            output = self.dec_conv(input)
        elif self.upsample_type == 'upsampleConv':
            output = F.upsample(input, size, mode='bilinear',
                                align_corners=True)
            output = self.dec_conv(output)
        else:
            output = F.upsample(input, size, mode='bilinear',
                                align_corners=True)

        return output


class DownsampleModule(nn.Module):
    def __init__(self, inplanes, outplanes,
                 size_times, kernel_size, dilation=1,
                 stride=1, padding=0, bias=True):
        super(DownsampleModule, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = _normal_layer(outplanes)

    def forward(self, input):
        output = F.relu(self.bn(self.conv(input)))
        return output


class AttentionModule(nn.Module):
    def __init__(self, inplanes=[], outplanes=[],
                 size_times=[]):
        super(AttentionModule, self).__init__()
        self.attention_list = []


    def forward(self, input):
        pass


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, attention):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class OutputLayer(nn.Module):
    def __init__(self, inplanes=[]):
        super(OutputLayer, self).__init__()
        self.conv_list = []
        for planes in inplanes:
            self.conv_list.append(nn.Conv2d(planes, 1, 1))
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(F.sigmoid(self.conv_list[i](input)))
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torchvision.models.resnet50()
    model = PRNet()
    # print([name for name, param in model.named_parameters()])
    model = model.to(device)
    input = torch.randn(1, 3, 128, 128)
    pred = model(input)
    print(pred[0].size())
