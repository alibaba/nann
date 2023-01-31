import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import torch.nn.functional as F

__all__ = ['AutoDeeplab',
        ]

'''
    operations
'''
OPS = {
  'none': lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=True),
      )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=True),
      )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=True)
    self.conv_1 = nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
    # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    # out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
    out = self.conv_1(x)
    out = self.bn(out)
    out = self.relu(out)
    return out

class FactorizedIncrease(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()

        self._in_channel = in_channel
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, paddings, dilations):
        # todo depthwise separable conv
        super(ASPP, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False, ),
                                    nn.BatchNorm2d(out_channels))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                    padding=paddings, dilation=dilations, bias=False, ),
                                      nn.BatchNorm2d(out_channels))
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False, ),
                                    nn.BatchNorm2d(out_channels))

        self.concate_conv = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)

        # image pool and upsample
        image_pool = F.avg_pool2d(x, kernel_size=x.size()[2:], stride=1)
        upsample = F.upsample(
            image_pool, size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = self.conv_p(upsample)

        # concate
        concate = torch.cat([conv11, conv33, upsample], dim=1)

        return self.concate_conv(concate)

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

'''
    NAS cell
'''

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):

        super(Cell, self).__init__()
        self.C_out = C
        if C_prev_prev != -1:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        if rate == 2:
            self.preprocess1 = FactorizedReduce(C_prev, C, affine=False)
        elif rate == 0:
            self.preprocess1 = FactorizedIncrease(C_prev, C)
        else:
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        if C_prev_prev != -1:
            for i in range(self._steps):
                for j in range(2+i):
                    stride = 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
        else:
            for i in range(self._steps):
                for j in range(1+i):
                    stride = 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)
        self.ReLUConvBN = ReLUConvBN(self._multiplier * self.C_out, self.C_out, 1, 1, 0)


    def forward(self, s0, s1, weights):
        if s0 is not None:
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        if s0 is not None:
            states = [s0, s1]
        else:
            states = [s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)


        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return  self.ReLUConvBN(concat_feature)

'''
    Global Network
'''

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers,
                 criterion, num_channel=20, multiplier=5,
                 step=5, cell=Cell):
        super(AutoDeeplab, self).__init__()
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._multiplier = multiplier
        self._num_channel = num_channel
        self._criterion = criterion
        self._initialize_alphas()
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        C_prev_prev = 64
        C_prev = 128
        for i in range(self._num_layers):
        # def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate): rate = 0 , 1, 2  reduce rate
            if i == 0:
                cell1 = cell(self._step, self._multiplier, -1, C_prev, self._num_channel, 1)
                cell2 = cell(self._step, self._multiplier, -1, C_prev, self._num_channel * 2, 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1_1 = cell(self._step, self._multiplier, C_prev, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, C_prev, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, -1, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 2, 1)

                cell3 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell3]
            elif i == 2:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 4, 1)

                cell4 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell4]
            elif i == 3:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell(self._step, self._multiplier, -1, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]
            else:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell(self._step, self._multiplier, self._num_channel * 8, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell(self._step, self._multiplier, self._num_channel * 8, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]
        self.aspp_4 = nn.Sequential(
            ASPP(self._num_channel, 256, 24, 24)
        )
        self.aspp_8 = nn.Sequential(
            ASPP(self._num_channel * 2, 256, 12, 12)
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._num_channel * 4, 256, 6, 6)
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._num_channel * 8, 256, 3, 3)
        )
        self.final_conv = nn.Conv2d(1024, num_classes, 1, stride=1, padding=0)

    def forward(self, x):
        level_2 = []
        level_4 = []
        level_8 = []
        level_16 = []
        level_32 = []
        # self._init_level_arr(x)
        temp = self.stem0(x)
        level_2.append(self.stem1(temp))
        level_4.append(self.stem2(level_2[-1]))
        count = 0
        weight_network = F.softmax(self.alphas_network, dim = -1)
        weight_cells = F.softmax(self.alphas_cell, dim=-1)
        for layer in range(self._num_layers):
            if layer == 0:
                level4_new = self.cells[count](None, level_4[-1], weight_cells)
                count += 1
                level8_new = self.cells[count](None, level_4[-1], weight_cells)
                count += 1
                level_4.append(level4_new * weight_network[layer][0][0])
                level_8.append(level8_new * weight_network[layer][0][1])
                # print((level_4[-2]).size(),(level_4[-1]).size())
            elif layer == 1:
                level4_new_1 = self.cells[count](level_4[-2], level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count](level_4[-2], level_8[-1], weight_cells)
                count += 1
                level4_new = weight_network[layer][0][0] * level4_new_1 + weight_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count](None, level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count](None, level_8[-1], weight_cells)
                count += 1
                level8_new = weight_network[layer][1][0] * level8_new_1 + weight_network[layer][1][1] * level8_new_2

                level16_new = self.cells[count](None, level_8[-1], weight_cells)
                level16_new = level16_new * weight_network[layer][1][2]
                count += 1


                level_4.append(level4_new)
                level_8.append(level8_new)
                level_16.append(level16_new)
            elif layer == 2:
                level4_new_1 = self.cells[count](level_4[-2], level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count](level_4[-2], level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count](level_8[-2], level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count](level_8[-2], level_8[-1], weight_cells)
                count += 1
                # print(level_8[-1].size(),level_16[-1].size())
                level8_new_3 = self.cells[count](level_8[-2], level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count](None, level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count](None, level_16[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2


                level32_new = self.cells[count](None, level_16[-1], weight_cells)
                level32_new = level32_new * self.alphas_network[layer][2][2]
                count += 1

                level_4.append(level4_new)
                level_8.append(level8_new)
                level_16.append(level16_new)
                level_32.append(level32_new)
            elif layer == 3:
                level4_new_1 = self.cells[count](level_4[-2], level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count](level_4[-2], level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count](level_8[-2], level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count](level_8[-2], level_8[-1], weight_cells)
                count += 1
                level8_new_3 = self.cells[count](level_8[-2], level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count](level_16[-2], level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count](level_16[-2], level_16[-1], weight_cells)
                count += 1
                level16_new_3 = self.cells[count](level_16[-2], level_32[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2 + self.alphas_network[layer][2][2] * level16_new_3


                level32_new_1 = self.cells[count](None, level_16[-1], weight_cells)
                count += 1
                level32_new_2 = self.cells[count](None, level_32[-1], weight_cells)
                count += 1
                level32_new = self.alphas_network[layer][3][0] * level32_new_1 + self.alphas_network[layer][3][1] * level32_new_2


                level_4.append(level4_new)
                level_8.append(level8_new)
                level_16.append(level16_new)
                level_32.append(level32_new)
            else:
                level4_new_1 = self.cells[count](level_4[-2], level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count](level_4[-2], level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count](level_8[-2], level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count](level_8[-2], level_8[-1], weight_cells)
                count += 1
                level8_new_3 = self.cells[count](level_8[-2], level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count](level_16[-2], level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count](level_16[-2], level_16[-1], weight_cells)
                count += 1
                level16_new_3 = self.cells[count](level_16[-2], level_32[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2 + self.alphas_network[layer][2][2] * level16_new_3

                level32_new_1 = self.cells[count](level_32[-2], level_16[-1], weight_cells)
                count += 1
                level32_new_2 = self.cells[count](level_32[-2], level_32[-1], weight_cells)
                count += 1
                level32_new = self.alphas_network[layer][3][0] * level32_new_1 + self.alphas_network[layer][3][1] * level32_new_2

                level_4.append(level4_new)
                level_8.append(level8_new)
                level_16.append(level16_new)
                level_32.append(level32_new)
        # print(level_4[-1].size(),level_8[-1].size(),level_16[-1].size(),level_32[-1].size())
        # concate_feature_map = torch.cat([level_4[-1], level_8[-1],level_16[-1], level_32[-1]], 1)
        aspp_result_4 = self.aspp_4(level_4[-1])

        aspp_result_8 = self.aspp_8(level_8[-1])
        aspp_result_16 = self.aspp_16(level_16[-1])
        aspp_result_32 = self.aspp_32(level_32[-1])
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = F.upsample(
            aspp_result_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_8 = F.upsample(
            aspp_result_8, size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_16 = F.upsample(
            aspp_result_16, size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_32 = F.upsample(
            aspp_result_32, size=x.size()[2:], mode='bilinear', align_corners=True)
        concate_feature_map = torch.cat([aspp_result_4, aspp_result_8, aspp_result_16, aspp_result_32], 1)
        out = self.final_conv(concate_feature_map)
        return out

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.stem0.parameters()})
        lr_list.append({'params': self.stem1.parameters()})
        lr_list.append({'params': self.stem2.parameters()})
        lr_list.append({'params': self.cells.parameters()})
        lr_list.append({'params': self.aspp_4.parameters()})
        lr_list.append({'params': self.aspp_8.parameters()})
        lr_list.append({'params': self.aspp_16.parameters()})
        lr_list.append({'params': self.aspp_32.parameters()})
        lr_list.append({'params': self.final_conv.parameters()})

        return lr_list

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._step) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        self.alphas_cell = nn.Parameter(
            torch.tensor(1e-3*torch.randn(k, num_ops).cuda(),
                         requires_grad=True))
        self.alphas_network = nn.Parameter(
            torch.tensor(1e-3*torch.randn(self._num_layers, 4, 3).cuda(),
                         requires_grad=True))
        # self.alphas_cell = self.alphas_cell.cuda()
        # self.alphas_network = self.alphas_network.cuda()
        self._arch_parameters = [
            self.alphas_cell,
            self.alphas_network
        ]

    def decode_network(self):
        best_result = []
        max_prop = 0
        def _parse(weight_network, layer, curr_value, curr_result, last):
            nonlocal best_result
            nonlocal max_prop
            if layer == self._num_layers:
                if max_prop < curr_value:
                    # print(curr_result)
                    best_result = curr_result[:]
                    max_prop = curr_value
                return

            if layer == 0:
                print('begin0')
                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

            elif layer == 1:
                print('begin1')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num,2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()


            elif layer == 2:
                print('begin2')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num,2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num,2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()
            else:

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num,2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num,2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 3
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num,0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num,1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
        network_weight = F.softmax(self.alphas_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [],0)
        print(max_prop)
        return best_result

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._step):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_cell = _parse(F.softmax(self.alphas_cell, dim=-1).data.cpu().numpy())
        concat = range(2+self._step-self._multiplier, self._step+2)
        genotype = Genotype(
            cell=gene_cell, cell_concat=concat
        )

        return genotype

    def _loss(self, input, target):
        input = F.sigmoid(input)
        return self._criterion(input, target)

'''
    Network Architecture Optimizer
'''
class Architect() :
    def __init__(self, model, args):
        self.model = model
        self.distributed = args.distributed
        self.multi_gpu = args.multi_gpu
        self.optimizer = torch.optim.Adam(self.model.module.arch_parameters(),
            lr=args.arch_lr, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def step(self, input_valid, target_valid) :
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid) :
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

def main():
    model = AutoDeeplab(5, 12, None)
    model.cuda()
    input = torch.randn(1, 3, 256, 256)
    input = input.cuda()
    output = model(input)
    result = model.decode_network()
    print(output.shape)
    print(result)
    print(model.genotype())
    # x = x.cuda()
    # y = model(x)
    # print(model.arch_parameters())
    # print(y.size())

if __name__ == '__main__':
    main()
