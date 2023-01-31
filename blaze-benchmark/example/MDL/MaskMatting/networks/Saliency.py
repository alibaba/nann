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
from cqutils import initialize_weights
from networks.trimap_net import PSPNet50, PSPNet101
from networks.Layers import BatchNormCaffe
from config import *


__all__ = ['PSPNetResNet50',
           'FPNet',
           'PiCANet',
        ]

class PSPNetResNet50(nn.Module):
    def __init__(self, pretrained=True, hdfs_client=None):
        super(PSPNetResNet50, self).__init__()
        self.PSPNet = PSPNet50(1, pretrained=pretrained, hdfs_client=hdfs_client)

    def forward(self, input):
        if self.training:
            output = list(self.PSPNet(input))
            pred = []
            for item in output:
                pred.append(F.sigmoid(item))
        else:
            pred = F.sigmoid(self.PSPNet(input))
        return pred

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.PSPNet.base.parameters(), 'lr': 0.1*lr})
        lr_list.append({'params': self.get_parameters(self.PSPNet.ppm, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.PSPNet.final, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.PSPNet.ppm)})
        lr_list.append({'params': self.get_parameters(self.PSPNet.final)})

        return lr_list

    def get_parameters(self, model, bias=False):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.BatchNorm2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if bias and 'bias' in name:
                        yield param
                    elif not bias and 'weight' in name:
                        yield param

class PiCANet(nn.Module):
    def __init__(self, hdfs_client=None, freeze_base_bn=True,
                 base_type='pytorch', upsample_type='ConvTranspose'):
        super(PiCANet, self).__init__()
        self.hdfs_client = hdfs_client
        self.freeze_bn = freeze_base_bn
        self.upsample_type = upsample_type # 'ConvTranspose' 'upsampleConv' 'upsample'
        if base_type == 'pytorch':
            self.base = PiCANetBasePytorch(use_own_bn=True)
        else:
            self.base = PiCANetBase(use_own_bn=True)
        self.decoder1 = PiCANetDecoder(64, 64, 'C', is_upsample=True,
                                       upsample_type=self.upsample_type)
        self.decoder2 = PiCANetDecoder(256, 64, 'L', is_upsample=True,
                                       upsample_type=self.upsample_type)
        self.decoder3 = PiCANetDecoder(512, 256, 'L', is_upsample=False,
                                       upsample_type=self.upsample_type)
        self.decoder4 = PiCANetDecoder(1024, 512, 'G', is_upsample=False,
                                       upsample_type=self.upsample_type)
        self.decoder5 = PiCANetDecoder(2048, 1024, 'G', is_upsample=False,
                                       upsample_type=self.upsample_type)

        self.init_weight()

    def forward(self, input):
        encoder = self.base(input)
        dec5, side_output5 = self.decoder5(encoder[4])
        dec4, side_output4 = self.decoder4(encoder[3], dec5)
        dec3, side_output3 = self.decoder3(encoder[2], dec4)
        dec2, side_output2 = self.decoder2(encoder[1], dec3)
        _, side_output1 = self.decoder1(encoder[0], dec2)

        output = [side_output1,
                  side_output2,
                  side_output3,
                  side_output4,
                  side_output5,
                  ]

        return output

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(model_addr['res50']):
                raise RuntimeError('Please ensure {} exists.'.format(
                    model_addr['res50']))
            self.base.load_state_dict(torch.load(model_addr['res50']), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(model_addr['res50'],
                                           model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

        # initialize_weights(self.decoder1)
        # initialize_weights(self.decoder2)
        # initialize_weights(self.decoder3)
        # initialize_weights(self.decoder4)
        # initialize_weights(self.decoder5)
        self.decoder1.apply(self.layer_weights_init)
        self.decoder2.apply(self.layer_weights_init)
        self.decoder3.apply(self.layer_weights_init)
        self.decoder4.apply(self.layer_weights_init)
        self.decoder5.apply(self.layer_weights_init)

    def layer_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            # init.normal(m.weight.data, mean=0, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    length = len(param)
                    nn.init.constant(param, 0.0)
                    nn.init.constant(param[length//4:length//2], 1.0)
                elif 'weight' in name:
                    nn.init.uniform(param, -0.2, 0.2)
                    # nn.init.xavier_normal(param)
        elif isinstance(m, nn.ConvTranspose2d):
            size = m.weight.data.size()
            m.weight.data = bilinear_kernel(size[0], size[1], size[2])
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(PiCANet, self).train(mode)
        if self.freeze_bn:
            for m in self.base.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1*lr})
        lr_list.append({'params': self.get_parameters(self.decoder1, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder2, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder3, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder4, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder5, bias=True),
                        'lr': 2.0 * lr, 'weight_decay': 0.0})
        lr_list.append({'params': self.get_parameters(self.decoder1)})
        lr_list.append({'params': self.get_parameters(self.decoder2)})
        lr_list.append({'params': self.get_parameters(self.decoder3)})
        lr_list.append({'params': self.get_parameters(self.decoder4)})
        lr_list.append({'params': self.get_parameters(self.decoder5)})

        return lr_list

    def get_parameters(self, model, bias=False):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.BatchNorm2d):
                if bias and m.bias is not None:
                    yield m.bias
                elif not bias:
                    yield m.weight
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if bias and 'bias' in name:
                        yield param
                    elif not bias and 'weight' in name:
                        yield param

class PiCANetBasePytorch(nn.Module):
    def __init__(self, use_bn=True, use_own_bn=False):
        super(PiCANetBasePytorch, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.use_own_bn = use_own_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = BatchNormCaffe(64) if self.use_own_bn else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=1)
        self.layer4 = self._make_layer(512, 3, stride=1)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BatchNormCaffe(planes * self.expansion) if self.use_own_bn else \
                        nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, downsample,
                                 use_bn=self.use_bn, use_own_bn=self.use_own_bn))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes,
                                     use_bn=self.use_bn, use_own_bn=self.use_own_bn))

        return nn.Sequential(*layers)

class PiCANetBase(nn.Module):
    def __init__(self, use_bn=True, use_own_bn=False):
        super(PiCANetBase, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.use_own_bn = use_own_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=True)
        if self.use_bn:
            self.bn1 = BatchNormCaffe(64) if self.use_own_bn else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=1)
        self.layer4 = self._make_layer(512, 3, stride=1)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BatchNormCaffe(planes * self.expansion) if self.use_own_bn else \
                        nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, downsample,
                                 use_bn=self.use_bn, use_own_bn=self.use_own_bn))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes, use_bn=self.use_bn,
                                     use_own_bn=self.use_own_bn))

        return nn.Sequential(*layers)

class PiCANetDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, mode, is_upsample=True, upsample_type='upsample'):
        super(PiCANetDecoder, self).__init__()
        self.is_upsample = is_upsample
        self.upsample_type = upsample_type
        if self.is_upsample:
            if self.upsample_type == 'ConvTranspose':
                self.deconv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2,
                                                 padding=1, bias=False, groups=in_ch)
            elif self.upsample_type == 'upsampleConv':
                self.upsample_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PiCANetG(in_ch)
        elif mode == 'L':
            self.picanet = PiCANetL(in_ch)
        elif mode == 'C':
            self.picanet = PiCANetL(in_ch)

        self.conv2 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=1, padding=0)
        self.bn_feature = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, 1, kernel_size=1, padding=0)

    def forward(self, input1, input2=None):
        if input2 is None:
            fmap = input1
        else:
            en = input1
            dec = input2
            if self.is_upsample:
                if self.upsample_type == 'ConvTranspose':
                    dec = self.deconv(dec)
                elif self.upsample_type == 'upsampleConv':
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')
                    dec = self.upsample_conv(dec)
                else:
                    dec = F.upsample(dec, en.size()[2:], mode='bilinear')

            fmap = torch.cat((en, dec), dim=1)
            fmap = self.conv1(fmap)
            fmap = F.relu(fmap)

        fmap_att = self.picanet(fmap)
        x = torch.cat((fmap, fmap_att), 1)
        x = self.conv2(x)
        x = self.bn_feature(x)
        dec_out = F.relu(x)
        side_output = F.sigmoid(self.conv3(dec_out))

        return dec_out, side_output

class PiCANetG(nn.Module):
    def __init__(self, in_ch):
        super(PiCANetG, self).__init__()
        self.renet = ReNet(in_ch, 100)

    def forward(self, input):
        x = input
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)

        x = F.unfold(x, [10, 10], dilation=[3, 3])
        x = x.reshape(size[0], size[1], 10 * 10)
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])

        # wrong
        # kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, 10, 10)
        # x = torch.unsqueeze(x, 0)
        # x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
        #              padding=0, dilation=(1, 3, 3), groups=size[0])
        # x = torch.reshape(x, (size[0], size[1], size[2], size[3]))
        return x

class PiCANetL(nn.Module):
    def __init__(self, in_ch):
        super(PiCANetL, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(49)

    def forward(self, input):
        x = input
        size = x.size()
        kernel = self.conv1(x)
        kernel = F.relu(kernel)
        kernel = self.conv2(kernel)
        kernel = self.bn2(kernel)
        kernel = F.relu(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, 7 * 7, size[2] * size[3])
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        x = x.reshape(size[0], size[1], 7 * 7, -1)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=2)
        x = x.reshape(size[0], size[1], size[2], size[3])

        # wrong
        # kernel = kernel.reshape(size[0] * size[2] * size[3], 1, 1, 7, 7)
        # x = torch.unsqueeze(x, 0) # 1, batch, in_ch, height, width
        # x = F.conv3d(input=x, weight=kernel, bias=None, stride=1,
        #              padding=(0, 6, 6), dilation=(1, 2, 2), groups=size[0])
        # x = torch.reshape(x, (size[0], size[1], size[2], size[3]))

        return x

class ReNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        patch size = 1 and LSTM channel = 256 is default setting according to the origin code.
        outchannel: output channel size
        patch_size: num of patch to be cut.
        LSTM_channel: filters for LSTM.
        """
        super(ReNet, self).__init__()
        self.horizontal_LSTM = nn.LSTM(input_size=in_ch,
                                       hidden_size=256,
                                       batch_first=True,
                                       bidirectional=True)
        self.vertical_LSTM = nn.LSTM(input_size=512,
                                     hidden_size=256,
                                     batch_first=True,
                                     bidirectional=True)
        self.conv = nn.Conv2d(512, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, input):
        x = input
        size = x.size()
        height, width = size[2], size[3]
        x = torch.transpose(x, 1, 3) # batch, width, height, in_channel
        vertical_concat = []
        for i in range(height):
            h, _ = self.horizontal_LSTM(x[:, :, i, :])
            vertical_concat.append(h) # batch, width, 512
        x = torch.stack(vertical_concat, dim=2) # batch, width, height, 512
        horizontal_concat = []
        for i in range(width):
            h, _ = self.vertical_LSTM(x[:, i, :, :])
            horizontal_concat.append(h) # batch, height, 512
        x = torch.stack(horizontal_concat, dim=3) # batch, height, 512, width
        x = torch.transpose(x, 1, 2) # batch, 512, height, width
        x = self.conv(x)
        out = F.relu(self.bn(x))
        return out

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

class FPNet(nn.Module):
    def __init__(self, pretrained=True, hdfs_client=None):
        super(FPNet, self).__init__()
        self.pretrained = pretrained
        self.hdfs_client = hdfs_client
        # encoder part
        self.base = ResNet50()

        # decoder part
        self.decoder_in_ch = [256, 512, 1024, 2048]
        self.side_output_up_scale = [4, 8, 16, 32]

        self.decoder2 = FPNetDecoder(256)
        self.decoder3 = FPNetDecoder(512)
        self.decoder4 = FPNetDecoder(1024)
        self.decoder5 = FPNetDecoder(2048)
        # self.up1 = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding)

        # init weight
        self.init_weight()

    def forward(self, input):
        input_size = input.size()
        encoder = self.base(input)

        p5, side_output5 = self.decoder5(encoder[3])
        side_output5 = F.upsample(side_output5, input_size[2:], mode='bilinear')
        p4, side_output4 = self.decoder4(encoder[2], p5)
        side_output4 = F.upsample(side_output4, input_size[2:], mode='bilinear')
        p3, side_output3 = self.decoder3(encoder[1], p4)
        side_output3 = F.upsample(side_output3, input_size[2:], mode='bilinear')
        p2, side_output2 = self.decoder2(encoder[0], p3)
        side_output2 = F.upsample(side_output2, input_size[2:], mode='bilinear')

        return [side_output5, side_output4, side_output3, side_output2]

    def init_weight(self):
        if self.hdfs_client is None:
            if not os.path.exists(model_addr['res50']):
                raise RuntimeError('Please ensure {} exists.'.format(
                    model_addr['res50']))
            self.base.load_state_dict(torch.load(model_addr['res50']), strict=False)
        else:
            model_tmp_path = '/tmp/resnet50.pth'
            self.hdfs_client.copy_to_local(model_addr['res50'],
                    model_tmp_path)
            self.base.load_state_dict(torch.load(model_tmp_path), strict=False)

        initialize_weights(self.decoder2)
        initialize_weights(self.decoder3)
        initialize_weights(self.decoder4)
        initialize_weights(self.decoder5)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

class FPNetDecoder(nn.Module):
    def __init__(self, in_ch):
        super(FPNetDecoder, self).__init__()
        self.reduce_ch = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0)
        self.fusion = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input1, input2=None):
        if input2 is None:
            input1 = self.reduce_ch(input1)
            output = F.sigmoid(self.classifier(input1))
            return input1, output
        else:
            input2 = F.upsample(input2, input1.size()[2:], mode='bilinear')

            input1 = self.reduce_ch(input1)
            input1 = input1 + input2
            input1 = self.fusion(input1)

            output = F.sigmoid(self.classifier(input1))
            return input1, output

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        encoder = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        encoder.append(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes))

        return nn.Sequential(*layers)

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn=True,
                 use_own_bn=False):
        super(BottleNeck, self).__init__()
        self.use_bn = use_bn
        self.use_own_bn = use_own_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        if self.use_bn:
            self.bn1 = BatchNormCaffe(planes) if self.use_own_bn else nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        if self.use_bn:
            self.bn2 = BatchNormCaffe(planes) if self.use_own_bn else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.use_bn:
            self.bn3 = BatchNormCaffe(planes*4) if self.use_own_bn else nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def hook(module, input, output):
    anno_path = '/user/chenquan.cq/data/ECSSD/masks/0001.png'
    anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
    anno = cv2.resize(anno, (112, 112))
    anno = np.array(anno).astype(np.float)/255.0
    pred = output.detach().cpu().numpy()
    # loss = np.sum(anno*np.log(pred)+(1-anno)*np.log(1-pred))
    print(pred.shape, np.mean(pred).astype(np.float64))
    np.savetxt('pt_pool1.txt', pred[0,0], fmt='%.8f')
    # print(np.sum(np.abs(pred-anno)))

def load():
    import h5py
    def name_convert(name):
        num2alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        if len(name.split('.')) == 4:
            layer_num, branch, layer_type, tp = name.split('.')
            if 'conv' in layer_type:
                caffe_name = 'res'+str(int(layer_num[5])+1)+num2alpha[int(branch)]+'_branch2'+num2alpha[int(layer_type[-1])-1]+'.'+tp
            elif 'bn' in layer_type:
                if tp in 'running_mean':
                    caffe_name = 'bn'+str(int(layer_num[5])+1)+num2alpha[int(branch)]+'_branch2'+num2alpha[int(layer_type[-1])-1]+'.weight'
                elif tp in 'running_var':
                    caffe_name = 'bn'+str(int(layer_num[5])+1)+num2alpha[int(branch)]+'_branch2'+num2alpha[int(layer_type[-1])-1]+'.bias'
                elif tp in 'num_batches_tracked':
                    caffe_name = None
                else:
                    caffe_name = 'scale'+str(int(layer_num[5])+1)+num2alpha[int(branch)]+'_branch2'+num2alpha[int(layer_type[-1])-1]+'.'+tp
        elif len(name.split('.')) == 5:
            layer_num, branch, layer_type, num, tp = name.split('.')
            prefix = 'res' if num=='0' else 'scale'
            if num=='0':
                caffe_name = 'res'+str(int(layer_num[5])+1)+'a_branch1'+'.'+tp
            else:
                if tp in 'running_mean':
                    caffe_name = 'bn' + str(int(layer_num[5]) + 1)+'a_branch1'+'.weight'
                elif tp in 'running_var':
                    caffe_name = 'bn' + str(int(layer_num[5]) + 1)+'a_branch1'+'.bias'
                elif tp in 'num_batches_tracked':
                    caffe_name = None
                else:
                    caffe_name = 'scale' + str(int(layer_num[5]) + 1) + 'a_branch1' + '.' + tp
        else:
            layer_name, tp = name.split('.')
            if 'conv' in layer_name:
                caffe_name = name
            elif 'fc' in layer_name:
                caffe_name = 'fc1000.' + tp
            else:
                if tp in 'running_mean':
                    caffe_name = 'bn_conv1.weight'
                elif tp in 'running_var':
                    caffe_name = 'bn_conv1.bias'
                elif tp in 'num_batches_tracked':
                    caffe_name = None
                else:
                    caffe_name = 'scale_conv1.'+tp

        return caffe_name
    model = PiCANetBase()
    model.eval()
    state_dict = h5py.File('ResNet-50-caffe.h5', 'r')
    print(np.array(state_dict['scale3a_branch2b.weight']).astype(np.float64))
    # sd = torch.load('Res50_caffe.pth')
    # print(sd['conv1.bias'].data)
    new_dict = {}
    for l, p in model.state_dict().items():
        name_key = name_convert(l)
        print(name_key, l, p.size())
        if name_key is None:
            continue
            new_dict[l] = torch.from_numpy(np.array(1.0)).view_as(p)
        else:
            # if 'running_var' in l:
            #     new_dict[l] = torch.from_numpy(np.sqrt(np.array(state_dict[name_key]))).view_as(p)
            # else:
            new_dict[l] = torch.from_numpy(np.array(state_dict[name_key]).astype(np.float64))
    model.load_state_dict(new_dict, strict=True)
    torch.save(model.state_dict(), 'Res50_caffe.pth')

if __name__ == '__main__':
    # load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torchvision.models.resnet50()
    model = PiCANet()
    print(model)
    # print([name for name, param in model.named_parameters()])
    model = model.to(device)

    # from torchsummary import summary
    # summary(model, (3, 224, 224))

    img_path = '/user/chenquan.cq/data/ECSSD/images/0001.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32)
    img -= np.array((104.008,116.669,122.675))
    # img = img[:, :, [2,1,0]]
    img = img.transpose((2,0,1))
    # np.savetxt('img_pytorch.txt', img[0], fmt='%.8f')
    img = np.load('img_caffe.txt.npy')
    img = torch.from_numpy(img.copy().astype(np.float32))
    img = img.to(device)
    img = img.unsqueeze(0)
    handle = model.base.maxpool.register_forward_hook(hook)
    model.eval()
    res = model(img)
    handle.remove()
    # print(model.base.layer2[1].downsample[1].running_var)
    # print(model.base.layer2[1].downsample[1].running_mean)
    # print(model.base.layer2[0].bn2.weight.detach().cpu().numpy().astype(np.float64))
    print(model.base.layer1[0].conv1.weight[0].detach().cpu().numpy())
    # print([(name, param) for name, param in model.decoder5.picanet.renet.named_parameters() if 'bias' in name])

    # input = torch.randn(1, 3, 1, 1)
    # lstm = nn.LSTM(3,3)
    # input = F.upsample(input, (14, 14))
    # input = input.unfold(2, 3, 1)
    # size = input.size()
    # input = F.unfold(input, [10, 10], dilation=[3, 3])
    # input = input.reshape(size[0], size[1], 10 * 10)
    # input = torch.transpose(input, 1,2)
    # print([i for i in lstm.named_parameters()])


