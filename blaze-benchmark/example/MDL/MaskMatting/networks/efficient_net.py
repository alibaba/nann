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
import collections
import re

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)
from networks.Layers import layer_weights_init, PlaceHolderLayer,\
    DistributedBatchNorm2d, BatchNormCaffe
from cqutils import print_network
from config import *

__all__ = [
        'EfficientNet',
        'build_efficient_net'
        ]

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'features_only',
    'fix_bn',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

def _normal_layer(in_ch, fix_bn, **kwargs):
    if fix_bn:
        return BatchNormCaffe(in_ch, **kwargs)
    else:
        return nn.BatchNorm2d(in_ch, **kwargs)

class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, depth_multiplier=1, **kwargs):
        super(DepthwiseConv2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            **kwargs)

    def forward(self, input):
        return self.conv(input)

class MBConvBlock(nn.Module):
    """A class of MBConv: Mobile Inveretd Residual Bottleneck.
        Attributes:
        has_se: boolean. Whether the block contains a Squeeze
        and Excitation layer inside.
        endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, fix_bn=True):
        """Initializes a MBConv block.
            Args:
                block_args: BlockArgs, arguments to create a Block.
                global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self.has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self.endpoints = None

        # Builds the block accordings to arguments.
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = nn.Conv2d(
                self._block_args.input_filters,
                filters,
                kernel_size=1,
                bias=False)
            self._bn0 = _normal_layer(filters, fix_bn)

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        # self._depthwise_conv = DepthwiseConv2D(
        #     filters,
        #     depth_multiplier=1,
        #     groups=filters,
        #     kernel_size=kernel_size,
        #     stride=self._block_args.strides,
        #     padding=(kernel_size//2*self._block_args.strides[0],
        #              kernel_size//2*self._block_args.strides[1]),
        #     bias=False)
        self._depthwise_conv = nn.Conv2d(
            filters,
            filters,
            groups=filters,
            kernel_size=kernel_size,
            stride=self._block_args.strides,
            padding=(kernel_size // 2,
                     kernel_size // 2),
            bias=False)
        self._bn1 = _normal_layer(filters, fix_bn)

        if self.has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = nn.Conv2d(
                filters,
                num_reduced_filters,
                kernel_size=1)
            self._se_expand = nn.Conv2d(
                num_reduced_filters,
                filters,
                kernel_size=1)

        # Output phase:
        self._project_conv = nn.Conv2d(
            filters,
            self._block_args.output_filters,
            kernel_size=1,
            bias=False)
        self._bn2 = _normal_layer(
            self._block_args.output_filters, fix_bn)

        self.relu = nn.ReLU(inplace=True)

    def block_args(self):
        return self._block_args

    def _call_se(self, input_tensor):
        se_tensor = input_tensor.mean(2, keepdim=True).mean(3, keepdim=True)
        se_tensor = self._se_expand(self.relu(self._se_reduce(se_tensor)))
        return F.sigmoid(se_tensor) * input_tensor

    def forward(self, inputs, drop_connect_rate=None):
        if self._block_args.expand_ratio != 1:
            x = self.relu(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs

        x = self.relu(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x))
        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides)\
                    and self._block_args.input_filters == self._block_args.output_filters:
                if drop_connect_rate:
                    x = self.drop_connect(x, drop_connect_rate)
                    x += inputs
        return x

    def drop_connect(self, inputs, p):
        """ Drop connect. """
        if not self.training: return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - p
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class EfficientNet(nn.Module):
    """A class implements tf.keras.Model for MNAS-like model.
        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, blocks_args=None, global_params=None):
        """Initializes an `Model` instance.
            Args:
            blocks_args: A list of BlockArgs to construct block modules.
            global_params: GlobalParams, a set of global parameters.
            Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNet, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._blocks_args = blocks_args
        self._global_params = global_params
        self.endpoints = None
        self.features_only = global_params.features_only
        self.fix_bn = global_params.fix_bn
        self.encoder_idx = []

        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params),
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params))
            self.encoder_idx.append(block_args.num_repeat)
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.fix_bn))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
                for _ in range(block_args.num_repeat - 1):
                    self._blocks.append(MBConvBlock(block_args, self.fix_bn))
        self._blocks = nn.ModuleList(self._blocks)
        self.encoder_idx = np.cumsum(self.encoder_idx) - 1

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon

        # Stem part.
        self._conv_stem = nn.Conv2d(
            3,
            round_filters(32, self._global_params),
            kernel_size=3,
            padding=1,
            bias=False)
        self._bn0 = _normal_layer(
            round_filters(32, self._global_params),
            self.fix_bn,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon)

        # Head part.
        self._conv_head = nn.Conv2d(
            round_filters(self._blocks_args[-1].output_filters,
                          self._global_params),
            round_filters(1280, self._global_params),
            kernel_size=1,
            bias=False)
        self._bn1 = _normal_layer(
            round_filters(1280, self._global_params),
            self.fix_bn,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(
            round_filters(1280, self._global_params),
            self._global_params.num_classes)

        if self._global_params.dropout_rate > 0:
            self._dropout = nn.Dropout2d(self._global_params.dropout_rate)
        else:
            self._dropout = None

        self.relu = nn.ReLU(inplace=True)

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in EfficientNet: {}".format(num_params))

    def forward(self, inputs):
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        outputs = self.relu(self._bn0(self._conv_stem(inputs)))
        self.endpoints['stem'] = outputs

        # Calls blocks.
        reduction_idx = 0
        encoder = []
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or
                    self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            outputs = block(outputs)
            if idx in self.encoder_idx:
                encoder.append(outputs)
        #     drop_rate = self._global_params.drop_connect_rate
        #     if drop_rate:
        #         drop_rate *= float(idx) / len(self._blocks)
        #     self.endpoints['block_%s' % idx] = outputs
        #     print('{} {}'.format('block_%s' % idx, outputs.shape))
        #     if is_reduction:
        #         self.endpoints['reduction_%s' % reduction_idx] = outputs
        #     if block.endpoints:
        #         for k, v in block.endpoints.items():
        #             self.endpoints['block_%s/%s' % (idx, k)] = v
        #             if is_reduction:
        #                 self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        # self.endpoints['global_pool'] = outputs

        if not self.features_only:
            # Calls final layers and returns logits.
            outputs = self.relu(
                self._bn1(self._conv_head(outputs)))
            outputs = self._avg_pooling(outputs)
            if self._dropout:
                outputs = self._dropout(outputs)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = self._fc(outputs)
            self.endpoints['head'] = outputs
        return encoder

def decoder(string_list):
    blocks_args = []
    for block_string in string_list:
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        blocks_args.append(
            BlockArgs(
                kernel_size=int(options['k']),
                num_repeat=int(options['r']),
                input_filters=int(options['i']),
                output_filters=int(options['o']),
                expand_ratio=int(options['e']),
                id_skip=('noskip' not in block_string),
                se_ratio=float(options['se']) if 'se' in options else None,
                strides=(int(options['s'][0]), int(options['s'][1])))
        )
    return blocks_args

def load_pretained(hdfs_client, netowrks, pretrained):
    if hdfs_client is None:
        if not os.path.exists(pretrained):
            raise RuntimeError('Please ensure {} exists.'.format(
                pretrained))
        checkpoint = torch.load(pretrained)
    else:
        model_tmp_path = '/tmp/resnet50.pth'
        hdfs_client.copy_to_local(pretrained,
                                       model_tmp_path)
        checkpoint = torch.load(model_tmp_path)

    try:
        new_dict = OrderedDict()
        for k, _ in netowrks.state_dict().items():
            if 'num_batches_tracked' in k:
                new_dict[k] = torch.zeros(1)
            else:
                new_dict[k] = checkpoint[k]
        netowrks.load_state_dict(new_dict)
    except Exception as e:
        new_dict = OrderedDict()
        for k, _ in netowrks.state_dict().items():
            if 'num_batches_tracked' in k:
                new_dict[k] = torch.zeros(1)
            else:
                nk = 'module.' + k
                new_dict[k] = checkpoint[nk]
        netowrks.load_state_dict(new_dict)

def build_efficient_net(model_name, pretrained=None, hdfs_client=None):
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    width_coefficient, depth_coefficient, _, dropout_rate =\
        params_dict[model_name]

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_params = decoder(blocks_args)
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        features_only=True,
        fix_bn=True,
    )

    model = EfficientNet(blocks_params, global_params)
    if pretrained is not None:
        load_pretained(hdfs_client, model, pretrained)
    return model

if __name__ == '__main__':
    net = build_efficient_net('efficientnet-b3')
    model_path = 'D:/project/models/efficientnet-b0-08094119.pth'
    alpha = torch.randn(1, 3, 224, 224)
    # net.load_state_dict(torch.load(model_path))
    en = net(alpha)
    for i in en:
        print('==>{}'.format(i.shape))

    # for k, v in net.endpoints.items():
    #     print(k, v.shape)