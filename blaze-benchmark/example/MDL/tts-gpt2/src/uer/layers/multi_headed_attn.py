# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        self.layer_weights = None
        self.layer_bias = None

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def prepare_paras(self, device):
        if self.layer_weights is None:
            self.layer_weights = torch.cat((self.linear_layers[0].weight,
                                            self.linear_layers[1].weight,
                                            self.linear_layers[2].weight), 0).to(device)
            self.layer_bias = torch.cat((self.linear_layers[0].bias,
                                         self.linear_layers[1].bias,
                                         self.linear_layers[2].bias), 0).to(device)

    def forward(self, key, value, query, mask, past=None):
        self.prepare_paras(query.device)
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = query.size()
        if past is not None:
            assert seq_length == 1, seq_length
        ## Should be [batch, 2, heads, sequence, per_head_size], where 2 is [k, v]
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, -1, hidden_size)
                   #view(batch_size, seq_length, hidden_size)

        if id(key) == id(value) and id(value) == id(query):
          query, key, value = F.linear(query, self.layer_weights, self.layer_bias).\
                    view(batch_size, seq_length, 3, heads_num, per_head_size).\
                    permute(2, 0, 3, 1, 4)
        else:
          query, key, value = [l(x). \
                                view(batch_size, -1, heads_num, per_head_size). \
                                transpose(1, 2) \
                                for l, x in zip(self.linear_layers, (query, key, value))
                               ]
        #batchsize, heads_num, dst_seq_length, per_head_size
        #present = [key, value]
        present = torch.stack((key, value), dim=1)
        if past is not None:
            pk, pv = torch.unbind(past, dim=1)
            key = torch.cat((pk, key), dim=-2)
            value = torch.cat((pv, value), dim=-2)

        #batchsize, heads_num, seq_length, per_head_size

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        # (batchsize, heads_num, dst_seq_length, src_seq_length) + (batchsize, 1, dst_seq_length, src_seq_length)
        if past is None:
            scores = scores + mask.float()
        probs = nn.Softmax(dim=-1)(scores)
        # (batchsize, heads_num, dst_seq_length, src_seq_length) * (batchsize, heads_num, src_seq_length, per_head_size)
        output = unshape(torch.matmul(probs, value))
        # batchsize, head_nums, dst_seq_length, per_head_size -> batchsize, dst_seq_length, hidden_size
        output = self.final_linear(output)

        return output, present
