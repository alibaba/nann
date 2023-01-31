# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # TRT does not support negative Gather index, when converting gpt to ONNX
        # may need std = x.std(2, keepdim=True), but in getpictlabel, 3 is the last dim
        lastdim = x.dim() - 1
        std = x.std(lastdim, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta
