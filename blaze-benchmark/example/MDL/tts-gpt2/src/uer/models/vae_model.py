# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *
from uer.layers.transformer_qkv import TransformerLayer
from torch.autograd import Variable


class VaeModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(VaeModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.hidden_size = args.hidden_size

        self.condition_text_mulogvar = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.condition_title_mulogvar = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        )

        self.w_transformer = TransformerLayer(args)

        self.cluster_center = Variable(0.02 * torch.randn(10, self.hidden_size), requires_grad = True)

        '''
        self.condition_text_mu_bn = nn.BatchNorm1d(self.hidden_size * 10)
        self.condition_title_mu_bn = nn.BatchNorm1d(self.hidden_size * 10)
        '''


    def forward(self, src, tgt, seg, condition_title, condition_title_seg, condition_text, condition_text_seg, masks):
        # [batch_size, seq_length, emb_size]

        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg, masks)

        batch_size, condition_length = condition_title.size()
        mask = torch.zeros(batch_size, 1, condition_length, condition_length, device=emb.device)
        condition_title_emb = self.embedding(condition_title, condition_title_seg)
        '''
        mask = (condition_title_seg > 0). \
                unsqueeze(1). \
                repeat(1, condition_length, 1). \
                unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        '''
        condition_title_output = self.encoder(condition_title_emb, condition_title_seg, mask)

        condition_text_length = condition_text.size(1)
        mask = torch.zeros(batch_size, 1, condition_text_length, condition_text_length, device=emb.device)
        condition_text_emb = self.embedding(condition_text, condition_text_seg)
        '''
        mask = (condition_text_seg > 0). \
                unsqueeze(1). \
                repeat(1, condition_length, 1). \
                unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        '''
        condition_text_output = self.encoder(condition_text_emb, condition_text_seg, mask)

        query = self.cluster_center.to(device=condition_text_output.device).unsqueeze(0).repeat(batch_size, 1, 1)
        #query = self.cluster_center.unsqueeze(0).repeat(batch_size, 1, 1)

        mask = torch.zeros(batch_size, 1, 10, condition_length, device=condition_title_output.device)
        condition_title_output = self.w_transformer(query, condition_title_output, condition_title_output, mask)

        mask = torch.zeros(batch_size, 1, 10, condition_text_length, device=condition_text_output.device)
        condition_text_output = self.w_transformer(query, condition_text_output, condition_text_output, mask)

        hidden_size = condition_text_output.size(-1)

        condition_title_mu, condition_title_logvar = self.condition_title_mulogvar(condition_title_output).split(hidden_size, dim=-1)
        condition_text_mu, condition_text_logvar = self.condition_text_mulogvar(condition_text_output).split(hidden_size, dim=-1)

        '''
        condition_text_mu = self.condition_text_mu_bn(condition_text_mu.contiguous().view(batch_size, 10 * hidden_size))
        condition_text_mu = condition_text_mu.view(batch_size, 10, hidden_size)
        condition_title_mu = self.condition_title_mu_bn(condition_title_mu.contiguous().view(batch_size, 10 * hidden_size))
        condition_title_mu = condition_title_mu.view(batch_size, 10, hidden_size)
        '''

        loss_info = self.target(output, tgt, condition_title_mu, condition_title_logvar, condition_text_mu, condition_text_logvar)

        return loss_info