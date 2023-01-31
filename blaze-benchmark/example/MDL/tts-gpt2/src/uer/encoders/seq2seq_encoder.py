# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer


class Seq2seqEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(Seq2seqEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])

    def forward(self, emb, seg, mask, pasts):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        '''
        mask = mask.view(batch_size, 1, seq_length, seq_length)
        mask = (1.0 - mask) * -10000
        '''
        '''
        mask = torch.ones(seq_length, seq_length, device=emb.device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = mask.repeat(batch_size, 1, 1, 1)
        '''

        hidden = emb
        presents = []
        if pasts is not None:
            pasts_list = torch.unbind(pasts, dim=1)
            for i in range(self.layers_num):
                hidden, present = self.transformer[i](hidden, mask, pasts_list[i])
                presents.append(present)
        else:
            for i in range(self.layers_num):
                hidden, present = self.transformer[i](hidden, mask, None)
                presents.append(present)
        presents = torch.stack(presents, dim = 1)
        return hidden, presents
