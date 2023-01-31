# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer


class GptEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(GptEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        print('gpt encoder')

    def forward(self, emb, seg, pasts):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, dst_seq_length, _ = emb.size()

        hidden = emb
        presents = []
        if pasts is None:
            src_seq_length = dst_seq_length
            # Generate mask according to segment indicators.
            # mask: [batch_size x 1 x seq_length x seq_length]
            mask = torch.ones(dst_seq_length, src_seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
            for i in range(self.layers_num):
                hidden, present = self.transformer[i](hidden, mask, None)
                presents.append(present)
        else:
            # the query is actually the last query if past not None
            # no need to use mask
            mask = None
            pasts_list = torch.unbind(pasts, dim=1) # batch, n_layers, 2, heads, seq_len, feature
            for i in range(self.layers_num):
                hidden, present = self.transformer[i](hidden, mask, pasts_list[i])
                presents.append(present)
        presents = torch.stack(presents, dim = 1)
        return hidden, presents
