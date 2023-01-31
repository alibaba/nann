# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *
from uer.layers.transformer_qkv import TransformerLayer as TransformerLayer_qkv
from uer.layers.transformer import TransformerLayer


class StorylinepropModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(StorylinepropModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.hidden_size = args.hidden_size

        self.prop_transformer = TransformerLayer(args)
        self.w_transformer = TransformerLayer_qkv(args)
        self.condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())

        # Subencoder.
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def forward(self, src, tgt, seg, masks, prop_keys, prop_values):
        # [batch_size, seq_length, emb_size]

        emb, prop_keys_emb, prop_values_emb = self.embedding(src, seg, prop_keys, prop_values)

        batch_size, prop_num, prop_len, prop_embed_size = prop_keys_emb.size()
        seq_len = src.size(-1)

        prop_keys_hidden = prop_keys_emb.view(-1, prop_len, prop_embed_size)
        prop_keys_masks = (prop_keys.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_keys_masks = prop_keys_masks.float()
        prop_keys_masks = (1.0 - prop_keys_masks) * -10000.0
        prop_keys_hidden = self.prop_transformer(prop_keys_hidden, prop_keys_masks)
        prop_keys_hidden = prop_keys_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        prop_values_hidden = prop_values_emb.view(-1, prop_len, prop_embed_size)
        prop_values_masks = (prop_values.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_values_masks = prop_values_masks.float()
        prop_values_masks = (1.0 - prop_values_masks) * -10000.0
        prop_values_hidden = self.prop_transformer(prop_values_hidden, prop_values_masks)
        prop_values_hidden = prop_values_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        if self.subencoder is not None:
            sub_ids = word2sub(src, self.vocab, self.sub_vocab, self.subword_type)
            emb = emb + self.subencoder(sub_ids).contiguous().view(*emb.size())

        output = self.encoder(emb, seg, masks)

        prop_masks = (prop_keys.sum(dim=-1) > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        prop_masks = prop_masks.float()
        prop_masks = (1.0 - prop_masks) * -10000.0
        prop_output = self.w_transformer(output, prop_keys_hidden, prop_values_hidden, prop_masks)

        condition_gate = self.condition_gate(torch.cat([output, prop_output], -1))
        output = condition_gate * output + (1 - condition_gate) * prop_output

        loss_info = self.target(output, tgt)

        return loss_info
