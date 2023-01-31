# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *
from uer.layers.transformer_qkv import TransformerLayer as TransformerLayer_qkv
from uer.layers.transformer import TransformerLayer


class StorylinepropclsModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(StorylinepropclsModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.hidden_size = args.hidden_size

        self.word_transformer = TransformerLayer(args)
        self.w_transformer = TransformerLayer_qkv(args)
        self.condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())
        self.title_transformer = TransformerLayer(args)
        self.title_condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())
        self.title_w_transformer = TransformerLayer_qkv(args)

        # Subencoder.
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def forward(self, src, seg, prop_keys, prop_values, target_words):
        # [batch_size, seq_length, emb_size]

        emb, prop_keys_emb, prop_values_emb, target_words_emb = self.embedding(src, seg, prop_keys, prop_values, target_words)

        batch_size, prop_num, prop_len, prop_embed_size = prop_keys_emb.size()
        target_words_num, target_words_len = target_words_emb.size()[1:3]
        seq_len = src.size(-1)

        prop_keys_hidden = prop_keys_emb.view(-1, prop_len, prop_embed_size)
        prop_keys_masks = (prop_keys.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_keys_masks = prop_keys_masks.float()
        prop_keys_masks = (1.0 - prop_keys_masks) * -10000.0
        prop_keys_hidden = self.word_transformer(prop_keys_hidden, prop_keys_masks)
        prop_keys_hidden = prop_keys_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        prop_values_hidden = prop_values_emb.view(-1, prop_len, prop_embed_size)
        prop_values_masks = (prop_values.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_values_masks = prop_values_masks.float()
        prop_values_masks = (1.0 - prop_values_masks) * -10000.0
        prop_values_hidden = self.word_transformer(prop_values_hidden, prop_values_masks)
        prop_values_hidden = prop_values_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        target_words_hidden = target_words_emb.view(-1, target_words_len, prop_embed_size)
        target_words_masks = (target_words.view(-1, target_words_len) > 0).unsqueeze(1).repeat(1, target_words_len, 1).unsqueeze(1)
        target_words_masks = target_words_masks.float()
        target_words_masks = (1.0 - target_words_masks) * -10000.0
        target_words_hidden = self.word_transformer(target_words_hidden, target_words_masks)
        target_words_hidden = target_words_hidden[:, 0, :].squeeze(1).view(batch_size, target_words_num, prop_embed_size)

        title_masks = (seg > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        title_masks = title_masks.float()
        title_masks = (1.0 - title_masks) * -10000.0
        title_hidden = self.title_transformer(emb, title_masks)

        # target_mask is not right, not including feature blinding masks.
        target_mask = (target_words.sum(dim=-1) > 0).int()
        output = self.encoder(target_words_hidden, target_mask)

        combine_title_masks = (seg > 0).unsqueeze(1).repeat(1, target_words_num, 1).unsqueeze(1)
        combine_title_masks = combine_title_masks.float()
        combine_title_masks = (1.0 - combine_title_masks) * -10000.0
        combine_title_output = self.title_w_transformer(output, title_hidden, title_hidden, combine_title_masks)

        title_condition_gate = self.title_condition_gate(torch.cat([output, combine_title_output], -1))
        output = title_condition_gate * output + (1 - title_condition_gate) * combine_title_output

        org_prop_masks = (prop_keys.sum(dim=-1) > 0).int()
        prop_masks = org_prop_masks.unsqueeze(1).repeat(1, target_words_num, 1).unsqueeze(1)
        prop_masks = prop_masks.float()
        prop_masks = (1.0 - prop_masks) * -10000.0
        prop_output = self.w_transformer(output, prop_keys_hidden, prop_values_hidden, prop_masks)

        condition_gate = self.condition_gate(torch.cat([output, prop_output], -1))
        output = condition_gate * output + (1 - condition_gate) * prop_output

        loss_info = self.target(output, target_words_hidden, target_mask, prop_values_hidden)

        return loss_info
