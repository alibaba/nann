# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *
from uer.layers.transformer_qkv import TransformerLayer as TransformerLayer_qkv
from uer.layers.transformer import TransformerLayer
from uer.layers.resnet import *
from uer.layers.layer_norm import LayerNorm


class StorylinepropattrpictModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(StorylinepropattrpictModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.hidden_size = args.hidden_size

        self.word_transformer = TransformerLayer(args)

        self.prop_w_transformer = TransformerLayer_qkv(args)
        self.prop_condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())

        self.attr_w_transformer = TransformerLayer_qkv(args)
        self.attr_condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())

        pict_model = resnet101(pretrained=False, num_classes=1000)
        pict_model_state_dict = torch.load('/home/service/models/resnet101_best_model.pkl')
        pict_model.load_state_dict(pict_model_state_dict)
        self.pict_model = nn.Sequential(*list(pict_model.children())[:-2])

        self.pict_fc = nn.Sequential(nn.Linear(512 * 4, self.hidden_size),
                                LayerNorm(args.hidden_size))
        self.pict_w_transformer = TransformerLayer_qkv(args)
        self.pict_condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())

        # Subencoder.
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def forward(self, src, tgt, seg, masks, prop_keys, prop_values, attr_keys, attr_values, picts):
        # [batch_size, seq_length, emb_size]

        emb, prop_keys_emb, prop_values_emb, attr_keys_emb, attr_values_emb = self.embedding(src, seg, prop_keys, prop_values, attr_keys, attr_values)

        batch_size, prop_num, prop_len, prop_embed_size = prop_keys_emb.size()
        seq_len = src.size(-1)
        batch_size, attr_num, attr_len, attr_embed_size = attr_keys_emb.size()

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

        attr_keys_hidden = attr_keys_emb.view(-1, attr_len, attr_embed_size)
        attr_keys_masks = (attr_keys.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_keys_masks = attr_keys_masks.float()
        attr_keys_masks = (1.0 - attr_keys_masks) * -10000.0
        attr_keys_hidden = self.word_transformer(attr_keys_hidden, attr_keys_masks)
        attr_keys_hidden = attr_keys_hidden[:, 0, :].squeeze(1).view(batch_size, attr_num, attr_embed_size)

        attr_values_hidden = attr_values_emb.view(-1, attr_len, attr_embed_size)
        attr_values_masks = (attr_values.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_values_masks = attr_values_masks.float()
        attr_values_masks = (1.0 - attr_values_masks) * -10000.0
        attr_values_hidden = self.word_transformer(attr_values_hidden, attr_values_masks)
        attr_values_hidden = attr_values_hidden[:, 0, :].squeeze(1).view(batch_size, attr_num, attr_embed_size)

        if self.subencoder is not None:
            sub_ids = word2sub(src, self.vocab, self.sub_vocab, self.subword_type)
            emb = emb + self.subencoder(sub_ids).contiguous().view(*emb.size())

        output = self.encoder(emb, seg, masks)

        prop_masks = (prop_keys.sum(dim=-1) > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        prop_masks = prop_masks.float()
        prop_masks = (1.0 - prop_masks) * -10000.0
        prop_output = self.prop_w_transformer(output, prop_keys_hidden, prop_values_hidden, prop_masks)

        prop_condition_gate = self.prop_condition_gate(torch.cat([output, prop_output], -1))
        output = prop_condition_gate * output + (1 - prop_condition_gate) * prop_output

        attr_masks = (attr_keys.sum(dim=-1) > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        attr_masks = attr_masks.float()
        attr_masks = (1.0 - attr_masks) * -10000.0
        attr_output = self.attr_w_transformer(output, attr_keys_hidden, attr_values_hidden, attr_masks)

        attr_condition_gate = self.attr_condition_gate(torch.cat([output, attr_output], -1))
        output = attr_condition_gate * output + (1 - attr_condition_gate) * attr_output

        pict_feature = self.pict_model(picts)
        pict_feature = pict_feature.transpose(1, 3).transpose(1, 2)

        h = pict_feature.size(1)
        w = pict_feature.size(2)

        pict_feature = self.pict_fc(pict_feature.view(batch_size, h * w, -1))

        pict_masks = torch.zeros(batch_size, 1, seq_len, h * w, device=pict_feature.device).float()
        pict_output = self.pict_w_transformer(output, pict_feature, pict_feature, pict_masks)

        pict_condition_gate = self.pict_condition_gate(torch.cat([output, pict_output], -1))
        output = pict_condition_gate * output + (1 - pict_condition_gate) * pict_output

        loss_info = self.target(output, tgt)

        return loss_info
