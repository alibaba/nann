# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *
from uer.layers.transformer_qkv import TransformerLayer as TransformerLayer_qkv
from uer.layers.transformer import TransformerLayer
from uer.layers.resnet import *
from uer.layers.layer_norm import LayerNorm
from torch.autograd import Variable


class StorylinepropattrpictmultiModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(StorylinepropattrpictmultiModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        self.hidden_size = args.hidden_size

        self.word_transformer = TransformerLayer(args)

        self.prop_w_transformer = TransformerLayer_qkv(args)
        self.prop_condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
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

        self.class_start = Variable(0.02 * torch.randn(self.hidden_size), requires_grad = True)

        self.class_transformer = TransformerLayer(args)
        self.class_w_transformer = TransformerLayer_qkv(args)

        self.pict_gat = TransformerLayer(args)

        # Subencoder.
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def set_target(self, attr_keys, attr_values):
        attr_keys_word_emb = self.embedding.word_embedding(attr_keys)
        attr_keys_pos_emb = self.embedding.position_embedding(torch.arange(0, attr_keys_word_emb.size(1), device=attr_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_keys_word_emb.size(0), 1))
        attr_keys_emb = self.embedding.layer_norm(attr_keys_word_emb + attr_keys_pos_emb)

        attr_values_word_emb = self.embedding.word_embedding(attr_values)
        attr_values_pos_emb = self.embedding.position_embedding(torch.arange(0, attr_values_word_emb.size(1), device=attr_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_values_word_emb.size(0), 1))
        attr_values_emb = self.embedding.layer_norm(attr_values_word_emb + attr_values_pos_emb)

        attr_keys_num, attr_len, attr_embed_size = attr_keys_emb.size()
        attr_values_num, attr_len, attr_embed_size = attr_values_emb.size()

        attr_keys_hidden = attr_keys_emb.view(-1, attr_len, attr_embed_size)
        attr_keys_masks = (attr_keys.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_keys_masks = attr_keys_masks.float()
        attr_keys_masks = (1.0 - attr_keys_masks) * -10000.0
        attr_keys_hidden = self.word_transformer(attr_keys_hidden, attr_keys_masks)
        all_attr_keys_hidden = attr_keys_hidden[:, 0, :].view(attr_keys_num, attr_embed_size)

        attr_values_hidden = attr_values_emb.view(-1, attr_len, attr_embed_size)
        attr_values_masks = (attr_values.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_values_masks = attr_values_masks.float()
        attr_values_masks = (1.0 - attr_values_masks) * -10000.0
        attr_values_hidden = self.word_transformer(attr_values_hidden, attr_values_masks)
        all_attr_values_hidden = attr_values_hidden[:, 0, :].view(attr_values_num, attr_embed_size)

        return all_attr_keys_hidden, all_attr_values_hidden

    def forward(self, src, tgt, seg, masks, prop_keys, prop_values, attr_keys, attr_values, attr_keys_target, attr_values_target, all_attr_keys, all_attr_values, picts):
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
        attr_keys_hidden = attr_keys_hidden[:, 0, :].squeeze(1).view(batch_size, attr_num, 1, attr_embed_size)

        attr_values_hidden = attr_values_emb.view(-1, attr_len, attr_embed_size)
        attr_values_masks = (attr_values.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_values_masks = attr_values_masks.float()
        attr_values_masks = (1.0 - attr_values_masks) * -10000.0
        attr_values_hidden = self.word_transformer(attr_values_hidden, attr_values_masks)
        attr_values_hidden = attr_values_hidden[:, 0, :].squeeze(1).view(batch_size, attr_num, 1, attr_embed_size)

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

        pict_feature = self.pict_model(picts)
        pict_feature = pict_feature.transpose(1, 3).transpose(1, 2)

        h = pict_feature.size(1)
        w = pict_feature.size(2)

        pict_feature = self.pict_fc(pict_feature.view(batch_size, h * w, -1))
        pict_gat_masks = torch.zeros(batch_size, 1, h * w, h * w, device=pict_feature.device).float()
        pict_feature = self.pict_gat(pict_feature, pict_gat_masks)

        pict_masks = torch.zeros(batch_size, 1, seq_len, h * w, device=pict_feature.device).float()
        pict_output = self.pict_w_transformer(output, pict_feature, pict_feature, pict_masks)

        pict_condition_gate = self.pict_condition_gate(torch.cat([output, pict_output], -1))
        output = pict_condition_gate * output + (1 - pict_condition_gate) * pict_output

        class_start = self.class_start.to(device=attr_keys_hidden.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, attr_num, 1).unsqueeze(2)
        class_seq_hidden = torch.cat((class_start, attr_keys_hidden), 2).view(batch_size * attr_num, 2, -1)
        class_masks = torch.tril(torch.ones((2, 2), dtype=torch.long, device=class_seq_hidden.device)).unsqueeze(0).repeat(batch_size * attr_num, 1, 1).unsqueeze(1)
        class_masks = class_masks.float()
        class_masks = (1.0 - class_masks) * -10000.0
        class_seq_hidden = self.class_transformer(class_seq_hidden, class_masks)

        pict_feature = pict_feature.unsqueeze(1).repeat(1, attr_num, 1, 1).view(batch_size * attr_num, h * w, -1)
        class_pict_masks = torch.zeros(batch_size * attr_num, 1, 2, h * w, device=pict_feature.device).float()
        class_seq_output = self.class_w_transformer(class_seq_hidden, pict_feature, pict_feature, class_pict_masks).view(batch_size, attr_num, 2, -1)

        attr_masks = (attr_keys.sum(dim=-1) > 0).int()

        all_attr_keys_hidden, all_attr_values_hidden = self.set_target(all_attr_keys, all_attr_values)

        loss_info = self.target(output, tgt, class_seq_output, attr_keys_target, attr_values_target, attr_masks, all_attr_keys_hidden, all_attr_values_hidden)

        return loss_info