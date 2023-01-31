# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)
        print('bert')

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        #emb = self.dropout(self.layer_norm(emb))
        emb = self.layer_norm(emb)
        return emb


class WordEmbedding(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        emb = self.dropout(self.layer_norm(emb))
        return emb


class ReversedEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(ReversedEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)
        self.reversed_position_embedding = nn.Embedding(self.max_length, args.emb_size)

    def forward(self, src, seg, expected_target_length=None):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        reversed_index = seg.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        if expected_target_length != None:
            reversed_index += (reversed_index > 0) * expected_target_length
            reversed_index *= (reversed_index > 0)
        reversed_pos_emb = self.reversed_position_embedding(reversed_index)


        emb = word_emb + pos_emb + seg_emb + reversed_pos_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class FpdgEmbedding(nn.Module):
    """
    FPDG embedding consists of four parts:
    word embedding, type embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size, type_vocab_size):
        super(FpdgEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

        self.type_embedding = nn.Embedding(type_vocab_size, args.emb_size)

    def forward(self, src, src_type, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)
        type_emb = self.type_embedding(src_type)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb, type_emb

class StorylinepropEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(StorylinepropEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, prop_keys, prop_values):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))

        prop_keys_word_emb = self.word_embedding(prop_keys)
        prop_keys_pos_emb = self.position_embedding(torch.arange(0, prop_keys_word_emb.size(2), device=prop_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_keys_word_emb.size(0), prop_keys_word_emb.size(1), 1))
        prop_keys_emb = self.dropout(self.layer_norm(prop_keys_word_emb + prop_keys_pos_emb))

        prop_values_word_emb = self.word_embedding(prop_values)
        prop_values_pos_emb = self.position_embedding(torch.arange(0, prop_values_word_emb.size(2), device=prop_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_values_word_emb.size(0), prop_values_word_emb.size(1), 1))
        prop_values_emb = self.dropout(self.layer_norm(prop_values_word_emb + prop_keys_pos_emb))

        return emb, prop_keys_emb, prop_values_emb

class StorylinepropattrEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(StorylinepropattrEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, prop_keys, prop_values, attr_keys, attr_values):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))

        prop_keys_word_emb = self.word_embedding(prop_keys)
        prop_keys_pos_emb = self.position_embedding(torch.arange(0, prop_keys_word_emb.size(2), device=prop_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_keys_word_emb.size(0), prop_keys_word_emb.size(1), 1))
        prop_keys_emb = self.dropout(self.layer_norm(prop_keys_word_emb + prop_keys_pos_emb))

        prop_values_word_emb = self.word_embedding(prop_values)
        prop_values_pos_emb = self.position_embedding(torch.arange(0, prop_values_word_emb.size(2), device=prop_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_values_word_emb.size(0), prop_values_word_emb.size(1), 1))
        prop_values_emb = self.dropout(self.layer_norm(prop_values_word_emb + prop_keys_pos_emb))

        attr_keys_word_emb = self.word_embedding(attr_keys)
        attr_keys_pos_emb = self.position_embedding(torch.arange(0, attr_keys_word_emb.size(2), device=attr_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_keys_word_emb.size(0), attr_keys_word_emb.size(1), 1))
        attr_keys_emb = self.dropout(self.layer_norm(attr_keys_word_emb + attr_keys_pos_emb))

        attr_values_word_emb = self.word_embedding(attr_values)
        attr_values_pos_emb = self.position_embedding(torch.arange(0, attr_values_word_emb.size(2), device=attr_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_values_word_emb.size(0), attr_values_word_emb.size(1), 1))
        attr_values_emb = self.dropout(self.layer_norm(attr_values_word_emb + attr_keys_pos_emb))

        return emb, prop_keys_emb, prop_values_emb, attr_keys_emb, attr_values_emb

class StorylinepropclsEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(StorylinepropclsEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, prop_keys, prop_values, target_words):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))

        prop_keys_word_emb = self.word_embedding(prop_keys)
        prop_keys_pos_emb = self.position_embedding(torch.arange(0, prop_keys_word_emb.size(2), device=prop_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_keys_word_emb.size(0), prop_keys_word_emb.size(1), 1))
        prop_keys_emb = self.dropout(self.layer_norm(prop_keys_word_emb + prop_keys_pos_emb))

        prop_values_word_emb = self.word_embedding(prop_values)
        prop_values_pos_emb = self.position_embedding(torch.arange(0, prop_values_word_emb.size(2), device=prop_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(prop_values_word_emb.size(0), prop_values_word_emb.size(1), 1))
        prop_values_emb = self.dropout(self.layer_norm(prop_values_word_emb + prop_keys_pos_emb))

        target_words_word_emb = self.word_embedding(target_words)
        target_words_pos_emb = self.position_embedding(torch.arange(0, target_words_word_emb.size(2), device=target_words_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(target_words_word_emb.size(0), target_words_word_emb.size(1), 1))
        target_words_emb = self.dropout(self.layer_norm(target_words_word_emb + target_words_pos_emb))

        return emb, prop_keys_emb, prop_values_emb, target_words_emb
