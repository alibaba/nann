# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu
from uer.layers.transformer import TransformerLayer


class FpdgTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size, type_vocab_size):
        super(FpdgTarget, self).__init__()
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = args.hidden_size

        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.type_output_layer = nn.Linear(self.hidden_size, self.type_vocab_size)

        self.hidden_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.transformer = TransformerLayer(args)

    def forward(self, memory_bank, tgt, tgt_type, type_emb, mask):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """
        hidden = torch.cat((memory_bank, type_emb), -1)
        hidden = self.hidden_layer(hidden)
        hidden = self.transformer(hidden, mask)


        # Language modeling (LM) with full softmax prediction.
        output = self.output_layer(hidden)
        output = output.contiguous().view(-1, self.vocab_size)
        # Full probability distribution.
        output = self.softmax(output)

        tgt = tgt.contiguous().view(-1,1)
        label_mask = (tgt > 0).float().to(torch.device(output.device))
        one_hot = torch.zeros(label_mask.size(0),  self.vocab_size). \
           to(torch.device(output.device)). \
           scatter_(1, tgt, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt = tgt.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        correct = torch.sum(label_mask * (output.argmax(dim=-1).eq(tgt)).float())


        # Language modeling (LM) with full softmax prediction.
        type_output = self.type_output_layer(hidden)
        type_output = type_output.contiguous().view(-1, self.type_vocab_size)
        # Full probability distribution.
        type_output = self.softmax(type_output)

        tgt_type = tgt_type.contiguous().view(-1,1)
        label_mask = (tgt_type > 0).float().to(torch.device(type_output.device))
        one_hot = torch.zeros(label_mask.size(0),  self.type_vocab_size). \
           to(torch.device(type_output.device)). \
           scatter_(1, tgt_type, 1.0)

        type_numerator = -torch.sum(type_output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt_type = tgt_type.contiguous().view(-1)
        type_numerator = torch.sum(label_mask * type_numerator)
        type_denominator = torch.sum(label_mask) + 1e-6
        type_loss = type_numerator / type_denominator
        type_correct = torch.sum(label_mask * (type_output.argmax(dim=-1).eq(tgt_type)).float())


        return loss, correct, denominator, type_loss, type_correct, type_denominator
