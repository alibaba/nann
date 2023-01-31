# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu
from uer.layers.layer_norm import LayerNorm
import torch.nn.functional as F


class StorylinepropattrpictmultiTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(StorylinepropattrpictmultiTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

        self.keys_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.ReLU(),
                                            LayerNorm(args.hidden_size))
        self.values_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.ReLU(),
                                            LayerNorm(args.hidden_size))

    def forward(self, memory_bank, tgt, class_seq_output, attr_keys_target, attr_values_target, attr_masks, all_attr_keys_hidden, all_attr_values_hidden):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """

        batch_size, seq_len, hidden_size = memory_bank.size()
        attr_num = attr_keys_target.size(1)
        all_keys_size = all_attr_keys_hidden.size(0)
        all_values_size = all_attr_values_hidden.size(0)

        keys_predict_hidden, values_predict_hidden = class_seq_output.split(1, dim=2)
        keys_predict_hidden = keys_predict_hidden.view(batch_size, attr_num, -1)
        values_predict_hidden = values_predict_hidden.view(batch_size, attr_num, -1)

        keys_predict_hidden += self.keys_fc(keys_predict_hidden)
        values_predict_hidden += self.values_fc(values_predict_hidden)

        attr_denominator = torch.sum(attr_masks) + 1e-6


        keys_masks_base = 1 - (F.one_hot(attr_keys_target, all_keys_size) * attr_masks.unsqueeze(-1)).sum(1).unsqueeze(1).bool().int()
        keys_labels = F.one_hot(attr_keys_target, all_keys_size)
        keys_masks = keys_labels * attr_masks.unsqueeze(-1) + keys_masks_base

        all_attr_keys_hidden_t = all_attr_keys_hidden.transpose_(0, 1)
        keys_predict_logits = torch.matmul(keys_predict_hidden, all_attr_keys_hidden_t) + (1 - keys_masks) * -10000.0
        keys_predict_log_prob = self.softmax(keys_predict_logits)
        numerator = -torch.sum(keys_predict_log_prob * keys_labels, 2)
        numerator = torch.sum(attr_masks * numerator)
        keys_loss = numerator / attr_denominator
        keys_correct = torch.sum(attr_masks * (keys_predict_logits.argmax(dim=-1).eq(attr_keys_target)).float())


        values_masks_base = 1 - (F.one_hot(attr_values_target, all_values_size) * attr_masks.unsqueeze(-1)).sum(1).unsqueeze(1).bool().int()
        values_labels = F.one_hot(attr_values_target, all_values_size)
        values_masks = values_labels * attr_masks.unsqueeze(-1) + values_masks_base

        all_attr_values_hidden_t = all_attr_values_hidden.transpose_(0, 1)
        values_predict_logits = torch.matmul(values_predict_hidden, all_attr_values_hidden_t) + (1 - values_masks) * -10000.0
        values_predict_log_prob = self.softmax(values_predict_logits)
        numerator = -torch.sum(values_predict_log_prob * values_labels, 2)
        numerator = torch.sum(attr_masks * numerator)
        values_loss = numerator / attr_denominator
        values_correct = torch.sum(attr_masks * (values_predict_logits.argmax(dim=-1).eq(attr_values_target)).float())


        # Language modeling (LM) with full softmax prediction.
        output = self.output_layer(memory_bank)
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

        return loss, correct, denominator, keys_loss, keys_correct, values_loss, values_correct, attr_denominator
