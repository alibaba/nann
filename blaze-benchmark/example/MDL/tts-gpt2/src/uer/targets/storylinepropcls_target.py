# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class StorylinepropclsTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(StorylinepropclsTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        #self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        #self.step = 0


    def forward(self, memory_bank, target_words_hidden, target_mask, prop_values_hidden):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """

        '''
        batch_size, target_words_num, hidden_size = target_words_hidden.size()

        memory_bank = memory_bank.unsqueeze(2).repeat(1, 1, target_words_num, 1)
        target_words_hidden = target_words_hidden.unsqueeze(1).repeat(1, target_words_num, 1, 1)

        logits = (memory_bank * target_words_hidden).sum(-1)

        target = torch.cat((torch.eye(target_words_num, device=memory_bank.device)[1 : , :], torch.zeros(1, target_words_num, device=memory_bank.device)), 0).unsqueeze(0).repeat(batch_size, 1, 1)

        loss = self.criterion(logits, target)

        org_target_mask = target_mask.clone()
        col_mask = target_mask.clone().unsqueeze(-1)
        target_mask = target_mask.unsqueeze(1).repeat(1, target_words_num, 1)
        target_mask = target_mask * col_mask

        loss *= target_mask
        denominator = target_mask.sum()
        loss = loss.sum() / denominator

        prob = self.sigmoid(logits)
        _, idx = prob.topk(1, dim=-1)
        idx = idx.squeeze(-1)
        pred = F.one_hot(idx, num_classes=target_words_num)

        denominator = org_target_mask.sum()
        correct = torch.sum(org_target_mask * (idx.int().eq(torch.cat((torch.arange(1, target_words_num, device=memory_bank.device).int(), torch.zeros(1, device=memory_bank.device).int()), 0).unsqueeze(0).repeat(batch_size, 1))).float())
        '''

        batch_size, target_words_num, hidden_size = target_words_hidden.size()
        seq_len = target_words_hidden.size()[1]

        target_words_hidden = torch.cat((target_words_hidden, prop_values_hidden), 1)

        memory_bank = memory_bank.unsqueeze(2).repeat(1, 1, target_words_num * 2, 1)
        target_words_hidden = target_words_hidden.unsqueeze(1).repeat(1, seq_len, 1, 1)

        logits = (memory_bank * target_words_hidden).sum(-1)

        target = torch.cat((torch.arange(1, target_words_num, device=memory_bank.device).int(), torch.zeros(1, device=memory_bank.device).int()), 0).unsqueeze(0).repeat(batch_size, 1)

        loss = self.criterion(logits.view(-1, target_words_num * 2), target.long().view(-1))

        loss = loss.view(batch_size, target_words_num)
        loss *= target_mask
        denominator = target_mask.sum()
        loss = loss.sum() / denominator

        correct = torch.sum(target_mask * (logits.argmax(dim=-1).eq(target)).float())

        '''
        if self.step % 10 == 0 and str(logits.device) == 'cuda:0':
            #print((idx * org_target_mask)[0])
            #_, tgt_idx = target.topk(1, dim=-1)
            #print(idx[0])
            #print(tgt_idx.squeeze(-1)[0])
            print(self.softmax(logits)[0])

        self.step += 1
        '''

        return loss, correct, denominator
