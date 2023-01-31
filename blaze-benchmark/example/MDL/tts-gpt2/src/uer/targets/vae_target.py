# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu
from uer.layers.transformer_qkv import TransformerLayer
from torch.autograd import Variable


class VaeTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(VaeTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length

        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

        self.latent_transformer = TransformerLayer(args)

        self.condition_gate = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                            nn.Sigmoid())

        self.bow_query = Variable(0.02 * torch.randn(self.seq_length, self.hidden_size), requires_grad = True)
        self.bow_transformer = TransformerLayer(args)

    def forward(self, memory_bank, tgt, condition_title_mu, condition_title_logvar, condition_text_mu, condition_text_logvar):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """

        def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
            kld = -0.5 * torch.sum((1 + (recog_logvar - prior_logvar)
                    - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                    - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))).mean(dim=1), 1)
            return kld

        kld_loss = gaussian_kld(condition_text_mu, condition_text_logvar, condition_title_mu, condition_title_logvar).mean()

        batch_size, cluster_center_size, _ = condition_text_mu.size()
        batch_size, seq_length, _ = memory_bank.size()
        z = torch.randn([batch_size, cluster_center_size, self.hidden_size], device=memory_bank.device)
        condition_title_latent = z * torch.exp(0.5 * condition_title_logvar) + condition_title_mu
        condition_text_latent = z * torch.exp(0.5 * condition_text_logvar) + condition_text_mu
        mask = torch.zeros(batch_size, 1, seq_length, cluster_center_size, device=memory_bank.device)
        condition_hidden = self.latent_transformer(memory_bank, condition_text_latent, condition_text_latent, mask)

        condition_gate = self.condition_gate(torch.cat([condition_hidden, memory_bank], -1))
        memory_bank = condition_gate * condition_hidden + (1 - condition_gate) * memory_bank

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

        bow_query = self.bow_query.to(device=condition_text_latent.device).unsqueeze(0).repeat(batch_size, 1, 1)
        mask = torch.zeros(batch_size, 1, self.seq_length, cluster_center_size, device=condition_text_latent.device)
        bow_hidden = self.bow_transformer(bow_query, condition_text_latent, condition_text_latent, mask)
        bow_output = self.output_layer(bow_hidden)
        bow_output = bow_output.contiguous().view(-1, self.vocab_size)
        bow_output = self.softmax(bow_output)
        numerator = -torch.sum(bow_output * one_hot, 1)
        numerator = torch.sum(label_mask * numerator)
        bow_loss = numerator / denominator

        return loss, kld_loss, bow_loss, correct, denominator
