#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/17
# update: 2020/9/17
"""
generate_ost_text.py:
"""

from framework.component import Component

import sys
import os
import torch
import torch.nn.functional as F
import argparse
import random
import datetime
import time
import json
from PIL import Image
import torchvision.transforms as transforms
import urllib.request
import numpy as np
import cv2
import logging

from uer.utils.act_fun import gelu
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_builder import build_model


class GenerateModel(torch.nn.Module):
    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.model = model
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()

    def set_target(self, attr_keys, attr_values):
        attr_keys_word_emb = self.embedding.word_embedding(attr_keys)
        attr_keys_pos_emb = self.embedding.position_embedding(torch.arange(0, attr_keys_word_emb.size(2), device=attr_keys_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_keys_word_emb.size(0), attr_keys_word_emb.size(1), 1))
        attr_keys_emb = self.embedding.layer_norm(attr_keys_word_emb + attr_keys_pos_emb)

        attr_values_word_emb = self.embedding.word_embedding(attr_values)
        attr_values_pos_emb = self.embedding.position_embedding(torch.arange(0, attr_values_word_emb.size(2), device=attr_values_word_emb.device, \
                                          dtype=torch.long).unsqueeze(0).repeat(attr_values_word_emb.size(0), attr_values_word_emb.size(1), 1))
        attr_values_emb = self.embedding.layer_norm(attr_values_word_emb + attr_values_pos_emb)

        # batch_size is always 1
        batch_size, attr_keys_num, attr_len, attr_embed_size = attr_keys_emb.size()
        batch_size, attr_values_num, attr_len, attr_embed_size = attr_values_emb.size()

        attr_keys_hidden = attr_keys_emb.view(-1, attr_len, attr_embed_size)
        attr_keys_masks = (attr_keys.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_keys_masks = attr_keys_masks.float()
        attr_keys_masks = (1.0 - attr_keys_masks) * -10000.0
        attr_keys_hidden, _ = self.model.word_transformer(attr_keys_hidden, attr_keys_masks)
        self.all_attr_keys_hidden = attr_keys_hidden[:, 0, :].squeeze(1).view(batch_size * attr_keys_num, attr_embed_size).cuda()

        attr_values_hidden = attr_values_emb.view(-1, attr_len, attr_embed_size)
        attr_values_masks = (attr_values.view(-1, attr_len) > 0).unsqueeze(1).repeat(1, attr_len, 1).unsqueeze(1)
        attr_values_masks = attr_values_masks.float()
        attr_values_masks = (1.0 - attr_values_masks) * -10000.0
        attr_values_hidden, _ = self.model.word_transformer(attr_values_hidden, attr_values_masks)
        self.all_attr_values_hidden = attr_values_hidden[:, 0, :].squeeze(1).view(batch_size * attr_values_num, attr_embed_size).cuda()
        del self.embedding
        del self.model.embedding
        del self.encoder
        del self.model.encoder
        #del attr_keys_word_emb
        #del attr_keys_pos_emb


    def forward(self, attr_keys_hidden, picts):
      with torch.no_grad():

        batch_size, attr_num, _, attr_embed_size = attr_keys_hidden.size()

        pict_feature = self.model.pict_model(picts)
        pict_feature = pict_feature.transpose(1, 3).transpose(1, 2)

        h = pict_feature.size(1)
        w = pict_feature.size(2)

        pict_feature = self.model.pict_fc(pict_feature.view(batch_size, h * w, -1))
        # if gat module not exists, delete the following two lines
        #pict_gat_masks = torch.zeros(batch_size, 1, h * w, h * w, device=pict_feature.device).float()
        #pict_feature = self.model.pict_gat(pict_feature, pict_gat_masks)

        class_start = self.model.class_start.to(device=attr_keys_hidden.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, attr_num, 1).unsqueeze(2)
        class_seq_hidden = torch.cat((class_start, attr_keys_hidden), 2).view(batch_size * attr_num, 2, -1)
        class_masks = torch.tril(torch.ones((2, 2), dtype=torch.long, device=class_seq_hidden.device)).unsqueeze(0).repeat(batch_size * attr_num, 1, 1).unsqueeze(1)
        class_masks = class_masks.float()
        class_masks = (1.0 - class_masks) * -10000.0
        class_seq_hidden, _ = self.model.class_transformer(class_seq_hidden, class_masks)

        pict_feature = pict_feature.unsqueeze(1).repeat(1, attr_num, 1, 1).view(batch_size * attr_num, h * w, -1)
        class_pict_masks = torch.zeros(batch_size * attr_num, 1, 2, h * w, device=pict_feature.device).float()
        class_seq_output, _ = self.model.class_w_transformer(class_seq_hidden, pict_feature, pict_feature, class_pict_masks)
        class_seq_output = class_seq_output.view(batch_size * attr_num, 2, -1)

        keys_predict_hidden, values_predict_hidden = class_seq_output.split(1, dim=1)
        keys_predict_hidden += self.target.keys_fc(keys_predict_hidden)
        values_predict_hidden += self.target.values_fc(values_predict_hidden)

        class_keys_logits = keys_predict_hidden.view(batch_size * attr_num, -1).matmul(self.all_attr_keys_hidden.transpose(0, 1))
        class_values_logits = values_predict_hidden.view(batch_size * attr_num, -1).matmul(self.all_attr_values_hidden.transpose(0, 1))

        return class_keys_logits, class_values_logits


class GetPiclLabel(Component):
    """
    require: ["item_pict_url"]
    provide: ["attr_list"]
    """

    @classmethod
    def name(cls):
        return "get_pict_label"

    def __init__(self, conf_dict):
        # super.__init__(conf_dict)
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Path options.
        parser.add_argument("--pretrained_model_path", default="/home/service/models/output_model_smartphrase_storylinepropattrpictmulti_seq128_s2s_split_allneg.bin-500000", type=str,
                            help="Path of the pretrained model.")
        parser.add_argument("--vocab_path", default="/home/service/models/google_zh_vocab.txt", type=str,
                            help="Path of the vocabulary file.")
        parser.add_argument("--type_vocab_path", type=str, default=None,
                            help="Path of the preprocessed dataset.")
        parser.add_argument("--config_path", default="/home/service/models/bert_base_config.json", type=str,
                            help="Path of the config file.")

        # Model options.
        parser.add_argument("--seq_length", type=int, default=128,
                            help="Sequence length.")
        parser.add_argument("--sample_nums", type=int, default=1)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--top_p", type=float, default=0.95)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--embedding", choices=["bert", "word", "storylineprop", "storylinepropattr"], default="storylinepropattr",
                            help="Emebdding type.")
        parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                "cnn", "gatedcnn", "attn", \
                                                "rcnn", "crnn", "gpt", "bilstm", "seq2seq"], \
                                        default="seq2seq", help="Encoder type.")
        parser.add_argument("--target", choices=["lm", "seq2seq", "storylineprop", "storylinepropattr", "storylinepropattrpict", "storylinepropattrpictmulti"], default="storylinepropattrpictmulti",
                            help="The training target of the pretraining model.")

        # Subword options.
        parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                            help="Subword feature type.")
        parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                            help="Path of the subword vocabulary file.")
        parser.add_argument("--subencoder_type", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                            help="Subencoder type.")
        parser.add_argument("--gpu_id", default=0, type=int, help="Gpu id of each process.")

        # Tokenizer options.
        parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                            help="Specify the tokenizer."
                                "Original Google BERT uses bert tokenizer on Chinese corpus."
                                "Char tokenizer segments sentences into characters."
                                "Space tokenizer segments sentences into words according to space."
                                )


        args, _ = parser.parse_known_args()

        # Load the hyperparameters from the config file.
        args = load_hyperparam(args)

        # Load Vocabulary
        vocab = Vocab()
        vocab.load(args.vocab_path)
        args.vocab = vocab

        self.vocab = vocab

        # Build bert model.
        model = build_model(args)
        gpu_id = args.gpu_id
        torch.cuda.set_device(gpu_id)

        self.gpu_id = gpu_id

        # Load pretrained model.
        pretrained_model_dict = torch.load(args.pretrained_model_path)
        model.load_state_dict(pretrained_model_dict, strict=False)

        model.target.cuda(gpu_id)
        model.pict_model.cuda(gpu_id)
        model.pict_fc.cuda(gpu_id)
        model.pict_model.cuda(gpu_id)
        model.class_transformer.cuda(gpu_id)
        model.class_w_transformer.cuda(gpu_id)
        #model.cuda(gpu_id)

        self.model = GenerateModel(args, model)

        # Build tokenizer.
        self.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.args = args


        ATTR_LEN = 10
        attr_keys = []
        self.keys = []
        with open('/home/service/models/attr_all_key.txt') as file:
            for line in file:
                key = line.strip()
                if not key:
                    continue
                self.keys.append(key)
                attr_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(key)]

                if len(attr_key) >= ATTR_LEN:
                    attr_key = attr_key[:ATTR_LEN]
                else:
                    while len(attr_key) != ATTR_LEN:
                        attr_key.append(PAD_ID)

                attr_keys.append(attr_key)

        attr_values = []
        self.values = []
        with open('/home/service/models/attr_all_value.txt') as file:
            for line in file:
                value = line.strip()
                if not value:
                    continue
                self.values.append(value)
                attr_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(value)]

                if len(attr_value) >= ATTR_LEN:
                    attr_value = attr_value[:ATTR_LEN]
                else:
                    while len(attr_value) != ATTR_LEN:
                        attr_value.append(PAD_ID)

                attr_values.append(attr_value)

        attr_keys_tensor = torch.LongTensor([attr_keys])#.cuda(gpu_id)
        attr_values_tensor = torch.LongTensor([attr_values])#.cuda(gpu_id)
        #print('memory_allocated:', torch.cuda.memory_allocated())
        #print('max memory_allocated:', torch.cuda.max_memory_allocated())
        #print('max mem cached:', torch.cuda.max_memory_cached())
        self.model.set_target(attr_keys_tensor, attr_values_tensor)
        torch.cuda.empty_cache()
        #print('memory_allocated:', torch.cuda.memory_allocated())
        #print('max memory_allocated:', torch.cuda.max_memory_allocated())
        #print('max mem cached:', torch.cuda.max_memory_cached())
        #time.sleep(5)


        pass

    def process_internal(self, message):
        assigned_pict_url = message.get("assigned_pict_url")
        pict_url = message.get("item_pict_url")
        if assigned_pict_url is not None:
            logging.info("### assigned_pict_url is not None, use assigned_pict_url: %s", assigned_pict_url)
            pict_url = assigned_pict_url
        k = int(message.get("top_k"))

        try:
            attempts = 0
            success = False
            while attempts < 3 and not success:
                img_content = urllib.request.urlopen(pict_url, timeout=3).read()
                if img_content is not None:
                    success = True
                else:
                    attempts += 1

            if img_content is not None:
                img_data = np.asarray(bytearray(img_content), dtype='uint8')
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img = img.resize((224, 224))
                pict = self.train_transform(img)

        except Exception as e:
            logging.error(e)
            return False
        '''
        pict = Image.open("O1CN010ZGfPs1MTGzBUw3dS_!!0-item_pic.jpg")
        pict = self.train_transform(pict)
        '''

        gpu_id = self.gpu_id

        pict = pict.unsqueeze(0).cuda(gpu_id)

        attr_keys_hidden = self.model.all_attr_keys_hidden[0].view(1, 1, 1, -1)

        top_k_keys_idx = None
        top_k_keys_logit = None
        top_1_values_idx = None
        top_1_values_logit = None

        for i in range(2):
            class_keys_logits, class_values_logits = self.model(attr_keys_hidden, pict)
            if i == 0:
                top_k_keys_logit, top_k_keys_idx = class_keys_logits.topk(k)
                top_k_keys_idx = top_k_keys_idx.view(k)
                top_k_keys_logit = top_k_keys_logit.view(k)
                attr_keys_hidden = self.model.all_attr_keys_hidden[top_k_keys_idx].view(1, k, 1, -1)
            top_1_values_logit, top_1_values_idx = class_values_logits.topk(1)
            top_1_values_idx = top_1_values_idx.unsqueeze(-1)
            top_1_values_logit = top_1_values_logit.unsqueeze(-1)

        attr_list = []
        for i in range(k):
            key_idx = top_k_keys_idx[i].item()
            key_logit = top_k_keys_logit[i].item()
            value_idx = top_1_values_idx[i].item()
            value_logit = top_1_values_logit[i].item()
            attr_list.append({'attr_key' : self.keys[key_idx], 'attr_key_score' : key_logit, 'attr_value' : self.values[value_idx], 'attr_value_score' : value_logit})

        return attr_list


    def process(self, message):
        if isinstance(message, list):
            for m in message:
                attr_list = self.process_internal(m)
                m.set("attr_list", attr_list)
                logging.info('### after get_pict_label messages: %s', m._data)
        else:
            attr_list = self.process_internal(message)
            message.set("attr_list", attr_list)
            logging.info('### after get_pict_label messages: %s', message._data)

if __name__ == "__main__":
    gpl = GetPiclLabel(None)
    a = time.time()
    for _ in range(200):
      print(gpl.process_internal({"item_title": "玛丝菲尔真丝连衣裙女2019夏季紫色时尚气质中长款裙子", "item_pict_url": "https://img.alicdn.com/imgextra/i3/906821435/O1CN01bmowyo1MTGzHM9Mwf_!!0-item_pic.jpg", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")], "top_k": 15}))
    b = time.time()
    print((b-a)/200)

