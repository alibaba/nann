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
import json
from PIL import Image
import torchvision.transforms as transforms
import urllib.request
import numpy as np
import cv2
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from uer.utils.act_fun import gelu
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_builder import build_model
import tensorrt as trt
import nvtx

import tot

#tot.set_trt_log_level(trt.Logger.VERBOSE)

def top_k_top_p_filtering(logits, top_k, top_p):
    assert(logits.dim()==1)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits

def check_invalid(input):
    isnan = torch.isnan(input)
    has_nan = isnan.any()
    has_inf = torch.isinf(input).any()
    has_min = (input < 0).any()
    if has_nan:
        logging.info('----------HAS NAN-----------')
        print(input)
    if has_inf:
        logging.info('-------HAS INF-----')
        print(input)
    if has_min:
        logging.info('-------HAS MINUS-----')
        print(input)
    return has_nan or has_inf or has_min

def check_nan(input):
    isnan = torch.isnan(input)
    has_nan = isnan.any()
    if has_nan:
        logging.info('----------HAS NAN-----------')
        print(input)
    return has_nan

def check_inf(input):
    isinf = torch.isinf(input)
    has_inf = isinf.any()
    if has_inf:
        logging.info('----------HAS INF-----------')
        print(input)
    return has_inf


class GenerateModel(torch.nn.Module):
    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.model = model
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()
        #self.emb_stream = torch.cuda.Stream()
        #self.executor = ThreadPoolExecutor(max_workers=2)
        self.layers_num = args.layers_num

    #def emb_func(self, emb, seg, masks, pasts):
    #    if pasts is not None:
    #        emb = emb[:, -1:, :]
    #        seg = seg[:, -1:]
    #    with torch.cuda.stream(self.emb_stream):
    #    #with torch.cuda.amp.autocast(): #makes it slower, dont use
    #        output, presents = self.encoder(emb, seg, masks, pasts) #about 0.012~0.018 s, performance bottleneck
    #        return output, presents

    def update_pasts(self, pasts, presents):
        if pasts is None:
            pasts = presents
        else:# batch, n_layers, 2, heads, seq_len, feature
            pasts = torch.cat((pasts, presents), dim=-2)
        return pasts


    def forward(self, src, seg, masks, prop_keys, prop_values, attr_keys, attr_values, pict_feature, pasts, output_history, prop_keys_hidden, prop_values_hidden):
      with torch.no_grad():

        emb, prop_keys_emb, prop_values_emb, _, _ = self.embedding(src, seg, prop_keys, prop_values, prop_keys, prop_values)

        batch_size, prop_num, prop_len, prop_embed_size = prop_keys_emb.size()
        seq_len = src.size(-1)

        #prop_keys_hidden = prop_keys_emb.view(-1, prop_len, prop_embed_size)
        #prop_keys_masks = (prop_keys.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        #prop_keys_masks = prop_keys_masks.float()
        #prop_keys_masks = (1.0 - prop_keys_masks) * -10000.0
        #prop_keys_hidden, presents = self.model.word_transformer(prop_keys_hidden, prop_keys_masks, pasts['prop_keys_hidden'])
        #prop_keys_hidden = prop_keys_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        #prop_values_hidden = prop_values_emb.view(-1, prop_len, prop_embed_size)
        #prop_values_masks = (prop_values.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        #prop_values_masks = prop_values_masks.float()
        #prop_values_masks = (1.0 - prop_values_masks) * -10000.0
        #prop_values_hidden, presents = self.model.word_transformer(prop_values_hidden, prop_values_masks, pasts['prop_values_hidden'])
        #prop_values_hidden = prop_values_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)


        prop_masks = (prop_keys.sum(dim=-1) > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        prop_masks = prop_masks.float()
        prop_masks = (1.0 - prop_masks) * -10000.0

        #output, presents = self.emb_func(emb, seg, masks, pasts)
        if pasts is not None:
            emb = emb[:, -1:, :]
            seg = seg[:, -1:]
        output, presents = self.encoder(emb, seg, masks, pasts) #about 0.012~0.018 s, performance bottleneck
        #output, presents = future1.result() #output is batch, seq, hiddensize
        if pasts is None:
            pass
            #pasts['encoder_output_history'] = output
        else:
            output = torch.cat([output_history, output], dim=1)
            #pasts['encoder_output_history'] = output
        output_history = output
        pasts = self.update_pasts(pasts, presents)
        prop_output, presents = self.model.prop_w_transformer(output, prop_keys_hidden, prop_values_hidden, prop_masks, None) # Transformer_qkv about 0.001s

        prop_condition_gate = self.model.prop_condition_gate(torch.cat([output, prop_output], -1)) #0.0001s
        output = prop_condition_gate * output + (1 - prop_condition_gate) * prop_output

        h = pict_feature.size(1)
        w = pict_feature.size(2)
        pict_feature = self.model.pict_fc(pict_feature.view(batch_size, h * w, -1))# actually can extract, but due to batchsize and h, just 0.0002s
        pict_masks = torch.zeros(batch_size, 1, seq_len, h * w, device=pict_feature.device).float()
        pict_output, presents = self.model.pict_w_transformer(output, pict_feature, pict_feature, pict_masks, None) # about 0.001s

        pict_condition_gate = self.model.pict_condition_gate(torch.cat([output, pict_output], -1))
        output = pict_condition_gate * output + (1 - pict_condition_gate) * pict_output

        output = self.target.output_layer(output) #linear
        # output must be full output
        return output, pasts, output_history


class GenerateStorylineWithPict(Component):
    """
    require: ["item_title", "item_pict_url", "item_prop_list"]
    provide: ["text"]
    """

    @classmethod
    def name(cls):
        return "generate_storyline_with_pict"

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
        parser.add_argument("--target", choices=["lm", "seq2seq", "storylineprop", "storylinepropattr", "storylinepropattrpict"], default="storylinepropattrpict",
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
        print(args)

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
        model.eval()
        #现象是如果不加环境变量CUDA_LAUNCH_BLOCKING，多进程的情况下会偶尔出现tensorrt输出全部为NaN
        #而且是一句话中for循环已经循环几次了，前几次输出的字都是正确的，有意义的字
        #但某次就出现NaN。
        #进程0和1都有可能会出错的进程因此并不是其中一个的参数错了。
        #这种现象以前就有，也是出现在storyline中，只不过频率没有现在这么高
        #NaN的产生是由于计算标准差时开根号的tensor全是负数，原因是前面的乘法乘错了，乘了一个负数。一般是onnx中的
        #2639 乘2640 的错，但也出现过是其后面的某处出错。
        #这个错误有可能是TRT8.0的bug，也有可能与TRT无关毕竟之前的时候storyline没有使用TRT
        #此问题确实难以定位，未来其解决方法，应该是本服务多进程共用一个模型，只占用一份资源，不论使用high_service还是其他
        model.cuda(gpu_id)

        self.gpu_id = gpu_id

        # Load pretrained model.
        pretrained_model_dict = torch.load(args.pretrained_model_path)
        model.load_state_dict(pretrained_model_dict, strict=False)

        self.model = GenerateModel(args, model)
        self.pict_model = model.pict_model
        # Build tokenizer.
        self.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.args = args

        self.tril_matrix = torch.tril(torch.ones((args.seq_length, args.seq_length), dtype=torch.long))
        #self.executor = ThreadPoolExecutor(max_workers=2)

        torch.backends.cudnn.benchmark = True
        self.onnx_file_path = "./to_onnx_pb/story_line_kvcache.onnx"
        self.engine_file_path = "/home/service/models/story_line_kvcache_fp16.trt"
        self.trt_config_path = "./to_onnx_pb/story_line_kvcache.conf"
        self.engine = None
        self.context = None
        self.input_names = tot.get_input_names(self.trt_config_path)
        self.input_names.remove('mask')
        self.input_names.remove('prop_values')
        self.output_names = ['output', 'presents', 'output_history_out']


        pass

    def download_and_process_image(self, pict_url, gpu_id):
        try:
            attempts = 0
            success = False
            while attempts < 3 and not success:
                img_content = urllib.request.urlopen(pict_url, timeout=3).read() #0.35s
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
                pict = pict.unsqueeze(0).cuda(gpu_id)
                pict_feature = self.pict_model(pict)  # 20ms
                pict_feature = pict_feature.transpose(1, 3).transpose(1, 2)
                return True, pict_feature

        except Exception as e:
            logging.error(e)
            raise Exception("error processing the image, please check its format.")  #  a user friendly error_msg
            return False, None

    def process_prop_k_v(self, prop_keys, prop_values, prop_keys_emb, prop_values_emb):
        batch_size, prop_num, prop_len, prop_embed_size = prop_keys_emb.size()
        prop_keys_hidden = prop_keys_emb.view(-1, prop_len, prop_embed_size)
        prop_keys_masks = (prop_keys.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_keys_masks = prop_keys_masks.float()
        prop_keys_masks = (1.0 - prop_keys_masks) * -10000.0
        prop_keys_hidden, presents = self.model.model.word_transformer(prop_keys_hidden, prop_keys_masks, None)
        prop_keys_hidden = prop_keys_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)

        prop_values_hidden = prop_values_emb.view(-1, prop_len, prop_embed_size)
        prop_values_masks = (prop_values.view(-1, prop_len) > 0).unsqueeze(1).repeat(1, prop_len, 1).unsqueeze(1)
        prop_values_masks = prop_values_masks.float()
        prop_values_masks = (1.0 - prop_values_masks) * -10000.0
        prop_values_hidden, presents = self.model.model.word_transformer(prop_values_hidden, prop_values_masks, None)
        prop_values_hidden = prop_values_hidden[:, 0, :].squeeze(1).view(batch_size, prop_num, prop_embed_size)
        #about 0.002s
        return prop_keys_hidden, prop_values_hidden

    def process_internal(self, message):
      if self.engine is None:
        self.engine = tot.get_engine(self.onnx_file_path, self.engine_file_path, self.trt_config_path)
        self.context = self.engine.create_execution_context()

      with nvtx.annotate("stage1 process", color="red"):
        title = message.get("item_title")
        pict_url = message.get("item_pict_url")
        prop_list = message.get("item_prop_list")

        download_succ, pict_feature = self.download_and_process_image(pict_url, self.gpu_id) #0.5s

        #2ms
        src = [self.vocab.get(w) for w in self.tokenizer.tokenize(title)] + [DOS_ID]

        PROP_LEN = 10
        PROP_NUM = 50
        prop_keys = []
        prop_values = []
        for pair in prop_list:
            prop_key = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair[0])]
            prop_value = [CLS_ID] + [self.vocab.get(w) for w in self.tokenizer.tokenize(pair[1])]

            if len(prop_key) >= PROP_LEN:
                prop_key = prop_key[:PROP_LEN]
            else:
                while len(prop_key) != PROP_LEN:
                    prop_key.append(PAD_ID)
            if len(prop_value) >= PROP_LEN:
                prop_value = prop_value[:PROP_LEN]
            else:
                while len(prop_value) != PROP_LEN:
                    prop_value.append(PAD_ID)

            prop_keys.append(prop_key)
            prop_values.append(prop_value)

        if len(prop_keys) >= PROP_NUM:
            prop_keys = prop_keys[:PROP_NUM]
            prop_values = prop_values[:PROP_NUM]
        else:
            while len(prop_keys) != PROP_NUM:
                prop_keys.append([PAD_ID] * PROP_LEN)
                prop_values.append([PAD_ID] * PROP_LEN)


        '''
        pict = Image.open("O1CN010ZGfPs1MTGzBUw3dS_!!0-item_pic.jpg")
        pict = self.train_transform(pict)
        '''

        src = src[:int(self.args.seq_length / 4 - 1)]
        src += [DOS_ID]
        src_len = len(src)

        tgt_start = []

        tgt_len = len(tgt_start)
        src += tgt_start
        seg = [1] * len(src)
        start_length = len(src)
        src = [src]
        seg = [seg]
        prop_keys = [prop_keys]
        prop_values = [prop_values]

        gpu_id = self.gpu_id

        #0.15ms
        src_tensor = torch.LongTensor(src).cuda(gpu_id)
        seg_tensor = torch.LongTensor(seg).cuda(gpu_id)
        prop_keys_tensor = torch.LongTensor(prop_keys).cuda(gpu_id)
        prop_values_tensor = torch.LongTensor(prop_values).cuda(gpu_id)
        #download_succ, pict_feature = future_pict.result()
        if not download_succ:
            return False
        #pasts = {'seq2seq_encoder':None, 'encoder_output_history':None}
        pasts = None
        output_history = None  # because there is a linear layer in self.model that must use the full output, we have to know the full output history within self.model
        # output_history is not history of output of self.model, but output of self.model.encoder
        emb, prop_keys_emb, prop_values_emb, _, _ = self.model.embedding(src_tensor, seg_tensor, prop_keys_tensor, prop_values_tensor, prop_keys_tensor, prop_values_tensor)
        prop_keys_hidden, prop_values_hidden = self.process_prop_k_v(prop_keys_tensor, prop_values_tensor, prop_keys_emb, prop_values_emb)
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        for i in range(self.args.seq_length-start_length):
            if i == 0:
                mask = torch.zeros(src_len + tgt_len, src_len + tgt_len, dtype=torch.long)
                mask[:, :src_len].fill_(1)
                second_st = src_len
                second_end = src_len + tgt_len
                mask[second_st:second_end, second_st:second_end].copy_(
                    self.tril_matrix[:second_end-second_st, :second_end-second_st])
                mask = mask.view(1, 1, second_end, second_end)
                mask = (1.0 - mask) * -10000
                mask = mask.cuda(gpu_id)

                outputs, presents, output_history = self.model(src_tensor, seg_tensor, mask, prop_keys_tensor, prop_values_tensor, None, None, pict_feature, pasts, output_history, prop_keys_hidden, prop_values_hidden)
            else:
                #inputs = [src_tensor.int(), seg_tensor.int(), mask, prop_keys_tensor.int(), prop_values_tensor.int(), pict_feature,
                #          pasts, output_history, prop_keys_hidden, prop_values_hidden]
                inputs = [src_tensor.int(), seg_tensor.int(), prop_keys_tensor.int(), pict_feature,
                          pasts, output_history, prop_keys_hidden, prop_values_hidden]
                for j, input_name in enumerate(self.input_names):
                    idx = self.engine.get_binding_index(input_name)
                    dtype = tot.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                    # 设定shape
                    self.context.set_binding_shape(idx, tuple(inputs[j].shape)) # must set when dynamic shape
                    bindings[idx] = int(inputs[j].contiguous().data_ptr())

                output_list = []
                for j, output_name in enumerate(self.output_names):
                    idx = self.engine.get_binding_index(output_name)
                    dtype = tot.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                    shape = tuple(self.context.get_binding_shape(idx))
                    device = tot.torch_device_from_trt(self.engine.get_location(idx))
                    output = torch.ones(size=shape, dtype=dtype, device=device)
                    output_list.append(output)
                    bindings[idx] = int(output_list[j].data_ptr())
                self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
                output, presents, output_history = output_list
                if check_nan(presents) or check_nan(output_history):
                    raise Exception("imposible")
                if check_nan(output):
                    print('AHA')

                outputs = output
                #print(output[0][-1])


            pasts = presents
            next_token_logits = outputs[0][-1] / self.args.temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            filtered_logits = top_k_top_p_filtering(next_token_logits, self.args.top_k, self.args.top_p)

            filtered_probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            if next_token[0] == EOS_ID:
                break

            src_tensor = torch.cat([src_tensor, next_token.view(1,1)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).cuda(gpu_id)], dim=1)
            tgt_len += 1

        output_char_list = []
        for idx, token_id in enumerate(src_tensor[0]):
            char = self.vocab.i2w[token_id].replace('##', '')
            if token_id == EOS_ID:
                break
            if token_id == DOS_ID:
                #char = '\t'
                output_char_list = []
                continue
            if token_id == UNK_ID:
                continue
            char = char.replace('##', '')
            output_char_list.append(char)

        output_sentence = ''
        for idx, char in enumerate(output_char_list):
            output_sentence += char

        return output_sentence


    def process(self, message):
      with torch.no_grad():
        start = time.time()
        if isinstance(message, list):
            for m in message:
                text = self.process_internal(m)
                m.set("storyline", text)
                logging.info('### after generate_storyline_with_pict messages: %s', m._data)
        else:
            text = self.process_internal(message)
            message.set("storyline", text)
            logging.info('### after generate_storyline_with_pict messages: %s', message._data)

if __name__ == "__main__":
    gt = GenerateStorylineWithPict(None)
    print(gt.process_internal({"item_title": "玛丝菲尔真丝连衣裙女2019夏季紫色时尚气质中长款裙子", "item_pict_url": "https://246950.oss-cn-hangzhou-zmf.aliyuncs.com/O1CN01bmowyo1MTGzHM9Mwf_!!0-item_pic.jpeg", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    start = time.time()
    count = 20
    for _ in range(count):
      print(gt.process_internal({"item_title": "玛丝菲尔真丝连衣裙女2019夏季紫色时尚气质中长款裙子", "item_pict_url": "https://246950.oss-cn-hangzhou-zmf.aliyuncs.com/O1CN01bmowyo1MTGzHM9Mwf_!!0-item_pic.jpeg", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    print('cost time ', (time.time() - start) / count)
    start = time.time()
    count = 20
    for _ in range(count):
      print(gt.process_internal({"item_title": "玛丝菲尔真丝连衣裙女2019夏季紫色时尚气质中长款裙子", "item_pict_url": "https://246950.oss-cn-hangzhou-zmf.aliyuncs.com/O1CN01bmowyo1MTGzHM9Mwf_!!0-item_pic.jpeg", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    print('cost time ', (time.time() - start) / count)
