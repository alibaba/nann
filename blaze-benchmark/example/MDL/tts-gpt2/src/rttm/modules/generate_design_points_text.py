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
import numpy as np
#import cv2
import logging
import time
import nvtx
import tensorrt as trt

from uer.utils.act_fun import gelu
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_builder import build_model
from to_onnx_pb import common
import tot


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def trt_version():
    return trt.__version__


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1  #  now only 1 batch
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()
            profile.set_shape('input', (1, 1), (1, 128), (1, 256))
            profile.set_shape('seg', (1, 1), (1, 128), (1, 256))
            profile.set_shape('mask', (1, 1, 1, 1), (1, 1, 128, 128), (1, 1, 256, 256))
            config.add_optimization_profile(profile)
            #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            config.set_flag(trt.BuilderFlag.FP16)
            #config.set_flag(trt.BuilderFlag.INT8)  # need calibration
            #config.set_tactic_sources((1 << int(trt.TacticSource.CUBLAS)) | (1 << int(trt.TacticSource.CUDNN)))
            # By default the workspace size is 0, which means there is no temporary memory
            # i find this doesn't matter
            config.max_workspace_size = 1 << 26 # 1MiB
            #config.max_workspace_size = 1 << 30 # 16MiB
            #config.max_workspace_size = 1 << 33 # 256MiB

            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def top_k_top_p_filtering(logits, top_k, top_p):
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

def generate_masks(max_length, batch_size):
    res = []
    for seq_length in range(max_length+1):
        mask = np.ones((seq_length, seq_length))
        mask = np.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = np.repeat(mask, batch_size, axis=0)
        mask = np.reshape(mask, (batch_size, 1, seq_length, seq_length))
        res.append(torch.from_numpy(mask).cuda().float())
    return res

class GenerateModel(torch.nn.Module):
    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()

    def forward(self, src, seg, pasts = None):
      with torch.no_grad():
          # pasts must be None
          emb = self.embedding(src, seg)  # has position embedding
          output, presents = self.encoder(emb, seg, pasts)  # (seg is not used)
          output = gelu(self.target.output_layer(output))
          return output, presents


class GenerateDesignPointsText(Component):
    """
    require: ["item_title", "item_prop_list"]
    provide: ["text"]
    """

    @classmethod
    def name(cls):
        return "generate_design_points_text"

    def __init__(self, conf_dict):
        # super.__init__(conf_dict)
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Path options.
        parser.add_argument("--pretrained_model_path", default="/home/service/models/output_model_smartphrase_seq256_ner.bin-200000", type=str,
                            help="Path of the pretrained model.")
        parser.add_argument("--vocab_path", default="/home/service/models/google_zh_vocab.txt", type=str,
                            help="Path of the vocabulary file.")
        parser.add_argument("--type_vocab_path", type=str, default=None,
                            help="Path of the preprocessed dataset.")
        parser.add_argument("--config_path", default="/home/service/models/bert_base_config.json", type=str,
                            help="Path of the config file.")

        # Model options.
        parser.add_argument("--seq_length", type=int, default=256,
                            help="Sequence length.")
        parser.add_argument("--sample_nums", type=int, default=1)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--top_p", type=float, default=0.95)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--embedding", choices=["bert", "word", "storylineprop", "storylinepropattr"], default="bert",
                            help="Emebdding type.")
        parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                "cnn", "gatedcnn", "attn", \
                                                "rcnn", "crnn", "gpt", "bilstm", "seq2seq"], \
                                        default="gpt", help="Encoder type.")
        parser.add_argument("--target", choices=["lm", "seq2seq", "storylineprop", "storylinepropattr", "storylinepropattrpict"], default="lm",
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
        model.cuda(gpu_id)

        self.gpu_id = gpu_id

        # Load pretrained model.
        pretrained_model_dict = torch.load(args.pretrained_model_path)
        model.load_state_dict(pretrained_model_dict, strict=False)

        self.model = GenerateModel(args, model)

        # Build tokenizer.
        self.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

        self.args = args
        torch.backends.cudnn.benchmark = True
        self.onnx_file_path = "./to_onnx_pb/designpoints_kvcache.onnx"
        self.engine_file_path = "/home/service/models/design_points_text_kvcache_fp16.trt"

        self.engine = None
        self.context = None
        self.input_names = ['input', 'seg', 'pasts']
        self.output_names = ['output', 'presents']
        self.masks = generate_masks(args.seq_length, 1)

        pass

    def process_internal(self, message):
      with nvtx.annotate("gpt process", color="blue"):
        if self.engine is None:
          self.engine = tot.get_engine(self.onnx_file_path, self.engine_file_path, "./to_onnx_pb/designpoints_kvcache.conf")
          #self.engine = get_engine(self.onnx_file_path, self.engine_file_path)
          self.context = self.engine.create_execution_context()
          #self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        storyline = message.get("storyline")
        prop_list = message.get("item_prop_list")
        prefix_flag = message.get("prefix_flag")

        storyline_list = storyline.split(",")
        random.shuffle(storyline_list)
        src = [self.vocab.get(w) for w in self.tokenizer.tokenize("".join(storyline_list))] + [DOS_ID] + [self.vocab.get(t) for t in self.tokenizer.tokenize(" ".join(pair[0] + ":" + pair[1] for pair in prop_list))]

        src = src[:int(self.args.seq_length / 2 - 1)]
        src += [DOS_ID] + ([self.vocab.get(t) for t in self.tokenizer.tokenize("这是一款")] if prefix_flag else [])

        seg = [1] * len(src)
        start_length = len(src)
        src = [src]
        seg = [seg]

        gpu_id = self.gpu_id

        src_tensor = torch.IntTensor(src).cuda(gpu_id)
        #seg_tensor = torch.IntTensor(seg).cuda(gpu_id)
        pasts = None

        bindings = [None] * (len(self.input_names) + len(self.output_names))
        seg_tensor = torch.tensor([[1]]).cuda(gpu_id)
        for i in range(self.args.seq_length-start_length):
          if i == 0:
              outputs, presents = self.model(src_tensor, seg_tensor, pasts)
          else:
              inputs = [src_tensor, seg_tensor, pasts]
              with nvtx.annotate("gpt loop", color="blue"):
               with nvtx.annotate("gpt trt prepare", color="blue"):
                for j, input_name in enumerate(self.input_names):
                    idx = self.engine.get_binding_index(input_name)
                    dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                    # 设定shape
                    self.context.set_binding_shape(idx, tuple(inputs[j].shape)) # must set when dynamic shape
                    bindings[idx] = int(inputs[j].contiguous().data_ptr())

                outputs = []
                for j, output_name in enumerate(self.output_names):
                    idx = self.engine.get_binding_index(output_name)
                    dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                    shape = tuple(self.context.get_binding_shape(idx))
                    device = torch_device_from_trt(self.engine.get_location(idx))
                    output = torch.ones(size=shape, dtype=dtype, device=device)
                    outputs.append(output)
                    bindings[idx] = int(outputs[j].data_ptr())

               with nvtx.annotate("gpt trt run", color="blue"):
                self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
                outputs, presents = outputs
          pasts = presents


          with nvtx.annotate("gpt post", color="blue"):
            next_token_logits = outputs[0][-1] / self.args.temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            filtered_logits = top_k_top_p_filtering(next_token_logits, self.args.top_k, self.args.top_p)

            filtered_probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            next_token = next_token.int()
            if next_token[0] == EOS_ID:
                break

            src_tensor = torch.cat([src_tensor, next_token.view(1,1)], dim=1)
            #seg_tensor = torch.cat([seg_tensor, torch.IntTensor([[1]]).cuda(gpu_id)], dim=1)

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
        start = time.time()
        if isinstance(message, list):
            for m in message:
                text = self.process_internal(m)
                m.set("text", text)
                logging.info('### after generate_design_points_text messages: %s', m._data)
        else:
            text = self.process_internal(message)
            message.set("text", text)
            logging.info('### after generate_design_points_text messages: %s', message._data)
        print('stage 2 comst ', time.time() - start)


if __name__ == "__main__":
    gt = GenerateDesignPointsText(None)
    count = 40
    print(gt.process_internal({"storyline": "宽松版裙,时尚,优雅,裙子,立领,领口,简约大方,衬托,颈部,曲线,大方,纯,设计,高贵,气质,优质,面料,柔软,轻薄,舒适,透气", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    start = time.time()
    for _ in range(count):
      print(gt.process_internal({"storyline": "宽松版裙,时尚,优雅,裙子,立领,领口,简约大方,衬托,颈部,曲线,大方,纯,设计,高贵,气质,优质,面料,柔软,轻薄,舒适,透气", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    print('cost time ', (time.time() - start)/ count)
    start = time.time()
    for _ in range(count):
      print(gt.process_internal({"storyline": "宽松版裙,时尚,优雅,裙子,立领,领口,简约大方,衬托,颈部,曲线,大方,纯,设计,高贵,气质,优质,面料,柔软,轻薄,舒适,透气", "item_prop_list": [("品牌", "Marisfrolg/玛丝菲尔"),("适用年龄", "35-39周岁"),("材质", "蚕丝"),("尺码", "1/S/36"),("尺码", "2/M/38"),("尺码", "3/L/40"),("尺码", "4/XL/42"),("尺码", "5/XXL/44"),("面料", "其他"),("图案", "纯色"),("风格", "通勤"),("通勤", "简约"),("领型", "其他"),("腰型", "宽松腰"),("衣门襟", "套头"),("颜色分类", "紫色"),("袖型", "其他"),("组合形式", "两件套"),("货号", "A1HF24356A"),("成分含量", "95%以上"),("裙型", "其他"),("年份季节", "2019年夏季"),("袖长", "短袖"),("裙长", "中长裙"),("流行元素/工艺", "系带"),("款式", "其他/other"),("销售渠道类型", "商场同款(线上线下都销售)"),("廓形", "H型"),("材质成分", "桑蚕丝100%")]}))
    print('cost time ', (time.time() - start)/ count)
