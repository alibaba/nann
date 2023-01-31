#!/usr/bin/python
#****************************************************************#
# ScriptName: design_points_text.py
# Author: @alibaba-inc.com
# Create Date: 2021-08-24 14:22
# Modify Author: @alibaba-inc.com
# Modify Date: 2021-08-24 14:22
# Function:
#***************************************************************#
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
#from onnx_tf.backend import prepare
import onnx
import os
import torch
import argparse

from uer.model_builder import build_model
from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.utils.act_fun import gelu

import numpy as np

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def generate_masks(max_length, batch_size):
    res = []
    for seq_length in range(max_length):
        mask = np.ones((seq_length, seq_length))
        mask = np.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = np.repeat(mask, batch_size, axis=0)
        mask = np.reshape(mask, (batch_size, 1, seq_length, seq_length))
        res.append(torch.from_numpy(mask).float())
    return res

def export_onnx(model, onnx_file_path):

    x = torch.IntTensor([[1]*3])
    seg = torch.IntTensor([[1]*3])
    masks = generate_masks(64, 1)
    mask = masks[3]


    # Export the model
    torch.onnx.export(model,                # model being run
                      (x.cpu(), seg.cpu(), mask ),    # model input (or a tuple for multiple inputs)
                      onnx_file_path,       # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input', 'seg', 'mask' ],   # the model's input names
                      output_names = ['output'], # the model's output names
                      #)
                      dynamic_axes={'input' : {1 : 'seq_length'},    # variable lenght axes
                                    'seg' : {1 : 'seq_length'},
                                    'mask' : {2:'seq_length', 3:'seq_length'},
                                    'output' : {1 : 'seq_length'} })

def top_k_top_p_filtering(logits, top_k:int, top_p:float):
    # type: (Tensor, int, float) -> Tensor
    #prim_min not supported for onnx
    top_k = min(top_k, logits.size(-1))  # Safety check
    minus_inf = torch.tensor(float("-inf"), dtype=torch.float)
    print('input of topk:', logits)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # boolean
        #HACK: TF Select not support, pytorch 1.8.1
        indices_to_remove = indices_to_remove * 9999999
        indices_to_remove = -torch.square(indices_to_remove)
        logits = logits + indices_to_remove
        #END HACK
        #logits[indices_to_remove] = minus_inf
        print('output of topk',logits)

    if top_p > 0.0:
        #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        #cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        ## Remove tokens with cumulative probability above the threshold
        #sorted_indices_to_remove = cumulative_probs > top_p
        ## it seems below act differently in torchscript and results in different dims
        ## Shift the indices to the right to keep also the first token above the threshold
        #sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        #sorted_indices_to_remove[..., 0] = 0

        #indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #logits[indices_to_remove] = -float("Inf")#minus_inf
        #HACK: ONNX-TF GatherNd ScatterNd has bug
        #take the last token of top-p, just like how we do in topk
        probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs_to_remove = sorted_probs[sorted_indices_to_remove]
        threshold_prob = probs_to_remove[0]
        indices_to_remove = probs < threshold_prob
        #logits = torch.where(indices_to_remove, minus_inf, logits)
        indices_to_remove = indices_to_remove * 9999999
        indices_to_remove = -torch.square(indices_to_remove)
        logits = logits + indices_to_remove

        #END HACK
    return logits

class GenerateModel(torch.nn.Module):
    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()

    def forward(self, src, seg, mask):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg, mask)
        output = gelu(self.target.output_layer(output))
        return output

class LoopBodyModel(torch.nn.Module):
    def __init__(self, generated_model, args):
        super(LoopBodyModel, self).__init__()
        self.generated_model = generated_model
        self.args = args
    def forward(self, src_tensor, seg_tensor, mask):#NOTE
        with torch.no_grad():
            outputs = self.generated_model(src_tensor, seg_tensor, mask)
            next_token_logits = outputs[0][-1] / self.args.temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            filtered_logits = top_k_top_p_filtering(next_token_logits, self.args.top_k, self.args.top_p)

            filtered_probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(torch.reshape(filtered_probs, (1, -1)), num_samples=1)
            return next_token


def getargs():
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

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Load Vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    return args


prefix = 'to_onnx_pb/'
onnx_file   = prefix + 'design_points_text.onnx'
pb_file     = prefix + 'design_points_text.pb'
build_args = getargs()
statedict = torch.load(build_args.pretrained_model_path)
model = build_model(build_args)
model = GenerateModel(build_args, model)
model.load_state_dict(statedict, strict=False)
#model = torch.load(prefix+ 'generatemodel_design_points.pt')
#model = LoopBodyModel(model, build_args)
#model = FinalModel(model, build_args)
export_onnx(model, onnx_file)
print('export onnx to %s over' % onnx_file)
#onnx_model = onnx.load(onnx_file)  # load onnx model
#tf_exp = prepare(onnx_model)       # prepare tf representation
#tf_exp.export_graph(pb_file)       # export the model
#print('export pb over')
#print(tf_exp.outputs)
