# -*- encoding:utf-8 -*-
import torch
from uer.layers.embeddings import BertEmbedding, WordEmbedding, ReversedEmbedding, FpdgEmbedding, StorylinepropEmbedding, StorylinepropclsEmbedding, StorylinepropattrEmbedding
from uer.encoders.bert_encoder import BertEncoder
from uer.encoders.rnn_encoder import LstmEncoder, GruEncoder
from uer.encoders.birnn_encoder import BilstmEncoder
from uer.encoders.cnn_encoder import CnnEncoder, GatedcnnEncoder
from uer.encoders.attn_encoder import AttnEncoder
from uer.encoders.gpt_encoder import GptEncoder
from uer.encoders.mixed_encoder import RcnnEncoder, CrnnEncoder
from uer.encoders.seq2seq_encoder import Seq2seqEncoder
from uer.targets.bert_target import BertTarget
from uer.targets.lm_target import LmTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.mlm_target import MlmTarget
from uer.targets.nsp_target import NspTarget
from uer.targets.s2s_target import S2sTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.seq2seq_target import Seq2seqTarget
from uer.targets.fpdg_target import FpdgTarget
from uer.targets.vae_target import VaeTarget
from uer.targets.storylineprop_target import StorylinepropTarget
from uer.targets.storylinepropattr_target import StorylinepropattrTarget
from uer.targets.storylinepropattrpict_target import StorylinepropattrpictTarget
from uer.targets.storylinepropattrpictmulti_target import StorylinepropattrpictmultiTarget
from uer.targets.storylinepropcls_target import StorylinepropclsTarget
from uer.subencoders.avg_subencoder import AvgSubencoder
from uer.subencoders.rnn_subencoder import LstmSubencoder
from uer.subencoders.cnn_subencoder import CnnSubencoder
from uer.models.model import Model
from uer.models.fpdg_model import FpdgModel
from uer.models.vae_model import VaeModel
from uer.models.storylineprop_model import StorylinepropModel
from uer.models.storylinepropattr_model import StorylinepropattrModel
from uer.models.storylinepropattrpict_model import StorylinepropattrpictModel
from uer.models.storylinepropattrpictmulti_model import StorylinepropattrpictmultiModel
from uer.models.storylinepropcls_model import StorylinepropclsModel


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](args, len(args.sub_vocab))
    else:
        subencoder = None

    if args.type_vocab_path:
        embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab), len(args.type_vocab))
        target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab), len(args.type_vocab))
    else:
        embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    if args.target == "fpdg":
        model = FpdgModel(args, embedding, encoder, target, subencoder)
    elif args.target == "vae":
        model = VaeModel(args, embedding, encoder, target, subencoder)
    elif args.target == "storylineprop":
        model = StorylinepropModel(args, embedding, encoder, target, subencoder)
    elif args.target == "storylinepropcls":
        model = StorylinepropclsModel(args, embedding, encoder, target, subencoder)
    elif args.target == "storylinepropattr":
        model = StorylinepropattrModel(args, embedding, encoder, target, subencoder)
    elif args.target == "storylinepropattrpict":
        model = StorylinepropattrpictModel(args, embedding, encoder, target, subencoder)
    elif args.target == "storylinepropattrpictmulti":
        model = StorylinepropattrpictmultiModel(args, embedding, encoder, target, subencoder)
    else:
        model = Model(args, embedding, encoder, target, subencoder)

    return model
