#!/usr/bin/env python
# -*— coding: utf-8 —*-
import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


def _makedirs(path):
  os.makedirs(path, exist_ok=True)
  return path


def parse_opt():
  parser = argparse.ArgumentParser("single machine demo of NANN",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--job-type", type=str, default="train",
                      choices=["train", "extract_feature", "test", "test_all", "export"],
                      help="job type (train, extract_feature, test, test_all, export)")
  parser.add_argument("--batch-size", type=int, default=800,
                      help="batch size")
  parser.add_argument("--train-epochs", type=int, default=5,
                      help="number of epoch to training")
  parser.add_argument("--emb-dim", type=int, default=32,
                      help="embedding dim")
  parser.add_argument("--print-freq", type=int, default=50,
                      help="print frequency")
  # train
  parser.add_argument("--save-freq", type=int, default=5000,
                      help="save model checkpoint frequency")
  parser.add_argument("--learning-rate", type=float, default=3e-3,
                      help="learning rate")
  parser.add_argument("--weight-decay", type=float, default=1e-4,
                      help="weight decay")
  parser.add_argument("--adv-eps", type=float, default=0.00003,
                      help="eps parameter for adversarial gradient training, set to 0 for disable it")
  parser.add_argument("--adv-weight", type=float, default=1,
                      help="loss weight for the adversarial gradient training")
  parser.add_argument("--num-neg", type=int, default=200,
                      help="number of negative samples")
  # paths
  parser.add_argument("--output-root", type=str, default="./output",
                      help="root directory to save outputs, "
                           "such as tensorboard summary / model checkpoint / item embeddings / HNSW index")
  parser.add_argument("--dataset-dir", type=str, default="./data",
                      help="root directory to the dataset, including train, test, item tfrecords and meta file")
  # test
  parser.add_argument("--hnsw-start-level", type=int, default=2,
                      help="the retrieval start level. all the items in this layer will be scored")
  parser.add_argument("--num-scoring-per-level", type=int, nargs="+", default=[3, 1, 1],
                      help="times scoring from layer 0 to start_level")
  parser.add_argument("--top-k-per-level", type=int, nargs="+", default=[400, 200, 100],
                      help="retrieved top-k from layer 0 to start_level")
  parser.add_argument("--topk-eval", type=int, nargs="+", default=[200, ],
                      help="top-k to test recall")
  parser.add_argument("--num-test-batch", type=int, default=10000,
                      help="number of user to test")

  args = parser.parse_args()

  # other paths
  args.model_save_dir = _makedirs(os.path.join(args.output_root, "model"))
  args.summary_output_dir = _makedirs(os.path.join(args.output_root, "tensorboard_summary"))
  item_embs_dir = _makedirs(os.path.join(args.output_root, "embeddings"))
  args.item_ids_file = os.path.join(item_embs_dir, "item_ids.npy")
  args.item_embs_file = os.path.join(item_embs_dir, "item_embs.npy")
  args.index_dir = _makedirs(os.path.join(args.output_root, "index"))
  args.model_converted_dir = _makedirs(os.path.join(args.output_root, "converted_model"))
  args.log_file = os.path.join(args.output_root, f"{args.job_type}.log")
  args.model_export_dir = _makedirs(os.path.join(args.output_root, "export"))

  # dataset file
  if args.job_type == "train":
    args.dataset_file = os.path.join(args.dataset_dir, "ub_train.tfrecords")
    args.item_raw_features_file = os.path.join(args.dataset_dir, "ub_items.npz")
  elif args.job_type == "extract_feature":
    args.dataset_file = os.path.join(args.dataset_dir, "ub_items.tfrecords")
  else:
    args.dataset_file = os.path.join(args.dataset_dir, "ub_test.tfrecords")

  # parse meta file and add to args
  args.meta_file = os.path.join(args.dataset_dir, "ub_meta.json")
  meta = json.load(open(args.meta_file, "r"))
  args.num_item = meta["num_item"]
  args.num_cate = meta["num_cate"]
  args.num_train_samples = meta["num_train_samples"]
  args.max_seq_length = meta["max_length"]

  return args
