import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.logger import get_logger

logger = get_logger("nann.data_preprocess")


def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)


def parse_option():
  parser = argparse.ArgumentParser("preprocess raw data_provider to tfrecords",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--max-length", type=int, default=50,
                      help="max length of sequence feature")
  parser.add_argument("--train-min-length", type=int, default=10,
                      help="min length of sequence feature")
  parser.add_argument("--test-min-length", type=int, default=7,
                      help="min length of sequence feature in the test set")
  parser.add_argument("--num-validate-user", type=int, default=10000,
                      help="number of validate user")
  parser.add_argument("--num-test-user", type=int, default=10000,
                      help="number of test user")
  parser.add_argument("-i", "--input", type=str, default="data_provider/UserBehavior.csv",
                      help="path to the input UserBehavior dataset")
  parser.add_argument("-o", "--output-folder", type=str, default="data_provider",
                      help="root director for output")

  return parser.parse_args()


def _int64_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(sample):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  feature = {
    'item_ids': _int64_feature(sample['item_ids']),
    'cate_ids': _int64_feature(sample['cate_ids']),
    'gt_item_id': _int64_feature(sample['gt_item_id']),
    'gt_cate_id': _int64_feature(sample['gt_cate_id']),
    'weight_tag': _float_feature(sample['weight_tag']),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def write_tfrecord(samples, filename):
  with tf.io.TFRecordWriter(filename) as writer:
    for i in trange(len(samples), desc=f"writing tfrecords", ncols=80):
      writer.write(serialize_example(samples[i]))


def main(args):
  set_seed(seed=0)

  item_cate_map = {}
  weight_tag = defaultdict(int)
  user_behavior = defaultdict(list)
  user_behavior_timestamp = defaultdict(list)
  logger.info(f"loading data_provider from {args.input}")
  for line in tqdm(open(args.input, "r"), total=100150807, desc="loading data_provider", ncols=80):
    all_terms = line.split(',')
    user = all_terms[0].strip()
    item_id = all_terms[1].strip()
    cate = all_terms[2].strip()
    # behav = all_terms[3].strip()
    timestamp = all_terms[4].strip()

    item_cate_map[item_id] = cate
    user_behavior[user].append(item_id)
    user_behavior_timestamp[user].append(timestamp)
    weight_tag[item_id] += 1

  logger.info("converting count to probability")
  s = sum(weight_tag.values()) * 1.0
  for item_id in weight_tag.keys():
    weight_tag[item_id] /= s

  logger.info(f"sorting user behaviors by timestamp")
  for user in tqdm(user_behavior.keys(), desc="sorting user behavior", ncols=80):
    idx = np.argsort(user_behavior_timestamp[user])
    user_behavior[user] = np.array(user_behavior[user])[idx]

  # avoid 0, which is used as ``missed''
  item_iid_map = {item_id: i + 1 for (i, item_id) in enumerate(item_cate_map.keys())}
  cate_cid_map = {cate: i + 1 for (i, cate) in enumerate(set(item_cate_map.values()))}

  train_users = set([k for k, v in user_behavior.items() if len(v) > args.test_min_length])
  test_users = random.sample(train_users, args.num_test_user)
  train_users -= set(test_users)
  validate_users = random.sample(train_users, args.num_validate_user)
  train_users -= set(validate_users)

  def generate_sample(_behaviors, _idx_gt):
    idx_start = max(0, _idx_gt - args.max_length)
    _item_ids = _behaviors[idx_start: _idx_gt]
    _cate_ids = [item_cate_map[item] for item in _item_ids]
    _gt_item_id = _behaviors[_idx_gt]
    _gt_cate_id = item_cate_map[_gt_item_id]

    # convert to iid and last padding 0 to args.max_length
    padding = [0] * (args.max_length - len(_item_ids))
    return {
      "item_ids": [item_iid_map[_item_id] for _item_id in _item_ids] + padding,
      "cate_ids": [cate_cid_map[_cate_id] for _cate_id in _cate_ids] + padding,
      "gt_item_id": item_iid_map[_gt_item_id],
      "gt_cate_id": cate_cid_map[_gt_cate_id],
      "weight_tag": weight_tag[_gt_item_id],
    }

  os.makedirs(args.output_folder, exist_ok=True)

  # write train samples
  train_output = os.path.join(args.output_folder, "ub_train.tfrecords")
  logger.info(f"generating train samples and write to {train_output}")
  train_samples = []
  for user in tqdm(train_users, desc="generating train samples", ncols=80):
    behaviors = user_behavior[user]
    for idx_gt in range(args.train_min_length, len(behaviors) - 1):
      train_samples.append(generate_sample(behaviors, idx_gt))
  random.shuffle(train_samples)
  write_tfrecord(train_samples, train_output)

  # write test samples
  test_output = os.path.join(args.output_folder, "ub_test.tfrecords")
  logger.info(f"generating test samples and write to {test_output}")
  test_samples = []
  for user in tqdm(test_users, desc="generating test samples", ncols=80):
    behaviors = user_behavior[user]
    idx_gt = args.test_min_length + math.floor((len(behaviors) - args.test_min_length) / 2)
    test_samples.append(generate_sample(behaviors, idx_gt))
  write_tfrecord(test_samples, test_output)

  # write validate samples
  validate_output = os.path.join(args.output_folder, "ub_validate.tfrecords")
  logger.info(f"generating validate samples and write to {validate_output}")
  validate_samples = []
  for user in tqdm(validate_users, desc="generating validate samples", ncols=80):
    behaviors = user_behavior[user]
    idx_gt = args.test_min_length + math.floor((len(behaviors) - args.test_min_length) / 2)
    validate_samples.append(generate_sample(behaviors, idx_gt))
  write_tfrecord(validate_samples, validate_output)

  # write item samples for extract feature of all items
  items_output = os.path.join(args.output_folder, "ub_items.tfrecords")
  logger.info(f"generating item samples and write to {items_output}")
  item_samples = []
  for item_id, iid in item_iid_map.items():
    item_samples.append({
      "item_ids": [],
      "cate_ids": [],
      "gt_item_id": iid,
      "gt_cate_id": cate_cid_map[item_cate_map[item_id]],
      "weight_tag": weight_tag[item_id],
    })
  write_tfrecord(item_samples, items_output)

  # write item raw features to npy file
  item_raw_features_output = os.path.join(args.output_folder, "ub_items.npz")
  logger.info(f"caching item raw feature to {item_raw_features_output}")
  item_ids, cate_ids, weight_tags = [], [], []
  for item_sample in item_samples:
    item_ids.append(item_sample["gt_item_id"])
    cate_ids.append(item_sample["gt_cate_id"])
    weight_tags.append(item_sample["weight_tag"])
  np.savez(item_raw_features_output, item_id=item_ids, cate_id=cate_ids, weight_tag=weight_tags)

  meta_output = os.path.join(args.output_folder, "ub_meta.json")
  logger.info(f"writing meta to {meta_output}")
  meta = {
    "num_item": len(item_iid_map.values()),
    "num_cate": len(cate_cid_map.values()),
    "num_train_samples": len(train_samples),
    "num_train_user": len(train_users),
    "num_test_user": len(test_users),
    "num_validate_user": len(validate_users),
    "max_length": args.max_length,
    "train_min_length": args.train_min_length,
    "test_min_length": args.test_min_length,
  }
  json.dump(meta, open(meta_output, "w"))


if __name__ == '__main__':
  main(parse_option())
