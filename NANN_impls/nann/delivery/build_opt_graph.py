#!/usr/bin/env python
# -*— coding: utf-8 —*-

import argparse
import math
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import state_ops

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.delivery.constant import native_delivery_config
from nann.logger import get_logger
from nann.model.model_util import huge_constant
from nann.util import save_pbtxt, load_meta_graph, save_pb
import os

logger = get_logger("dvf.build_opt_graph")

# TODO: move the following into parse_opt function
emb_dim = 32 * 2
seq_max_len = 50


def fake_row_splits(x):
  return tf.cast(tf.concat([tf.constant([0]), tf.shape(x)], axis=0), tf.int64)


def set_difference(a, flags):
  row_splits = fake_row_splits(a)
  values, row_splits, flags = tf.bitmap_ref_difference(a, row_splits, flags)
  return values, flags


def ragged_gather(ragged_tensor, idx):
  """
  e.g. t = [[0,1],[2,3,4],[5,6],[7,8,9]], indicator = [0,1]
  return [0,1,2,3,4]
  """
  values, _ = tf.group_gather(ragged_tensor.values,
                              ragged_tensor.row_splits,
                              tf.cast(idx, tf.int64),
                              fake_row_splits(idx),
                              unique=False)
  return values


def top_k(ids, scores, k):
  """Return top k indices and scores based on scores

  Args:
      ids: shape (..., n)
      scores: same shape and order with ids
      k (int): top k to return

  Returns:
      ids: shape (..., min(k, n))
      scores: same shape and order with ids
  """
  scores, indices = tf.math.top_k(scores, k)
  ids = tf.gather(ids, indices)
  return ids, scores


def build_model(args):
  enter_points = np.load(join(args.index_dir, "enter_points.npy")).astype(np.int32)

  g = tf.Graph()
  with g.as_default() as g:
    user_seq_emb_fp16 = tf.placeholder(tf.float16, [1, seq_max_len * emb_dim], name='comm_seq')
    level_topn = tf.placeholder(tf.int32, [6], name='level_topn')

    #user_seq_emb = tf.cast(user_seq_emb_fp16, dtype=tf.float32)
    user_seq_emb = user_seq_emb_fp16
    user_seq_emb = tf.reshape(user_seq_emb, [1, seq_max_len, emb_dim])

    neighbor_ragged_tensors = [0] * args.hnsw_start_level
    with tf.device("/CPU:0"):
      item_embs = huge_constant(join(args.item_embs_dir, "item_embs.npy"), dtype=tf.float16)
      item_ids = huge_constant(join(args.item_embs_dir, "item_ids.npy"), dtype=tf.int64)
      for level in range(args.hnsw_start_level - 1, -1, -1):
        neighbor_ragged_tensors[level] = tf.RaggedTensor.from_row_splits(
          values=huge_constant(join(args.index_dir, f"neighbors_level_{level}_values.npy"), dtype=tf.int32),
          row_splits=huge_constant(join(args.index_dir, f"neighbors_level_{level}_row_splits.npy"), dtype=tf.int64),
          validate=False)

    def forward(idx):
      item_emb = tf.gather(item_embs, idx)
      # shape (num_item, d_item)
      #item_emb = tf.cast(item_emb, tf.float32)

      # use blaze_xla_op to scoring user-item pair
      blaze_xla_op_inputs = [user_seq_emb, item_emb]
      blaze_xla_op_input_names = native_delivery_config['inputs']
      blaze_xla_op_output_names = native_delivery_config['outputs']
      blaze_xla_op_output_types = [tf.float32]
      blaze_kernel = tf.blaze_xla_op(blaze_xla_op_inputs,
                                     blaze_xla_op_output_types,
                                     blaze_xla_op_input_names,
                                     blaze_xla_op_output_names,
                                     os.path.join(args.model_dir, "converted_model", native_delivery_config["target_graph_def"]),
                                     os.path.join(os.path.dirname(os.path.realpath(__file__)), native_delivery_config["target_opt_conf_path"]))
      return tf.squeeze(blaze_kernel[0])

    # level 2
    with tf.device("/CPU:0"):
      scores = forward(enter_points)
      idx_results, scores_result = top_k(enter_points, scores, level_topn[0])

      # level 1
      bucket_size = int(math.ceil(int(item_ids.shape[0]) / 32))
      idx_next = ragged_gather(neighbor_ragged_tensors[1], idx_results)
      flags = gen_state_ops.temporary_variable([bucket_size], tf.int32, name="ref_difference_bitmap_flags")
      update_op = state_ops.assign(flags, [0] * bucket_size)
      with tf.control_dependencies([update_op]):
        idx_results, _ = set_difference(idx_results, flags)
      with tf.control_dependencies([idx_results]):
        idx_next, _ = set_difference(idx_next, flags)

      scores_next = forward(idx_next)
      idx_result, scores_result = top_k(tf.concat([idx_results, idx_next], axis=0),
                                         tf.concat([scores_result, scores_next], axis=0),
                                         level_topn[1])
    # level 0
      idx_candidate = idx_result
      with tf.control_dependencies([idx_candidate]):
        update_op = state_ops.assign(flags, [0] * bucket_size)
      with tf.control_dependencies([update_op]):
        idx_candidate, _ = set_difference(idx_candidate, flags)

      for i in range(3):
        idx_next = ragged_gather(neighbor_ragged_tensors[0], idx_candidate)
        idx_next, _ = set_difference(idx_next, flags)
        scores_next = forward(idx_next)
        idx_candidate, scores_candidate = top_k(idx_next, scores_next, level_topn[i + 2])
        idx_result = tf.concat([idx_result, idx_candidate], axis=0)
        scores_result = tf.concat([scores_result, scores_candidate], axis=0)

      idx_result, scores_result = top_k(idx_result, scores_result, level_topn[5])
      item_ids = tf.gather(item_ids, idx_result)

      with tf.control_dependencies([item_ids]):
        final = gen_state_ops.destroy_temporary_variable(flags, var_name="ref_difference_bitmap_flags")
      with tf.control_dependencies([final]):
        final_topk_expand = tf.expand_dims(item_ids, axis=0, name='top_k')
    # build signature_def
    signature_def = tf.saved_model.build_signature_def(
      inputs={
        'comm_seq': tf.saved_model.build_tensor_info(user_seq_emb_fp16),
        'level_topn': tf.saved_model.build_tensor_info(level_topn),
      },
      outputs={
        'top_k': tf.saved_model.build_tensor_info(final_topk_expand),
      }
    )
  return g.as_graph_def(), signature_def


def parse_opt():
  # Parse commandline
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-dir", type=str, required=True,
                      help="directory contains model checkpoints")
  parser.add_argument("--hnsw-start-level", default=2, type=int,
                      help="hnsw search start level, should be same with that in index build process")
  parser.add_argument("--index-dir", type=str, required=True,
                      help="directory contains index")
  parser.add_argument("--item-embs-dir", type=str, required=True,
                      help="directory contains item ids and embs")
  parser.add_argument("--output-dir", type=str, required=True,
                      help="Output graph_def and signature_def folder")
  args = parser.parse_args()
  return args


def main():
  args = parse_opt()
  model_restore_path = tf.train.latest_checkpoint(args.model_dir)

  logger.info("building model")
  graph_def, signature_def = build_model(args)
  logger.info(f"loading meta graph from {model_restore_path}")
  meta_graph = load_meta_graph(model_restore_path + '.meta')
  meta_graph.signature_def.clear()
  meta_graph.signature_def['predict'].CopyFrom(signature_def)
  meta_graph.graph_def.Clear()
  meta_graph.graph_def.CopyFrom(graph_def)
  meta_graph.meta_info_def.ClearField('tensorflow_version')
  meta_graph.meta_info_def.ClearField('tensorflow_git_version')
  logger.info("saving 0.meta to %s" % args.output_dir)
  save_pbtxt(meta_graph, join(args.output_dir, 'exec.meta.pbtxt'))
  save_pb(meta_graph, join(args.output_dir, 'exec.meta.pb'))
  save_pbtxt(graph_def, join(args.output_dir, 'exec.pbtxt'))
  save_pb(graph_def, join(args.output_dir, 'exec.pb'))
  # save checkpoint
  with tf.gfile.GFile(join(args.output_dir, 'checkpoint'), 'w') as f:
    f.write('model_checkpoint_path: "0"\nall_model_checkpoint_paths: "0"\n')


if __name__ == "__main__":
  main()
