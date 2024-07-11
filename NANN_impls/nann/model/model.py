# encoding=utf-8
import logging
import math
import os.path

import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.delivery.constant import native_delivery_input_prefix, native_delivery_output_prefix
from .model_util import DNN, nonlinear_attention, kl_divergence_with_logit, huge_constant, accuracy

logger = logging.getLogger(__name__)


class Model:

  def __init__(self, args, batch=None, item_raw_features=None):
    """Model definition for train, extract features, test, and test all

    Args:
        args: all arguments
        item_raw_features (dict, optional): item raw features (keys contains item_ids, cate_ids, weight_tag).
                                            Only used for training
    """
    self.item_emb_dim = 2 * args.emb_dim

    self.item_raw_features = item_raw_features

    # zero means missing
    self.ht_item = self.get_hash_table(args.num_item + 1, args.emb_dim, name="ht_item")
    self.ht_cate = self.get_hash_table(args.num_cate + 1, args.emb_dim, name="ht_cate")

    if args.job_type == "train":
      # for training job, build graph in the step_fn of mirrored_strategy.experimental_run_v2
      pass
    elif args.job_type == "extract_feature":
      self.item_ids_batch = batch["gt_item_id"]
      self.item_embs_batch = tf.squeeze(
        self.get_item_emb(
          batch["gt_item_id"][:, tf.newaxis],
          batch["gt_cate_id"][:, tf.newaxis],
          training=False
        )
      )

    elif args.job_type == "test":
      self.gt_item_ids_batch = batch["gt_item_id"]
      self.user_seq_embs_batch = self.get_user_emb(batch["item_ids"], batch["cate_ids"])

      assert len(args.top_k_per_level) == args.hnsw_start_level + 1

      self.item_ids = huge_constant(path=args.item_ids_file, dtype=tf.int64)
      self.item_embs = huge_constant(path=args.item_embs_file, dtype=tf.float32)

      # neighborhood relationship
      self.neighbors_all = {}
      for level in range(args.hnsw_start_level - 1, -1, -1):
        self.neighbors_all[level] = tf.RaggedTensor.from_row_splits(
          values=huge_constant(os.path.join(args.index_dir, f"neighbors_level_{level}_values.npy")),
          row_splits=huge_constant(os.path.join(args.index_dir, f"neighbors_level_{level}_row_splits.npy")),
          validate=False)
      # use all the nodes on the start_level as enter points
      self.enter_points = tf.constant(np.load(os.path.join(args.index_dir, "enter_points.npy")), dtype=tf.int64)
      # for search on hnsw graph
      self.start_level = args.hnsw_start_level
      self.top_k_per_level = args.top_k_per_level
      self.num_scoring_per_level = args.num_scoring_per_level
      self.topk_eval = max(args.topk_eval)

      self.user_seq_emb_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[1, args.max_seq_length, 2 * args.emb_dim],
                                            name="user_sequence_embedding_placeholder")
      self.scores_in_retrieval = []
      self.retrieval_results = self.retrieval(self.user_seq_emb_ph)
    elif args.job_type in ("test_all", "export"):
      self.gt_item_ids_batch = batch["gt_item_id"]
      self.user_seq_embs_batch = self.get_user_emb(batch["item_ids"], batch["cate_ids"])

      with tf.name_scope(native_delivery_input_prefix):
        self.user_seq_emb_ph = tf.placeholder(dtype=tf.float32,
                                            shape=[1, args.max_seq_length, 2 * args.emb_dim],
                                            name="user_seq_emb")
        self.item_emb_ph = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.item_emb_dim],
                                      name="item_emb")
      # shape (1=num_user, num_item, )
      logits = self.forward(self.user_seq_emb_ph, self.item_emb_ph[tf.newaxis, ...], training=False)
      # shape (1=num_user, num_item, ) -> (num_item, )
      self.scores_batch = tf.squeeze(logits)

  def train_func(self, batch, args):
    item_ids = tf.constant(self.item_raw_features["item_id"], np.int64)
    cate_ids = tf.constant(self.item_raw_features["cate_id"], np.int64)
    weight_tag_list = list(self.item_raw_features["weight_tag"])

    sampled_idx, true_expcnt, sample_expcnt = tf.random.fixed_unigram_candidate_sampler(
      true_classes=batch["gt_item_id"][:, tf.newaxis],
      num_true=1,
      num_sampled=args.num_neg * args.batch_size,
      unique=True,
      range_max=args.num_item,
      unigrams=list(weight_tag_list))
    sampled_idx = tf.reshape(sampled_idx, [args.batch_size, args.num_neg])
    sample_expcnt = tf.reshape(sample_expcnt, [args.batch_size, args.num_neg])

    # shape (batch_size, 1 + num_neg)
    target_item_id = tf.concat([batch["gt_item_id"][:, tf.newaxis], tf.gather(item_ids, sampled_idx)], axis=-1)
    target_cate_id = tf.concat([batch["gt_cate_id"][:, tf.newaxis], tf.gather(cate_ids, sampled_idx)], axis=-1)
    expected_cnt = tf.concat([true_expcnt, sample_expcnt], axis=-1)
    expected_cnt = tf.math.log(1e-20 + expected_cnt)

    # shape (batch_size, 1 + num_neg)
    label_reshaped = tf.concat([tf.ones([args.batch_size, 1]),
                                tf.zeros([args.batch_size, args.num_neg])],
                               axis=-1)

    # item embedding
    item_emb = self.get_item_emb(target_item_id, target_cate_id, training=True)
    user_seq_emb = self.get_user_emb(batch["item_ids"], batch["cate_ids"])

    with tf.GradientTape() as tape:
      tape.watch(item_emb)

      # (batch_size, 1 + num_neg)
      logits_origin = self.forward(user_seq_emb, item_emb, training=True)
      logits = logits_origin - expected_cnt
      acc = accuracy(logits, label_reshaped)
      acc_origin = accuracy(logits_origin, label_reshaped)
      loss_xe = tf.losses.sigmoid_cross_entropy(label_reshaped, logits, label_smoothing=0)

    if args.adv_eps > 0:
      # adv
      # Get the gradients of the loss w.r.t to the item embedding
      gradient = tape.gradient(loss_xe, item_emb)
      # Get the sign of the gradients to create the perturbation
      signed_grad = tf.sign(gradient)
      item_emb_adv = item_emb + args.adv_eps * signed_grad
      logits_adv = self.forward(user_seq_emb, item_emb_adv, training=True)
      logits_adv -= expected_cnt
      loss_adv = kl_divergence_with_logit(tf.stop_gradient(logits), logits_adv)
    else:
      loss_adv = tf.constant(0.0)

    loss = loss_xe + args.adv_weight * loss_adv

    return loss, loss_xe, loss_adv, acc, acc_origin

  def get_user_emb(self, item_ids, cate_ids):
    # user_seq_emb
    input_item_emb = self.embedding_lookup(self.ht_item, item_ids)
    input_cate_emb = self.embedding_lookup(self.ht_cate, cate_ids)
    user_seq_emb = tf.concat([input_item_emb, input_cate_emb], axis=-1)
    # shape (batch_size, max_seq_length, 2*emb_dim)
    logger.info("user sequence shape: %s" % user_seq_emb.shape)
    return user_seq_emb

  def get_item_emb(self, target_item_id, target_cate_id, training):
    # item embedding
    gt_item_emb = self.embedding_lookup(self.ht_item, target_item_id)
    gt_cate_emb = self.embedding_lookup(self.ht_cate, target_cate_id)
    item_emb = tf.concat([gt_item_emb, gt_cate_emb], axis=-1)
    for layer in [
      DNN(output_dim=self.item_emb_dim, active_op="prelu", norm_op="bn", name='dnn_item_1'),
      DNN(output_dim=self.item_emb_dim, active_op="prelu", norm_op="bn", name='dnn_item_2'),
      DNN(output_dim=self.item_emb_dim, active_op=None, norm_op=None, name='dnn_item_3'),
    ]:
      item_emb = layer.call(item_emb, training=training)
    # shape (batch_size, 1 + num_neg, item_emb_dim)
    logger.info("item emb shape: %s" % item_emb.shape)
    return item_emb

  @staticmethod
  def embedding_lookup(ht, inputs):
    emb = tf.nn.embedding_lookup(ht, inputs)
    # mask out zero
    mask = tf.cast(inputs > 0, tf.float32)
    emb *= mask[..., tf.newaxis]
    return emb

  @staticmethod
  def get_hash_table(n, d, name, dtype=tf.float32):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1 / math.sqrt(d))
    return tf.get_variable(name=name, dtype=dtype, trainable=True, initializer=initializer, shape=[n, d])

  @staticmethod
  def forward(user_seq_emb, item_emb, training=True):
    """Calculate scores for each pair of user_seq_emb and item_emb

    Args:
        user_seq_emb: user sequence embedding with shape (num_user, max_seq_length, emb_dim)
        item_emb: item embedding with shape (num_user, num_item_per_user, d_item)
        training (bool, optional): whether in training mode. Defaults to True.

    Returns:
        logits: scores with shape (num_user, num_item_per_user)
    """

    # marking input for native delivery
    batch_size = tf.shape(user_seq_emb)[0]

    # shape (num_user, num_item_per_user, max_seq_length, emb_dim)
    att_out = nonlinear_attention(item_emb, user_seq_emb, user_seq_emb)
    # shape (num_user, num_item_per_user, emb_dim)
    att_out = tf.reduce_sum(att_out, axis=-2)

    # shape (num_user, num_item_per_user, d_item)
    # item_emb_expand = tf.tile(item_emb[tf.newaxis, :, :], [batch_size, 1, 1])
    item_emb_expand = item_emb
    # shape (num_user, num_item_per_user, emb_dim + d_item)
    logits = tf.concat([att_out, item_emb_expand], axis=-1)
    for layer in [
      DNN(output_dim=128, active_op="prelu", norm_op="bn", name='1_dnn'),
      DNN(output_dim=64, active_op="prelu", norm_op="bn", name='2_dnn'),
      DNN(output_dim=32, active_op="prelu", norm_op="bn", name='3_dnn'),
      # no bias for the last fc layer, avoid interrupting XLA optimizations.
      DNN(output_dim=1, use_bias=False, active_op=None, norm_op=None, name='4_dnn')
    ]:
      logits = layer.call(logits, training=training)

    # (num_user, num_item_per_user, 1) -> (num_user, num_item_per_user)
    if training:
      logits = tf.reshape(logits, [batch_size, -1], name="final_logit")
    else:
      logits = tf.reshape(logits, [-1, batch_size], name="final_logit")

    # marking output for native delivery
    with tf.name_scope(native_delivery_output_prefix):
      logits = tf.identity(logits, name="logits")

    return logits

  # the following functions are used for retrieval
  def get_scores(self, item_idx, user_seq_emb):
    """Alias method to get scores in the test stage. the first dim of user_seq_emb should be 1.

    Args:
        item_idx (Tensor): index of items with shape (n, )
        user_seq_emb (Tensor): user sequence embedding with shape (1, max_seq_length, emb_dim)

    Returns:
        scores: scores with shape (n)
    """
    # shape: (1, n, emb_dim)
    item_embs = tf.gather(self.item_embs, item_idx)[tf.newaxis, ...]
    # shape: (1, n)
    scores = self.forward(user_seq_emb, item_embs, training=False)
    # shape: (n)
    scores = tf.squeeze(scores)
    self.scores_in_retrieval.append(scores)
    return scores

  @staticmethod
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
    k = tf.reduce_min([k, tf.shape(ids)[-1]])
    scores, indices = tf.math.top_k(scores, k)
    ids = tf.gather(ids, indices)
    return ids, scores

  @staticmethod
  def set_difference(a, b):
    """return set(a) - set(b)

    Args:
        a (Tensor): shape (n_a, )
        b (Tensor): shape (n_b, )

    Returns:
        diff (Tensor): set difference results
    """
    return tf.sets.set_difference(a[tf.newaxis, :], b[tf.newaxis, :]).values

  @staticmethod
  def set_union(a, b):
    """return set(a) | set(b)

    Args:
        a (Tensor): shape (n_a, )
        b (Tensor): shape (n_b, )

    Returns:
        diff (Tensor): set union results
    """
    return tf.sets.set_union(a[tf.newaxis, :], b[tf.newaxis, :]).values

  def search_level(self, user_embedding, idx_ep, scores_ep, level):
    """search algorithm in certain level

    Args:
        user_embedding (Tensor): embedding of the user to retrieval
        idx_ep (Tensor): indices of enter points
        scores_ep (Tensor): scores for enter points. Same shape and order with idx_ep, and is sorted as descending order
        level (int): index of current level. Range from 0 to start_level

    Returns:
        idx_result: indices of results in the current level, with shape (self.top_k_per_level[level], )
        scores_result: scores of results in the current level, same shape and order with ids, sorted as descending order
    """
    visited_idx = idx_ep

    idx_candidate = idx_ep
    idx_result, scores_result = idx_ep, scores_ep

    for i in range(self.num_scoring_per_level[level]):
      # propagation to get next idx_candidate
      idx_next = tf.gather(self.neighbors_all[level], idx_candidate)
      idx_next, _ = tf.unique(idx_next.flat_values)
      # mask out visited candidates
      idx_next = self.set_difference(idx_next, visited_idx)
      # flag as visited
      visited_idx = self.set_union(visited_idx, idx_next)

      scores_next = self.get_scores(idx_next, user_embedding)

      # concat to results and get top num_search idx/score
      idx_result, scores_result = self.top_k(tf.concat([idx_result, idx_next], axis=0),
                                             tf.concat([scores_result, scores_next], axis=0),
                                             self.top_k_per_level[level])

      mask = tf.reshape(scores_next >= scores_result[-1], [-1])
      idx_candidate = tf.boolean_mask(idx_next, mask)

    # idx is sorted in descending order by score
    return idx_result, scores_result

  def retrieval(self, user_embedding):
    """retrieval algorithm based on hnsw index.

    Args:
        user_embedding: embedding of the user to retrieval

    Returns:
        item_ids: retrieved item ids with shape (self.topk_eval, )
    """
    assert self.num_scoring_per_level[self.start_level] == 1

    # get all the nodes in the start_level
    results = self.enter_points
    scores = self.get_scores(results, user_embedding)
    # use the topk nodes as enter points for the next level
    results, scores = self.top_k(results, scores, self.top_k_per_level[self.start_level])

    for level in range(self.start_level - 1, -1, -1):
      results, scores = self.search_level(user_embedding, results, scores, level)

    results = results[:self.topk_eval]

    item_ids = tf.gather(self.item_ids, results)
    return item_ids
