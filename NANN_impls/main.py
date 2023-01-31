#!/usr/bin/python
# -- coding:utf8 --
import math
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

from nann.data_provider import dataio
from nann.config import parse_opt
from nann.logger import get_logger
from nann.model import Model
from nann.util import AverageMeter, calc_pr, fast_argtopk


def num_batch(num_samples, batch_size, epochs=1, drop_remainder=False):
  """Calculate total number of batch

  Args:
      num_samples: total number of samples in the dataset
      batch_size: batch size
      epochs (int, optional): number of epochs. Defaults to 1.
      drop_remainder (bool, optional): if true, the last batch will be dropped if it is smaller than batch_size.
                                       Should be same with that in the dataset. Defaults to False.

  Returns:
      num_batch: total number of batch
  """
  to_int = math.floor if drop_remainder else math.ceil
  return int(to_int(num_samples / batch_size) * epochs)


def train(mean_replica_results, saver, sess, args):
  writer = tf.summary.FileWriter(args.summary_output_dir, sess.graph)
  num_train_batch = num_batch(args.num_train_samples, args.global_batch_size, args.train_epochs, True)

  tic = time.time()
  batch_time_meter = AverageMeter("batch time", ":.2f", moving_average=True)
  loss_meter = AverageMeter("loss", ":.4f", moving_average=True)
  loss_xe_meter = AverageMeter("loss_xe", ":.4f", moving_average=True)
  loss_adv_meter = AverageMeter("loss_adv", ":.2E", moving_average=True)
  acc_meter = AverageMeter("acc", ":.2%", moving_average=True)
  acc_origin_meter = AverageMeter("acc_origin", ":.2%", moving_average=True)

  for train_step in range(num_train_batch):
    loss, loss_xe, loss_adv, acc, acc_origin = sess.run(mean_replica_results)

    acc_meter.update(acc)
    acc_origin_meter.update(acc_origin)
    loss_meter.update(loss)
    loss_xe_meter.update(loss_xe)
    loss_adv_meter.update(loss_adv)
    batch_time_meter.update(time.time() - tic)
    tic = time.time()

    # print
    if train_step % args.print_freq == 0:
      logger.info(f'[{train_step}/{num_train_batch}] '
                  f'{loss_meter}, {loss_xe_meter}, {loss_adv_meter}, '
                  f'{acc_meter}, {acc_origin_meter}, {batch_time_meter}')

    # save checkpoint
    if train_step % args.save_freq == 0 or train_step + 1 == num_train_batch:
      logger.info(f"saving model to {args.model_save_dir}")
      saver.save(sess, args.model_save_dir + "/ckpt", global_step=train_step)

    # tensorboard summary
    summary = tf.Summary()
    summary.value.add(tag='loss', simple_value=loss_meter.val)
    summary.value.add(tag='acc', simple_value=acc_meter.val)
    summary.value.add(tag='acc_origin', simple_value=acc_origin_meter.val)
    summary.value.add(tag='loss_xe', simple_value=loss_xe_meter.val)
    summary.value.add(tag='loss_adv', simple_value=loss_adv_meter.val)
    summary.value.add(tag='acc', simple_value=acc_meter.val)
    summary.value.add(tag='batch_time', simple_value=batch_time_meter.val)
    writer.add_summary(summary, train_step)


def extract_item_features(model, sess, args):
  """extract item features from model

  Returns:
      item_ids: item ids
      item_embs: item embeddings, same order as item_ids
  """
  item_ids = []
  item_embs = []

  batch_time_meter = AverageMeter("batch time", ":.2f")

  tic = time.time()
  num_extract_batch = num_batch(args.num_item, args.global_batch_size)
  for step in range(num_extract_batch):
    item_ids_batch, item_embs_batch = sess.run([model.item_ids_batch, model.item_embs_batch])

    item_ids.append(item_ids_batch)
    item_embs.append(item_embs_batch)

    batch_time_meter.update(time.time() - tic)
    tic = time.time()

    if step % args.print_freq == 0:
      logger.info(f'Extracting features: [{step}/{num_extract_batch}] {batch_time_meter}')

  item_embs = np.concatenate(item_embs, axis=0)
  item_ids = np.concatenate(item_ids, axis=0)

  return item_ids, item_embs


def extract_user_features(model, sess, args):
  """
  return:
    user_seq_embs: shape (#num_test_batch, #user_seq_emb_dim)
    ground_truths: (#num_test_batch),
  """
  ground_truths = []
  user_seq_embs = []

  batch_time_meter = AverageMeter("batch time", ":.2f")

  step = 0
  num_user = 0
  tic = time.time()
  while num_user < args.num_test_batch:
    ground_truths_batch, user_seq_embs_batch = sess.run([model.gt_item_ids_batch, model.user_seq_embs_batch])
    ground_truths.append(ground_truths_batch)
    user_seq_embs.append(user_seq_embs_batch)

    batch_time_meter.update(time.time() - tic)

    tic = time.time()
    step += 1
    num_user += len(ground_truths_batch)

    logger.info(f'Extracting user features: [{num_user}/{args.num_test_batch}] {batch_time_meter}')

  user_seq_embs = np.concatenate(user_seq_embs, axis=0)
  ground_truths = np.concatenate(ground_truths, axis=0)
  return ground_truths, user_seq_embs


def test(model, sess, args):
  ground_truths, user_seq_embs = extract_user_features(model, sess, args)

  tic = time.time()
  batch_time_meter = AverageMeter("batch time", ":.2f", moving_average=True)
  prec_meters = defaultdict(lambda: AverageMeter("prec", ":.2%"))
  recall_meters = defaultdict(lambda: AverageMeter("recall", ":.2%"))
  f1_meters = defaultdict(lambda: AverageMeter("f1", ":.2%"))

  num_test_batch = min(args.num_test_batch, len(ground_truths))

  times_scoring = sum(args.num_scoring_per_level)
  num_scorings = np.zeros([num_test_batch, times_scoring])
  for idx_user in range(num_test_batch):
    scores, retrieval = sess.run([model.scores_in_retrieval, model.retrieval_results],
                                 feed_dict={model.user_seq_emb_ph: user_seq_embs[idx_user:idx_user + 1, :]})
    for i, score in enumerate(scores):
      num_scorings[idx_user, i] = score.shape[0]

    batch_time_meter.update(time.time() - tic)
    tic = time.time()

    for topk in args.topk_eval:
      assert retrieval.shape[0] >= topk
      prec, recall, f1 = calc_pr(ground_truths[idx_user], retrieval[:topk])

      prec_meters[topk].update(prec)
      recall_meters[topk].update(recall)
      f1_meters[topk].update(f1)

      if idx_user % args.print_freq == 0:
        logger.info(f"[{idx_user}/{num_test_batch}] {topk}: "
                    f"{prec_meters[topk]}, {recall_meters[topk]}, {f1_meters[topk]}, "
                    f"{batch_time_meter}")

    if idx_user % args.print_freq == 0:
      logger.info(f"num scoring: mean {num_scorings[:idx_user + 1, :].mean(axis=0)}, "
                  f"total {num_scorings[:idx_user + 1, :].sum(axis=-1).mean()}")

  logger.info(f"num scoring: mean {num_scorings.mean(axis=0)}, "
              f"min {num_scorings.min(axis=0)}, "
              f"max {num_scorings.max(axis=0)}, "
              f"total {num_scorings.sum(axis=1).mean()}")

  return recall_meters, num_scorings.sum(axis=1).mean()

def export(mode, sess, args, saver):
  saver.save(sess, args.model_export_dir + "/export")
  logger.info(f"export model to {args.model_export_dir}")

def test_all(model, sess, args):
  logger.info('testing all')

  item_ids = np.load(args.item_ids_file)
  item_embs = np.load(args.item_embs_file)

  ground_truths, user_seq_embs = extract_user_features(model, sess, args)

  tic = time.time()
  batch_time_meter = AverageMeter("batch time", ":.2f", moving_average=True)
  prec_meters = defaultdict(lambda: AverageMeter("prec", ":.2%"))
  recall_meters = defaultdict(lambda: AverageMeter("recall", ":.2%"))
  f1_meters = defaultdict(lambda: AverageMeter("f1", ":.2%"))

  num_test_batch = min(args.num_test_batch, len(ground_truths))

  for idx_user in range(num_test_batch):
    scores_all = []
    # split items to multiple parts for avoiding OOM issue
    for item_embs_batch in np.array_split(item_embs, 50, axis=0):
      scores = sess.run(model.scores_batch,
                        feed_dict={model.user_seq_emb_ph: user_seq_embs[idx_user:idx_user + 1, :],
                                   model.item_emb_ph: item_embs_batch})
      scores_all.append(scores)

    scores_all = np.concatenate(scores_all, axis=0)
    indices_topk = fast_argtopk(scores_all, max(args.topk_eval))
    retrieval = item_ids[indices_topk]

    batch_time_meter.update(time.time() - tic)
    tic = time.time()

    for topk in args.topk_eval:
      prec, recall, f1 = calc_pr(ground_truths[idx_user], retrieval[:topk])
      prec_meters[topk].update(prec)
      recall_meters[topk].update(recall)
      f1_meters[topk].update(f1)

      if idx_user % args.print_freq == 0:
        logger.info(f"[{idx_user}/{num_test_batch}] {topk}: "
                    f"{prec_meters[topk]}, {recall_meters[topk]}, {f1_meters[topk]}, "
                    f"{batch_time_meter}")

  return recall_meters


def main(args):
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  # build model
  if args.job_type == "train":
    mirrored_strategy = tf.distribute.MirroredStrategy()
    args.global_batch_size = args.batch_size * mirrored_strategy.num_replicas_in_sync

    batch_per_replica, data_initializer = dataio.tfrecords_dataio(
      filename=args.dataset_file,
      max_seq_length=args.max_seq_length,
      batch_size=args.global_batch_size,
      epochs=args.train_epochs,
      # for train dataset, drop the remainder to make the bath size consistent
      drop_remainder=True,
      mirrored_strategy=mirrored_strategy)

    logger.info(f"loading item raw feature from {args.item_raw_features_file}")
    item_raw_features = np.load(args.item_raw_features_file)

    with mirrored_strategy.scope():
      model = Model(args, None, item_raw_features)
      optimizer = tf.contrib.opt.AdamWOptimizer(args.weight_decay, args.learning_rate)

      def step_fn(_batch):
        loss_var, loss_xe_var, loss_adv_var, acc_var, acc_origin_var = model.train_func(_batch, args)
        train_op = optimizer.minimize(loss_var)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([train_op] + update_ops):
          return tf.identity(loss_var), loss_xe_var, loss_adv_var, acc_var, acc_origin_var

      per_replica_results = mirrored_strategy.experimental_run_v2(step_fn, args=(batch_per_replica,))
      mean_replica_results = [mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, result, axis=None)
                              for result in per_replica_results]
      sess.run([tf.global_variables_initializer(), data_initializer])
  else:
    args.global_batch_size = args.batch_size

    batch, _ = dataio.tfrecords_dataio(
      filename=args.dataset_file,
      max_seq_length=args.max_seq_length,
      batch_size=args.global_batch_size,
      epochs=1,
      only_item=args.job_type == "extract_feature",
      drop_remainder=False)

    model = Model(args, batch, None)
    sess.run(tf.global_variables_initializer())

  for v in tf.global_variables():
    logger.info(f"{v.name} {v.get_shape()}")

  saver = tf.train.Saver(sharded=False, max_to_keep=50)

  # auto resume from model directory
  model_restore_path = tf.train.latest_checkpoint(args.model_save_dir)
  if model_restore_path:
    logger.info(f"resuming from {model_restore_path}")
    saver.restore(sess, model_restore_path)

  if args.job_type == "train":
    train(mean_replica_results, saver, sess, args)

  elif args.job_type == "extract_feature":
    item_ids, item_embs = extract_item_features(model, sess, args)
    logger.info(f"dumping item id to {args.item_ids_file}")
    np.save(args.item_ids_file, item_ids, allow_pickle=False)
    logger.info(f"dumping item embeddings to {args.item_embs_file}")
    np.save(args.item_embs_file, item_embs, allow_pickle=False)

  elif args.job_type == "test":
    recall_meters, num_scoring = test(model, sess, args)

    for topk in args.topk_eval:
      logger.info(f"Test Recall@{topk} {recall_meters[topk].avg:.2%}, num_scoring {num_scoring:.0f}")

  elif args.job_type == "test_all":
    recall_meters = test_all(model, sess, args)

    for topk in args.topk_eval:
      logger.info(f"Test all Recall@{topk} {recall_meters[topk].avg:.2%}")

  elif args.job_type == "export":
    export(model, sess, args, saver)


if __name__ == '__main__':
  _args = parse_opt()

  logger = get_logger("nann", output=_args.log_file)
  logger.info("tensorflow version: %s, file %s" % (tf.__version__, tf.__file__))
  logger.info(_args)

  # for pretty logging number scoring in the test stage
  np.set_printoptions(precision=0)

  main(_args)
