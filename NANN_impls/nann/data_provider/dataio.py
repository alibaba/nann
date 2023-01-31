#!/usr/bin/env python
# -*— coding: utf-8 —*-

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def tfrecords_dataio(filename,
                     max_seq_length=50,
                     batch_size=256,
                     epochs=5,
                     drop_remainder=True,
                     shuffle_buffer_size=100000,
                     prefetch_buffer_size=1,
                     only_item=False,
                     mirrored_strategy=None):
  """dataset wrapper to load tfrecords, shuffle, batch

  Args:
      filename (str): path to the tfrecords file.
      max_seq_length (int, optional): max length for user sequence features. Should be same with that in the tfrecords.
                                      Defaults to 50.
      batch_size (int, optional): batch size. Defaults to 256.
      epochs (int, optional): number of epochs, should be set to 1 for non-training jobs. Defaults to 5.
      drop_remainder (bool, optional): if true, the last batch will be dropped if it is smaller than batch_size.
                                       Should be True for training, False for other jobs.
                                       Defaults to True.
      shuffle_buffer_size (int, optional): buffer size for shuffling. Defaults to 100000.
      prefetch_buffer_size (int, optional): prefetch to buffer size. Defaults to 1.
      only_item (bool, optional): If True, only load item features, not load user features.
                                  If False, both load user and item features.
                                  Defaults to False.
      mirrored_strategy (MirroredStrategy, optional): If not none, distribute the dataset to multiple parts

  Return:
      batch: a nested structure of tf.Tensors representing the next batch of dataset.
      data_initializer: None if mirrored_strategy is None else iterator.initializer
  """

  def _parse_function(proto):
    feature_description = {
      'gt_item_id': tf.io.FixedLenFeature([], tf.int64),
      'gt_cate_id': tf.io.FixedLenFeature([], tf.int64),
      'weight_tag': tf.io.FixedLenFeature([], tf.float32),
    }
    if not only_item:
      feature_description.update({
        'item_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
        'cate_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      })
    return tf.io.parse_single_example(proto, feature_description)

  with tf.name_scope("dataio"):
    dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=8)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(prefetch_buffer_size)

    if mirrored_strategy is not None:
      dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
      iterator = dataset.make_initializable_iterator()
      data_initializer = iterator.initializer
    else:
      iterator = dataset.make_one_shot_iterator()
      data_initializer = None

    return iterator.get_next(), data_initializer


if __name__ == '__main__':
  batch = tfrecords_dataio("data_provider/tiny/ub_train.tfrecords", max_seq_length=50)
  sess = tf.InteractiveSession()
  print(batch["cate_ids"].eval())
