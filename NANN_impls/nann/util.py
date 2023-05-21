import os
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf
from tensorflow_core.python.framework import meta_graph


def fast_argtopk(arr, k):
  indices = np.argpartition(-arr, k)[:k]
  return indices[np.argsort(-arr[indices])]


def calc_pr(ground_truth, retrievals):
  ground_truths = {ground_truth}
  retrievals = set(retrievals)
  hit_num = len(ground_truths & retrievals)
  p = hit_num * 1.0 / len(retrievals)
  r = hit_num * 1.0 / len(ground_truths)
  if p + r > 0:
    f1 = 2 * p * r / (p + r)
  else:
    f1 = 0.0

  return p, r, f1


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f',
               moving_average=False,
               moving_average_momentum=0.99,
               moving_average_count=10000):
    self.name = name
    self.fmt = fmt
    self.moving_average = moving_average
    self.moving_average_momentum = moving_average_momentum
    self.moving_average_count = moving_average_count
    self.reset()

  def reset(self):
    self.val = 0
    self.sum = 0
    self.count = 0
    self.avg = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    if self.moving_average and self.count > self.moving_average_count:
      self.avg = self.moving_average_momentum * self.avg + (1 - self.moving_average_momentum) * val
    else:
      self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


def np_savez_hdfs(d, hdfs_file, overwrite=True):
  """
  save dict to hdfs by numpy savez
  :param d: dict, each value should be a numpy array.
  :param hdfs_file: hdfs path to save the numpy arrays.
  :param overwrite: if true, overwrite the existing hdfs file
  """
  with NamedTemporaryFile() as f:
    np.savez(f, **d)
    tf.gfile.Copy(f.name, hdfs_file, overwrite=overwrite)


def np_load_hdfs(hdfs_file, remove_local=True):
  """
  load dict from hdfs by numpy load.
  """
  local_file_name = os.path.basename(hdfs_file)
  tf.gfile.Copy(hdfs_file, local_file_name)
  results = np.load(local_file_name, allow_pickle=True)
  if remove_local:
    os.remove(local_file_name)
  return results


def load_meta_graph(input_meta_graph):
  if not tf.gfile.Exists(input_meta_graph):
    raise IOError("Input meta graph file '" + input_meta_graph + "' does not exist!")
  input_meta_graph_def = meta_graph.read_meta_graph_file(input_meta_graph)
  return input_meta_graph_def


def save_pb(pb, file_name):
  with tf.gfile.GFile(file_name, "wb") as f:
    f.write(pb.SerializeToString())


def save_pbtxt(pb, file_name):
  from google.protobuf import text_format
  with tf.gfile.GFile(file_name, 'w') as f:
    f.write(text_format.MessageToString(pb))
