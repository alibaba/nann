import numpy as np
import tensorflow as tf

validation = True

def group_gather(rt, indicator, unique = False):
  '''
  e.g. t = [[0,1],[2,3,4],[5,6],[7,8,9]], indicator = [[0,1],[3]]
  return [[0,1,2,3,4],[7,8,9]]
  '''
  #indicator = splits_gather(rt.row_splits, indicator)
  #return tf.RaggedTensor.from_row_splits(tf.gather(rt.values, indicator.values), indicator.row_splits, validate=validation)
  values, row_splits = tf.group_gather(rt.values, rt.row_splits, indicator.values, indicator.row_splits, unique=unique)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate=validation)



rt = tf.ragged.constant([[0,1,1,2,3,4],[3,4,5,5,6],[7,8,8,9],[10,11,12]], dtype=tf.int64)
indicator = tf.ragged.constant([[0,1],[3]], dtype=tf.int64)

empty = tf.RaggedTensor.from_row_splits(tf.constant([], dtype=tf.int64), [0], validate = validation)

a = group_gather(rt, indicator)
b = group_gather(rt, indicator, unique=True)
c = group_gather(empty, indicator)
d = group_gather(rt, empty)

output = [a,b,c,d]
with tf.Session() as sess:
  res = sess.run(output)
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
  print(res[3].to_list())
