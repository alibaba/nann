import numpy as np
import tensorflow as tf

validation = True

def splits_gather(splits, indicator):
  '''
  e.g. splits = [0,2,5,7,10], indicators = [[0,1],[3]]
  return [[0,1,2,3,4],[7,8,9]]
  '''
  #rt = splits_to_rt(splits)
  #return group_gather(rt. indicator)
  values, row_splits = tf.splits_gather(splits, indicator.values, indicator.row_splits)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation)


splits = tf.constant([0,2,5,7,10], dtype=tf.int64)
indicator = tf.ragged.constant([[0,1],[3]], dtype=tf.int64)

v_empty = tf.constant([], dtype=tf.int64)
#empty = tf.ragged.constant([], dtype=tf.int64)
empty = tf.RaggedTensor.from_row_splits(tf.constant([], dtype=tf.int64), [0], validate = validation)

#should be [[0,1,2,3,4],[7,8,9]]
a = splits_gather(splits, indicator)

b = splits_gather(v_empty, indicator)
c = splits_gather(splits, empty)


output = [a,b,c]
with tf.Session() as sess:
  res = sess.run(output)
  print(res)
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
