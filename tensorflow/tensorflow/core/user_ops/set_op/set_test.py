import numpy as np
import tensorflow as tf

validation = True

def union(a, b):
  values, row_splits = tf.set_union(a.values, a.row_splits, b.values, b.row_splits)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation)

def intersection(a, b):
  values, row_splits = tf.set_intersection(a.values, a.row_splits, b.values, b.row_splits)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation)

def difference(a, b):
  values, row_splits = tf.set_difference(a.values, a.row_splits, b.values, b.row_splits)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation)


with tf.device("/CPU:0"):
  a = tf.ragged.constant([[1,1,2,2,3,4,5],[11,12,13]], dtype=tf.int64)
  b = tf.ragged.constant([[4,5,6,7,7,8],[13,14]], dtype=tf.int64)

  #empty = tf.ragged.constant([], dtype=tf.int64)
  empty = tf.RaggedTensor.from_row_splits(tf.constant([], dtype=tf.int64), [0], validate = validation)

  c0 = union(a, b)
  c1 = union(a, empty)
  c2 = union(empty, b)
  d0 = intersection(a, b)
  d1 = intersection(a, empty)
  d2 = intersection(empty, b)
  e0 = difference(a, b)
  e1 = difference(a, empty)
  e2 = difference(empty, b)


output = [c0,c1,c2,
          d0,d1,d2,
          e0,e1,e2]
with tf.Session() as sess:
  res = sess.run(output)
  print("===============")
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
  print("===============")
  print(res[3].to_list())
  print(res[4].to_list())
  print(res[5].to_list())
  print("===============")
  print(res[6].to_list())
  print(res[7].to_list())
  print(res[8].to_list())
  
