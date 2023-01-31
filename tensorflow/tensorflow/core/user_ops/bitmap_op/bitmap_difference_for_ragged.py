import numpy as np
import tensorflow as tf

validation = True


bucket_size = int(1e6)

op_module = tf.load_op_library("./bloom_filter_difference.so")


def difference(a, flags):
  values, row_splits, flags = op_module.bitmap_difference_for_ragged(a.values, a.row_splits, flags)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation), flags


with tf.device("/CPU:0"):
  flags = tf.zeros([4], tf.int32) 
  a = tf.ragged.constant([[1,1,2,2,3,4,5],[11,12,13]], dtype=tf.int32)
  b = tf.ragged.constant([[4,5,6,7,7,8,10],[13,14]], dtype=tf.int32)
  
  init = tf.global_variables_initializer()

  #empty = tf.ragged.constant([], dtype=tf.int64)
  c0, flags = difference(a, flags)
  
  c1, flags = difference(b, flags)
    
  c2, flags = difference(b, flags)
  
  


output = [c0,c1, c2, flags]
with tf.Session() as sess:
  sess.run(init)
  res = sess.run(output)
  print("===============")
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
  print(res[3])
  
