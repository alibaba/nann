import numpy as np
import tensorflow as tf

validation = True


bucket_size = int(1e6)


def difference(a, flags):
  values, row_splits, flags = tf.bitmap_ref_difference(a.values, a.row_splits, flags)
  return tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation), flags


with tf.device("/CPU:0"):
  flags = tf.Variable([0,0,0,0]) 
  a = tf.ragged.constant([[1,1,2,2,3,4,5],[11,12,13]], dtype=tf.int32)
  b = tf.ragged.constant([[4,5,6,7,7,8,10],[13,14]], dtype=tf.int32)
  
  init = tf.global_variables_initializer()

  #empty = tf.ragged.constant([], dtype=tf.int64)
  c0, _ = difference(a, flags)
  
  with tf.control_dependencies([c0]):
    c1, _ = difference(b, flags)
    
  with tf.control_dependencies([c1]):
    c2, _ = difference(b, flags)
  
  


output = [c0,c1, c2, flags]
with tf.Session() as sess:
  sess.run(init)
  res = sess.run(output)
  print("===============")
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
  print(res[3])
  
