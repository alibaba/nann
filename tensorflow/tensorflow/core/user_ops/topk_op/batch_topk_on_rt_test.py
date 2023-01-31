import numpy as np
import tensorflow as tf

validation = True

def topk_on_RT(rt, k, ascending = False):
  values, idxes, row_splits = tf.batch_top_k_on_rt(rt.values, rt.row_splits, k, ascending = ascending)
  values_rt = tf.RaggedTensor.from_row_splits(values, row_splits, validate = validation)
  idxes_rt = tf.RaggedTensor.from_row_splits(idxes, row_splits, validate = validation)
  return values_rt, idxes_rt

a = tf.ragged.constant([[1,2,3,4,5,6,7],[11,12,13,14,15],[21,22,23,24,25,26,27],[31,32,33,34,35]], dtype=tf.float32)
empty = tf.RaggedTensor.from_row_splits(tf.constant([], dtype=tf.float32), [0], validate = validation)

b, c = topk_on_RT(a, [3,2,3,2], ascending = False)
d, e = topk_on_RT(a, 6, ascending = True)
f, g = topk_on_RT(empty, 3, ascending = True)

output = [b,c, d,e, f,g]
with tf.Session() as sess:
  res = sess.run(output)
  print(res[0].to_list())
  print(res[1].to_list())
  print("\n")  
  print(res[2].to_list())
  print(res[3].to_list())
  print("\n")  
  print(res[4].to_list())
  print(res[5].to_list())
