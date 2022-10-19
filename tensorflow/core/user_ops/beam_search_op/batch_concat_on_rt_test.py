import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

validation = True

g = tf.Graph()

with g.as_default():
  with tf.device("/CPU:0"):
    # a = [[1,2,3],[4,5]]
    v0 = tf.constant([1,2,3,4,5], dtype=tf.int64)
    a = tf.RaggedTensor.from_row_splits(v0, row_splits=[0,3,5], validate = validation)
    # b = [[0,1],[1]]
    v1 = tf.constant([0,1,1], dtype=tf.int64) 
    b = tf.RaggedTensor.from_row_splits(v1, row_splits=[0,2,3], validate = validation)

    empty = tf.RaggedTensor.from_row_splits(tf.constant([], dtype=tf.int64), [0], validate = validation) 

    # should be [[1,2,3,0,1],[4,5,1]]
    c_value, c_row_splits = tf.batch_concat_on_rt(a.values, a.row_splits, b.values, b.row_splits)
    c = tf.RaggedTensor.from_row_splits(c_value, c_row_splits, validate = validation)

    # should be [[1,2,3],[4,5]]
    d_value, d_row_splits = tf.batch_concat_on_rt(a.values, a.row_splits, empty.values, empty.row_splits)
    d = tf.RaggedTensor.from_row_splits(d_value, d_row_splits, validate = validation)

    # should be [[0,1],[1]]
    e_value, e_row_splits = tf.batch_concat_on_rt(empty.values, empty.row_splits, b.values, b.row_splits)
    e = tf.RaggedTensor.from_row_splits(e_value, e_row_splits, validate = validation)

    # should be []
    f_value, f_row_splits = tf.batch_concat_on_rt(empty.values, empty.row_splits, empty.values, empty.row_splits)
    f = tf.RaggedTensor.from_row_splits(f_value, f_row_splits, validate = validation)



output = [c,d,e,f]
with tf.Session(graph = g) as sess:
  res = sess.run(output)
  print(res[0].to_list())
  print(res[1].to_list())
  print(res[2].to_list())
  print(res[3].to_list())

#print(g.as_graph_def())

from google.protobuf import text_format

def save_pbtxt(pb, file_name):
  #with open(file_name, "w") as f:
  with tf.gfile.GFile(file_name, 'w') as f:
    f.write(text_format.MessageToString(pb))

save_pbtxt(g.as_graph_def(), "./g.pbtxt")

