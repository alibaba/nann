import argparse

import numpy as np
import tensorflow as tf

from inference_wrapper import tf_predict

parser = argparse.ArgumentParser("benchmark")
parser.add_argument("--graph-file", type=str, required=True)
parser.add_argument("--meta-path", type=str, required=True)
args = parser.parse_args()

scale = 0.1


@tf_predict(['comm_seq', 'level_topn'], ['top_k'], args.meta_path)
def run():
  with tf.io.gfile.GFile(args.graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  comm_seq_data = scale * np.random.normal(size=[10, 64 * 50]).astype(np.float16)

  feed_data = []
  for comm_seq in comm_seq_data:
    comm_seq = comm_seq.reshape((1, 64 * 50))
    level_topn = np.array([10, 10, 10, 10, 10, 10]).astype(np.int32)
    feed_data.append([comm_seq, level_topn])
  tf.import_graph_def(graph_def, name="")
  return feed_data


if __name__ == '__main__':
  np.random.seed(0)
  result = run()
  print(result)
