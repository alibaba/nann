import tensorflow as tf
import numpy as np
import argparse
import os

def get_args():
    # Parse commandline
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True,
                        help="TensorFlow variables file to load.")
    return parser.parse_args()

args = get_args()

def save_pbtxt(pb, file_name):
  from google.protobuf import text_format
  with open(file_name, "w") as f:
    f.write(text_format.MessageToString(pb))

def main():
  runmeta = tf.RunMetadata()
  infos = runmeta.tensor_infos.name_tensors
  infos.add(name = "level_topn", tensor = tf.make_tensor_proto([100, 200, 400, 400, 400, 200]))

  '''
  comm_seq [1, 50*64]
  '''
  comm_seq_data = np.random.rand(1, 50*64)*0.01
  comm_seq_data = comm_seq_data.astype(np.float16)
  infos.add(name = "comm_seq", tensor = tf.make_tensor_proto(comm_seq_data))
  save_pbtxt(runmeta, os.path.join(args.output_dir, 'mock.runmeta'))

  print("gen runmeta done!")

if __name__ == "__main__":
    main()
