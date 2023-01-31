#!/usr/bin/env python
# encoding: utf-8

import argparse

import tensorflow as tf
from tensorflow.saved_model import signature_constants
from tensorflow.saved_model import tag_constants

from nann.util import load_meta_graph

def parse_opt():
    parser = argparse.ArgumentParser("benchmark")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--meta-path", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    return parser.parse_args()


def convert_pb_to_saved_model(model_path, meta_path, export_dir, inputs, outputs):
    tf.reset_default_graph()
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf.Session() as sess:


        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        meta_graph = load_meta_graph(meta_path)
        input_tensor_names = [meta_graph.signature_def['predict'].inputs[key].name for key in inputs]
        input_tensors_dict = {key: sess.graph.get_tensor_by_name(name) for key, name in zip(inputs, input_tensor_names)}

        output_tensor_names = [meta_graph.signature_def['predict'].outputs[key].name for key in outputs]
        output_tensors_dict = {key: sess.graph.get_tensor_by_name(name) for key, name in zip(outputs, output_tensor_names)}

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(inputs=input_tensors_dict, outputs=output_tensors_dict)

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()


if __name__ == '__main__':
    args = parse_opt()
    convert_pb_to_saved_model(args.model_path, args.meta_path, args.export_dir, inputs=['comm_seq', 'level_topn'], outputs=['top_k'])
