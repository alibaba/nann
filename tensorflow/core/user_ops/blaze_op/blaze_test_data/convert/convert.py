#!/usr/bin/env python
# encoding: utf-8

import copy
import sys
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
graph = './frozen_graph.pb.bak'
meta = './0.meta'
sig_key = 'predict'
option_path = 'blaze_option_path'

def create_node_def(op, name, inputs):
  new_node = node_def_pb2.NodeDef()
  new_node.op = op
  new_node.name = name
  new_node.input.extend(inputs)
  return new_node

input_attr_name = 'input_names'
output_attr_name = 'output_names'
input_type = 'InT'
output_type = 'OutT'
graph_name = 'graph_def'

def add_put(put_name, put_type, puts, blaze_node, node_map):
  # add inputs & input types
  for put in puts:
      name = put.split(':')[0]
     # print(node_map[name])
      blaze_node.attr[put_name].list.s.append(compat.as_bytes(name))
      blaze_node.attr[put_type].list.type.append(node_map[name].attr['dtype'].type)

option_xla_path = '/home/jingshan.ljs/tensorflow/tensorflow/core/kernels/blaze_test_data/convert/succ_options'
def convert():
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    pbstr = gfile.Open(meta).read()
    text_format.Parse(pbstr, meta_graph_def)

    inputs = []
    input_bytes = []
    input_types = []
    outputs = []
    output_bytes = []
    output_types = []
    sig_def = meta_graph_def.signature_def['predict']
    for key, value in sig_def.inputs.items():
        inputs.append(value.name)
        input_bytes.append(compat.as_bytes(value.name.split(':')[0]))
        input_types.append(value.dtype)

    for key, value in sig_def.outputs.items():
        outputs.append(value.name)
        output_bytes.append(compat.as_bytes(value.name.split(':')[0]))
        output_types.append(value.dtype)

    gdef = graph_pb2.GraphDef()
    with open(graph, 'rb') as fh:
        graph_str = fh.read()
        gdef.ParseFromString(graph_str)
        #print(gdef)

    output_def = copy.deepcopy(gdef)
    output_def.ClearField('node')
    node_map = {}
    for node in gdef.node:
        node_map[node.name] = node
    
    blaze_node = create_node_def('BlazeXlaOp', 'blaze_op', inputs)
    blaze_node.attr[input_attr_name].list.s[:] = input_bytes
    blaze_node.attr[input_type].list.type[:] = input_types

    blaze_node.attr[output_attr_name].list.s[:] = output_bytes
    blaze_node.attr[output_type].list.type[:] = output_types
    blaze_node.attr[graph_name].s = compat.as_bytes(('/home/jingshan.ljs/tensorflow/tensorflow/core/kernels/blaze_test_data/convert/ss'))
    blaze_node.attr[option_path].s = compat.as_bytes(option_xla_path)

    output_def.node.append(blaze_node)

    for input in inputs:
        output_def.node.append(node_map[input.split(':')[0]])

    for i in range(len(outputs)):
        output = outputs[i]
        name = output.split(':')[0]
        ip_name = 'blaze_op'
        if i != 0:
            ip_name = ip_name + ':' + str(i)
        ips = [ip_name]
        opt = create_node_def("Identity", name, ips)
        opt.attr['T'].type = output_types[i]
        output_def.node.append(opt)
    print(output_def)
convert()
