from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy
import os
import re


from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.tools.graph_transforms import TransformGraph

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nann.delivery.constant import native_delivery_input_prefix, native_delivery_output_prefix 
from nann.util import load_meta_graph, save_pb

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib

FLAGS = None

def freeze_graph(input_graph_def,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_blacklist=""):
  """Converts all variables in a graph and checkpoint into constants."""

  if input_saver and not gfile.Exists(input_saver):
    print("Input saver file '" + input_saver + "' does not exist!")
    return -1

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    return -1

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ""
  _ = importer.import_graph_def(input_graph_def, name="")

  with session.Session() as sess:
    if input_saver:
      with gfile.FastGFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
          saver_def.ParseFromString(f.read())
        else:
          text_format.Merge(f.read(), saver_def)
        saver = saver_lib.Saver(saver_def=saver_def)
        saver.restore(sess, input_checkpoint)
    else:
      sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
      if initializer_nodes:
        sess.run(initializer_nodes)

    variable_names_blacklist = (variable_names_blacklist.split(",") if
                                variable_names_blacklist else None)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names,
        variable_names_blacklist=variable_names_blacklist)
  
  return output_graph_def

def _check_tensor_compatible(source_tensor, target_tensor):
  expect_type = source_tensor.dtype
  actual_type = target_tensor.dtype
  if not expect_type.is_compatible_with(actual_type):
    print('input %s -> %s dtype not match, '
          'expect %s, actual %s' % (
            str(source_tensor), str(target_tensor),
            expect_type, actual_type))
    # allow ref type. may replace AssignOp, but not used online
    if not expect_type.base_dtype.is_compatible_with(actual_type.base_dtype):
      raise Exception('incompatible type between %s and %s' %
                      (str(source_tensor), str(target_tensor)))
  # item/query input may not match, fix this later
  expect_shape = source_tensor.shape
  actual_shape = target_tensor.shape
  if not expect_shape.is_compatible_with(actual_shape):
    print('input %s -> %s shape not match, '
          'expect %s, actual %s' % (
            source_tensor, target_tensor.name,
            expect_shape, actual_shape))
    if (len(expect_shape.dims) == 1 and len(actual_shape.dims) == 2 and
        expect_shape.dims[0].is_compatible_with(actual_shape.dims[0]) and
        actual_shape.dims[1] == 1):
      pass
    # walk around for rnn
    elif target_tensor.op.node_def.op == 'IndexLookupOp':
      pass


def RerouteTensor(t0, t1, can_modify=None):
  """Reroute the end of the tensor t0 to the ends of the tensor t1.
  """
  if (isinstance(t0, tf.SparseTensor) and
      isinstance(t1, tf.SparseTensor)):
    RerouteTensor(t0.indices, t1.indices)
    RerouteTensor(t0.values, t1.values)
    RerouteTensor(t0.dense_shape, t1.dense_shape)
    return

  if (isinstance(t0, list) and
      isinstance(t1, list)):
    for idx, t in enumerate(t0):
      RerouteTensor(t, t1[idx])
    return

  nb_update_inputs = 0
  consumers = copy.copy(t1.consumers())
  if can_modify is not None:
    consumers = [c for c in consumers if c in can_modify]
  consumers_indices = {}
  for c in consumers:
    consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
  for c in consumers:
    for i in consumers_indices[c]:
      _check_tensor_compatible(c.inputs[i], t0)
      c._update_input(i, t0)  # pylint: disable=protected-access
    nb_update_inputs += 1
  return nb_update_inputs


def del_repeated_nodes(graph):
  graph_def = copy.deepcopy(graph)
  name_set = set()
  to_del = []
  new_map = {}
  for node in graph_def.node:
    new_map[node.name] = node
    if node.name in name_set:
      to_del.append(node)
    else:
      name_set.add(node.name)
  for node in to_del:
    graph_def.node.remove(node)

  for node in graph_def.node:
    if '_class' in node.attr and node.attr['_class'].list.s:
      to_remove = []
  # item/query input may not match, fix this later
  expect_shape = source_tensor.shape
  actual_shape = target_tensor.shape
  if not expect_shape.is_compatible_with(actual_shape):
    print('input %s -> %s shape not match, '
          'expect %s, actual %s' % (
            source_tensor, target_tensor.name,
            expect_shape, actual_shape))
    if (len(expect_shape.dims) == 1 and len(actual_shape.dims) == 2 and
        expect_shape.dims[0].is_compatible_with(actual_shape.dims[0]) and
        actual_shape.dims[1] == 1):
      pass
    # walk around for rnn
    elif target_tensor.op.node_def.op == 'IndexLookupOp':
      pass


def RerouteTensor(t0, t1, can_modify=None):
  """Reroute the end of the tensor t0 to the ends of the tensor t1.
  """
  if (isinstance(t0, tf.SparseTensor) and
      isinstance(t1, tf.SparseTensor)):
    RerouteTensor(t0.indices, t1.indices)
    RerouteTensor(t0.values, t1.values)
    RerouteTensor(t0.dense_shape, t1.dense_shape)
    return

  if (isinstance(t0, list) and
      isinstance(t1, list)):
    for idx, t in enumerate(t0):
      RerouteTensor(t, t1[idx])
    return

  nb_update_inputs = 0
  consumers = copy.copy(t1.consumers())
  if can_modify is not None:
    consumers = [c for c in consumers if c in can_modify]
  consumers_indices = {}
  for c in consumers:
    consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
  for c in consumers:
    for i in consumers_indices[c]:
      _check_tensor_compatible(c.inputs[i], t0)
      c._update_input(i, t0)  # pylint: disable=protected-access
    nb_update_inputs += 1
  return nb_update_inputs


def del_repeated_nodes(graph):
  graph_def = copy.deepcopy(graph)
  name_set = set()
  to_del = []
  new_map = {}
  for node in graph_def.node:
    new_map[node.name] = node
    if node.name in name_set:
      to_del.append(node)
    else:
      name_set.add(node.name)
  for node in to_del:
    graph_def.node.remove(node)

  for node in graph_def.node:
    if '_class' in node.attr and node.attr['_class'].list.s:
      to_remove = []
      for s in node.attr['_class'].list.s:
        news = s.decode("utf-8")
        if news.startswith('loc'):
          to_remove.append(s)

      for need_remove_s in to_remove:
        node.attr['_class'].list.s.remove(need_remove_s)
  return graph_def


def replace_variable(reader, graph_def):
  graph = tf.Graph()
  type_map = reader.get_variable_to_dtype_map()
  shape_map = reader.get_variable_to_shape_map()
  with graph.as_default() as g:
    tf.import_graph_def(graph_def, name='')
    for node in graph_def.node:
      if node.op == 'VariableV2' or node.op == 'Variable':
        p_node = g.get_tensor_by_name(node.name + ':0')
        new_node = tf.constant(reader.get_tensor(node.name),
                               type_map.get(node.name), shape_map.get(node.name), node.name)
        RerouteTensor(new_node, p_node)
    return g.as_graph_def()


def is_placeholder(node):
  if node.op == "Placeholder" or node.op == "PlaceholderWithDefault":
    return True
  return False

def copy_placeholder_attr(input_graph_def, output_graph_def, input_names):
    node_map = {}
    for node in input_graph_def.node:
        if node.name in input_names:
            node_map[node.name] = node
    for node in output_graph_def.node:
        if node.name in node_map:
            if node_map[node.name].op == "Placeholder" or node_map[node.name].op == "PlaceholderWithDefault":
                node.attr['dtype'].CopyFrom(node_map[node.name].attr['dtype'])
                node.attr['shape'].CopyFrom(node_map[node.name].attr['shape'])
                print(node.op, node.name, node.attr['shape'] ,"copy and convert from input_graph" )
            else:
                print('cant get placeholder dtype and shape info', node_map[node.name])
    return output_graph_def

def convert_placeholder(graph_def):
  out_graph_def = graph_pb2.GraphDef()
  placeholder_withdefault_nodes = []
  for node in graph_def.node:
    if node.op == "PlaceholderWithDefault":
      placeholder_withdefault_nodes.append(node)
      continue
    out_graph_def.node.extend([copy.deepcopy(node)])

  extend_list = []
  for node in placeholder_withdefault_nodes:
    print(node.name + 'convert PlaceholderWithDefault to Placeholder')
    placeholder = create_node_def("Placeholder", node.name, [])
    placeholder.attr['dtype'].CopyFrom(node.attr['dtype'])
    print(node.name, node.attr['shape'])
    placeholder.attr['shape'].CopyFrom(node.attr['shape'])
    extend_list.append(placeholder)
  for n in extend_list:
    out_graph_def.node.extend([copy.deepcopy(n)])
  out_graph_def.library.CopyFrom(graph_def.library)
  out_graph_def.versions.CopyFrom(graph_def.versions)
  return out_graph_def

def create_node_def(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    new_node.input.extend(inputs)
    return new_node

def set_attr_shape(node, shape):
    node.attr['shape'].CopyFrom(
            attr_value_pb2.AttrValue(shape=shape))

def set_attr_dtype(node, key, value):
    if isinstance(value, int):
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(type=value))
    else:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(type=value.as_datatype_enum))

def set_attr_tensor(node, key, value, dtype, shape=None):
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                value, dtype=dtype, shape=shape)))

def convert_feed_input_float2half(graph_def):
    out_graph_def = graph_pb2.GraphDef()
    float_nodes = []
    for node in graph_def.node:
        if node.op == "Placeholder":
            if node.attr['dtype'].type == dtypes.float32.as_datatype_enum:
                float_nodes.append(node)

    extend_list = []
    # set placeholder dtype to half
    for node in float_nodes:
        set_attr_dtype(node, "dtype", dtypes.float16)
        cast = create_node_def("Cast", node.name + '/cast_float2half', [node.name])
        set_attr_dtype(cast, "SrcT", dtypes.float16)
        set_attr_dtype(cast, "DstT", dtypes.float32)
        extend_list.append(cast)
        print(node.name + ' convert float to half')

    for node in graph_def.node:
        for i in range(len(node.input)):
            name = node.input[i]
            for ph in float_nodes:
                if name.count(ph.name):
                    node.input[i] = ph.name + '/cast_float2half'

        out_graph_def.node.extend([copy.deepcopy(node)])

    for n in extend_list:
        out_graph_def.node.extend([copy.deepcopy(n)])
    out_graph_def.library.CopyFrom(graph_def.library)
    out_graph_def.versions.CopyFrom(graph_def.versions)
    return out_graph_def

def main():
  model_restore_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  reader = pywrap_tensorflow.NewCheckpointReader(model_restore_path)
  meta_graph = load_meta_graph(model_restore_path + '.meta')
  # remove replica cause by data parallel
  r1 = re.compile(r".*_(\d+)$")
  

  input_names = [n.name for n in meta_graph.graph_def.node if is_placeholder(n) and n.name.count(native_delivery_input_prefix) and not r1.search(n.name)]
  output_names = [n.name for n in meta_graph.graph_def.node if n.name.count(native_delivery_output_prefix) and not r1.search(n.name)]

  print(f'feed input names: {input_names}')
  print(f'feed output names: {output_names}')

  con_graph = del_repeated_nodes(meta_graph.graph_def)
  no_var_graph = freeze_graph(con_graph, FLAGS.input_saver, FLAGS.input_binary,
                 model_restore_path, output_names,
                 FLAGS.restore_op_name, FLAGS.filename_tensor_name,
                 FLAGS.output_graph, FLAGS.clear_devices, FLAGS.initializer_nodes,
                 FLAGS.variable_names_blacklist)
  transforms = [
       "add_default_attributes",
       "strip_unused_nodes",
       "merge_duplicate_nodes",
       "remove_nodes(op=CheckNumerics,op=StopGradient)",
       "fold_constants(ignore_errors=true)",
       "sort_by_execution_order"]

  # strip unused nodes in graph
  output_graph_def = TransformGraph(no_var_graph, input_names, output_names,
                                    transforms=transforms)
  print('extract dense subgraph %d/%d ' % (len(output_graph_def.node), len(no_var_graph.node)))

  output_graph_def.library.Clear()
  output_graph_def = copy_placeholder_attr(no_var_graph, output_graph_def, input_names)
  output_graph_def = convert_feed_input_float2half(output_graph_def)
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  save_pb(output_graph_def, os.path.join(FLAGS.output_dir, 'frozen_graph.pb'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="input model.")
  parser.add_argument(
      "--output_dir",
      type=str,
      default="",
      help="output_dir.")
  parser.add_argument(
      "--input_saver",
      type=str,
      default="",
      help="TensorFlow saver file to load.")
  parser.add_argument(
      "--output_graph",
      type=str,
      default="",
      help="Output \'GraphDef\' file name.")
  parser.add_argument(
      "--input_binary",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="Whether the input files are in binary format.")
  parser.add_argument(
      "--restore_op_name",
      type=str,
      default="save/restore_all",
      help="The name of the master restore operator.")
  parser.add_argument(
      "--filename_tensor_name",
      type=str,
      default="save/Const:0",
      help="The name of the tensor holding the save path.")
  parser.add_argument(
      "--clear_devices",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="Whether to remove device specifications.")
  parser.add_argument(
      "--initializer_nodes",
      type=str,
      default="",
      help="comma separated list of initializer nodes to run before freezing.")
  parser.add_argument(
      "--variable_names_blacklist",
      type=str,
      default="",
      help="""\
      comma separated list of variables to skip converting to constants\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  main()
