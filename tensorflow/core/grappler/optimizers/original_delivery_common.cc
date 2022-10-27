/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/original_delivery_common.h"
#include <fstream>

namespace tensorflow {
namespace grappler {


Status NodeDefConstructor(NodeDef& def, const NodeDef& base,
                          const std::function<Status(NodeDef&)>& node_builder) {
  TF_RETURN_IF_ERROR(node_builder(def));
  def.set_device(base.device());
  VLOG(1) << def.DebugString();
  return Status::OK();
}

Node* NodeConstructor(Graph* graph, string name, const Node* base, const NodeDef& def) {
  Status status;
  Node *node = graph->AddNode(def, &status);
  TF_RETURN_NULL_IF_ERROR(status, name)
  node->set_assigned_device_name(base->assigned_device_name());
  return node;
}

Node* NodeConstructor(Graph* graph, string name, Node* base,
                     const std::function<Status(NodeDef&)>& node_builder) {
  NodeDef def;
  Status status = node_builder(def);
  TF_RETURN_NULL_IF_ERROR(status, name)
  def.set_device(base->def().device());
  VLOG(1) << def.DebugString();
  Node *node = graph->AddNode(def, &status);
  TF_RETURN_NULL_IF_ERROR(status, name)
  node->set_assigned_device_name(base->assigned_device_name());
  return node;
}

Status CreateConstNodeDef(NodeDef& def, string const_name, Tensor &t_const,
                          const NodeDef& base) {
  std::function<Status(NodeDef&)> const_builder = [&](NodeDef& def) {
    return NodeDefBuilder(const_name, "Const")
                          .Attr("dtype", t_const.dtype())
                          .Attr("value", t_const)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, const_builder));
  return Status::OK();
}

Node* CreateConstNode(Graph* graph, string const_name, Tensor &t_const, Node* base) {
  NodeDef def;
  Status status = CreateConstNodeDef(def, const_name, t_const, base->def());
  TF_RETURN_NULL_IF_ERROR(status, const_name)
  return NodeConstructor(graph, const_name, base, def);
}

Status ConstructSliceNodeDef(NodeDef& slice, const NodeDef& input, const NodeDef& begin,
                            const NodeDef& size, string name, DataType output_type) {
  std::vector<NodeDefBuilder::NodeOut> slice_inputs;
  slice_inputs.emplace_back(input.name(), 0, output_type);
  slice_inputs.emplace_back(begin.name(), 0, DT_INT64);
  slice_inputs.emplace_back(size.name(), 0, DT_INT64);
  std::function<Status(NodeDef&)> slice_builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "Slice")
                          .Input(slice_inputs[0])
                          .Input(slice_inputs[1])
                          .Input(slice_inputs[2])
                          .Attr("T", output_type)
                          .Attr("Index", DT_INT64)
                          .Finalize(&slice);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(slice, input, slice_builder));
  return Status::OK();
}

Node* ConstructSliceOp(Graph* graph, Node* base, int port, string name,
                       Tensor& t_begin, Tensor& t_size) {
  string begin_name = name + "_begin";
  Node* begin = CreateConstNode(graph, begin_name, t_begin, base);
  TF_RETURN_NULL_IF_NULL(begin, "construct begin const node")
  string size_name = name + "_size";
  Node* size = CreateConstNode(graph, size_name, t_size, base);
  TF_RETURN_NULL_IF_NULL(size, "construct size const node")

  NodeDef def;
  Status status = ConstructSliceNodeDef(def, base->def(), begin->def(), size->def(),
                                        name, base->output_type(0));
  TF_RETURN_NULL_IF_ERROR(status, name)
  Node* slice = NodeConstructor(graph, name, base, def);
  TF_RETURN_NULL_IF_NULL(slice, "construct slice node")
  graph->AddEdge(base, port, slice, 0);
  graph->AddEdge(begin, 0, slice, 1);
  graph->AddEdge(size, 0, slice, 2);
  return slice;
}

Status ConstuctConcatNodeDef(NodeDef& concat, const NodeDef& base, string name,
                             std::vector<NodeDefBuilder::NodeOut>& inputs,
                             NodeDefBuilder::NodeOut& idx, int concat_n,
                             DataType t_input, DataType t_idx) {
  std::function<Status(NodeDef&)> concat_builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "ConcatV2")
                          .Input(inputs)
                          .Input(idx)
                          .Attr("N", concat_n)
                          .Attr("T", t_input)
                          .Attr("Tidx", t_idx)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(concat, base, concat_builder));
  return Status::OK();
}

Node* ConstructConcatOp(Graph* graph, Node* base, int64 axis,
                     std::vector<Node*>& input_nodes, string concat_name) {
  string idx_name = concat_name + "_indice";
  Tensor t(DT_INT64, TensorShape({}));
  t.scalar<int64>()() = axis;
  Node* idx_const = CreateConstNode(graph, idx_name, t, base);
  TF_RETURN_NULL_IF_NULL(idx_const, "construct indice const")

  int input_size = input_nodes.size();
  VLOG(1) << concat_name << ", size:" << input_size;
  std::vector<NodeDefBuilder::NodeOut> inputs;
  for (auto n:input_nodes) {
    inputs.emplace_back(n->name(), 0, base->output_type(0));
  }
  NodeDef def;
  NodeDefBuilder::NodeOut concat_idx(idx_name, 0, DT_INT64);
  Status status = ConstuctConcatNodeDef(def, base->def(), concat_name, inputs,
                                        concat_idx, input_size,
                                        base->output_type(0), DT_INT64);
  TF_RETURN_NULL_IF_ERROR(status, concat_name)
  Node* concat = NodeConstructor(graph, concat_name, base, def);
  TF_RETURN_NULL_IF_NULL(concat, "construct concat node")

  int port = 0;
  for (auto n:input_nodes) {
    graph->AddEdge(n, 0, concat, port);
    port++;
  }
  return concat;
}

Status ConstuctAddNodeDef(NodeDef& def, const NodeDef& base, string name,
                          std::vector<NodeDefBuilder::NodeOut>& inputs,
                          DataType t_input) {
  std::function<Status(NodeDef&)> add_builder = [&](NodeDef& def) {
    return NodeDefBuilder(base.name(), "Add")
                          .Input(inputs[0])
                          .Input(inputs[1])
                          .Attr("T", t_input)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, add_builder));
  return Status::OK();
}

Status ConstructSplitNodeDef(NodeDef& def, const NodeDef& base, string split_name,
                             NodeDefBuilder::NodeOut& idx,
                             NodeDefBuilder::NodeOut& input, int split_num,
                             DataType t_input) {
  std::function<Status(NodeDef&)> split_builder = [&](NodeDef& def) {
    return NodeDefBuilder(split_name, "Split")
                          .Input(idx)
                          .Input(input)
                          .Attr("num_split", split_num)
                          .Attr("T", t_input)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, split_builder));
  return Status::OK();
}

Node* ConstructSplitOp(Graph* graph, Node* base, int split_num, int axis,
                       std::vector<std::vector<const Edge*>>& out_edges,
                       string split_name) {
  // dim const
  string dim_name = split_name + "_split_dim";
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32>()() = axis;
  Node* dim_const = CreateConstNode(graph, dim_name, t, base);
  TF_RETURN_NULL_IF_NULL(dim_const, "construct split dim const node")

  int out_size = out_edges.size();
  VLOG(1) << split_name << ", size:" << out_size;
  NodeDef def;
  NodeDefBuilder::NodeOut split_idx(dim_name, 0, DT_INT32);
  NodeDefBuilder::NodeOut split_input(base->name(), 0, base->output_type(0));
  Status status = ConstructSplitNodeDef(def, base->def(), split_name, split_idx,
                                        split_input, split_num, base->output_type(0));
  TF_RETURN_NULL_IF_ERROR(status, split_name)
  Node* split = NodeConstructor(graph, split_name, base, def);
  TF_RETURN_NULL_IF_NULL(split, "construct split node")
  int port = 0;
  graph->AddEdge(dim_const, 0, split, 0);
  graph->AddEdge(base, 0, split, 1);
  for (auto set:out_edges) {
    for (auto e:set) {
      status = graph->UpdateEdge(split, port, e->dst(), e->dst_input());
      TF_RETURN_NULL_IF_ERROR(status, "update split out edge failed")
    }
    port++;
  }
  return split;
}

Status ConstructPackNodeDef(NodeDef& def, const NodeDef& base, string pack_name,
                             std::vector<NodeDefBuilder::NodeOut>& inputs,
                             int input_size, DataType t_input) {
  std::function<Status(NodeDef&)> builder = [&](NodeDef& def) {
    return NodeDefBuilder(pack_name, "Pack")
                          .Input(inputs)
                          .Attr("T", t_input)
                          .Attr("N", input_size)
                          .Attr("axis", 0)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, builder));
  return Status::OK();
}

Node* ConstructPackOp(Graph* graph, Node* base, string pack_name,
                       std::vector<const Edge*>& input_edges) {
  std::vector<NodeDefBuilder::NodeOut> pack_inputs;
  for (auto e:input_edges) {
    pack_inputs.emplace_back(e->src()->name(), e->src_output(),
                             base->output_type(0));
  }
  int input_size = input_edges.size();
  NodeDef def;
  Status status = ConstructPackNodeDef(def, base->def(), pack_name, pack_inputs,
                                       input_size, base->output_type(0));
  TF_RETURN_NULL_IF_ERROR(status, pack_name)
  Node* pack = NodeConstructor(graph, pack_name, base, def);
  TF_RETURN_NULL_IF_NULL(pack, "construct pack node")
  int port = 0;
  for (auto e:input_edges) {
    graph->AddEdge(e->src(), e->src_output(), pack, port);
    port++;
  }
  return pack;
}

Status ConstructTransposeNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 NodeDefBuilder::NodeOut& perm,
                                 DataType t_input, DataType t_perm) {
  std::function<Status(NodeDef&)> builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "Transpose")
                          .Input(input)
                          .Input(perm)
                          .Attr("T", t_input)
                          .Attr("Tperm", t_perm)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, builder));
  return Status::OK();
}

Node* ConstructTransposeOp(Graph* graph, const Edge* in_edge,
                           string name, Tensor& perm_t) {
  // perm const
  string perm_name = name + "_perm";
  Node* perm_const = CreateConstNode(graph, perm_name, perm_t, in_edge->src());
  TF_RETURN_NULL_IF_NULL(perm_const, "construct perm const node")

  NodeDef def;
  NodeDefBuilder::NodeOut input(in_edge->src()->name(), in_edge->src_output(),
                                in_edge->src()->output_type(0));
  NodeDefBuilder::NodeOut perm(perm_const->name(), 0, DT_INT32);
  Status status = ConstructTransposeNodeDef(def, in_edge->src()->def(), name,input,
                                            perm, in_edge->src()->output_type(0),
                                            perm_t.dtype());
  TF_RETURN_NULL_IF_ERROR(status, name)
  Node* transpose = NodeConstructor(graph, name, in_edge->src(), def);
  TF_RETURN_NULL_IF_NULL(transpose, "construct transpose node")
  graph->AddEdge(in_edge->src(), in_edge->src_output(), transpose, 0);
  graph->AddEdge(perm_const, 0, transpose, 1);
  return transpose;
}

Status ConstructReshapeNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 NodeDefBuilder::NodeOut& perm,
                                 DataType t_input, DataType t_shape) {
  std::function<Status(NodeDef&)> builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "Reshape")
                          .Input(input)
                          .Input(perm)
                          .Attr("T", t_input)
                          .Attr("Tshape", t_shape)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, builder));
  return Status::OK();
}

Node* ConstructReshapeOp(Graph* graph, Node* base, string name, Tensor& shape_t) {
  // shape const
  string shape_name = name + "_shape";
  Node* shape_const = CreateConstNode(graph, shape_name, shape_t, base);
  TF_RETURN_NULL_IF_NULL(shape_const, "construct shape const node")

  NodeDefBuilder::NodeOut input(base->name(), 0, base->output_type(0));
  NodeDefBuilder::NodeOut shape(shape_const->name(), 0, shape_t.dtype());
  NodeDef def;
  Status status = ConstructReshapeNodeDef(def, base->def(), name, input, shape,
                                          base->output_type(0), shape_t.dtype());
  TF_RETURN_NULL_IF_ERROR(status, name)
  Node* reshape = NodeConstructor(graph, name, base, def);
  TF_RETURN_NULL_IF_NULL(reshape, "construct reshape node")
  graph->AddEdge(base, 0, reshape, 0);
  graph->AddEdge(shape_const, 0, reshape, 1);
  return reshape;
}

Status ConstructCastNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 DataType src, DataType dst) {
  std::function<Status(NodeDef&)> builder = [&](NodeDef& def) {
    return NodeDefBuilder(name, "Cast")
                          .Input(input)
                          .Attr("SrcT", src)
                          .Attr("DstT", dst)
                          .Finalize(&def);
  };
  TF_RETURN_IF_ERROR(NodeDefConstructor(def, base, builder));
  return Status::OK();
}

Node* ConstuctCastOp(Graph* graph, const Node* base, int port,
                     DataType src, DataType dst, string cast_name) {
  NodeDef def;
  NodeDefBuilder::NodeOut input(base->name(), port, src);
  Status status = ConstructCastNodeDef(def, base->def(), cast_name,
                                       input, src, dst);
  TF_RETURN_NULL_IF_ERROR(status, cast_name)
  Node* cast = NodeConstructor(graph, cast_name, base, def);
  TF_RETURN_NULL_IF_NULL(cast, "construct cast node")
  VLOG(1) << "cast " << cast->DebugString();
  return cast;
}

Status UpdateAllEdge(Graph* graph, Node* new_src_node, Node* old_dst_node) {
  std::vector<Node*> dst_nodes;
  std::vector<int> dst_inputs;
  std::vector<int> src_outputs;
  for (const Edge* e : old_dst_node->out_edges()) {
    dst_nodes.push_back(e->dst());
    dst_inputs.push_back(e->dst_input());
    src_outputs.push_back(e->src_output());
  }
  for (unsigned int i = 0; i < dst_nodes.size(); i++) {
    TF_RETURN_IF_ERROR(
        graph->UpdateEdge(new_src_node, src_outputs[i], dst_nodes[i], dst_inputs[i]));
  }
  return Status::OK();
}

void DumpModelFile(const GraphDef& graph, string file) {
  std::fstream f;
  f.open(file, std::fstream::out);
  f << graph.SerializeAsString();
  f.close();
}


}  // end namespace grappler
}  // end namespace tensorflow

