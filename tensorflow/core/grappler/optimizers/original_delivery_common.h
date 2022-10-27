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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ORIGINAL_DELIVERY_COMMON_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ORIGINAL_DELIVERY_COMMON_H_

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include <fstream>

namespace tensorflow {
namespace grappler {


#define TF_RETURN_NULL_IF_ERROR(status, msg) \
  if (!status.ok()) {                        \
    LOG(ERROR) << msg << " error, "          \
               << status.ToString();         \
    return nullptr;                          \
  }

#define TF_RETURN_NULL_IF_NULL(ptr, msg) \
  if (ptr == nullptr) {                        \
    LOG(ERROR) << msg << " encounter nullptr"; \
    return nullptr;                          \
  }

#define TF_RETURN_FALSE_IF_ERROR(status, msg) \
  if (!status.ok()) {                        \
    LOG(ERROR) << msg << " error, "          \
               << status.ToString();         \
    return false;                            \
  }

#define TF_RETURN_FALSE_IF_NULL(ptr, msg)        \
  if (ptr == nullptr) {                       \
    LOG(ERROR) << msg << " encounter nullptr"; \
    return false;                              \
  }

Status NodeDefConstructor(NodeDef& def, const NodeDef& base,
                          const std::function<Status(NodeDef&)>& node_builder);

Node* NodeConstructor(Graph* graph, string name, const Node* base, const NodeDef& def);

Node* NodeConstructor(Graph* graph, string name, Node* base,
                     const std::function<Status(NodeDef&)>& node_builder);

Status CreateConstNodeDef(NodeDef& def, string const_name, Tensor &t_const,
                          const NodeDef& base);
Node* CreateConstNode(Graph* graph, string const_name, Tensor &t_const, Node* base);

Status ConstructSliceNodeDef(NodeDef& slice, const NodeDef& input, const NodeDef& begin,
                            const NodeDef& size, string name, DataType output_type);

Node* ConstructSliceOp(Graph* graph, Node* base, int port, string name,
                       Tensor& t_begin, Tensor& t_size);

Status ConstuctConcatNodeDef(NodeDef& concat, const NodeDef& base, string name,
                             std::vector<NodeDefBuilder::NodeOut>& inputs,
                             NodeDefBuilder::NodeOut& idx, int concat_n,
                             DataType t_input, DataType t_idx);
Node* ConstructConcatOp(Graph* graph, Node* base, int64 axis,
                     std::vector<Node*>& input_nodes, string concat_name);

Status ConstuctAddNodeDef(NodeDef& def, const NodeDef& base, string name,
                          std::vector<NodeDefBuilder::NodeOut>& inputs,
                          DataType t_input);

Status ConstructSplitNodeDef(NodeDef& def, const NodeDef& base, string split_name,
                             NodeDefBuilder::NodeOut& idx,
                             NodeDefBuilder::NodeOut& input, int split_num,
                             DataType t_input);
Node* ConstructSplitOp(Graph* graph, Node* base, int split_num, int axis,
                       std::vector<std::vector<const Edge*>>& out_edges,
                       string split_name);

Status ConstructPackNodeDef(NodeDef& def, const NodeDef& base, string pack_name,
                             std::vector<NodeDefBuilder::NodeOut>& inputs,
                             int input_size, DataType t_input);
Node* ConstructPackOp(Graph* graph, Node* base, string pack_name,
                       std::vector<const Edge*>& input_edges);

Status ConstructTransposeNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 NodeDefBuilder::NodeOut& perm,
                                 DataType t_input, DataType t_perm);
Node* ConstructTransposeOp(Graph* graph, const Edge* in_edge,
                           string name, Tensor& perm_t);

Status ConstructReshapeNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 NodeDefBuilder::NodeOut& perm,
                                 DataType t_input, DataType t_shape);
Node* ConstructReshapeOp(Graph* graph, Node* base, string name, Tensor& shape_t);

Status ConstructCastNodeDef(NodeDef& def, const NodeDef& base, string name,
                                 NodeDefBuilder::NodeOut& input,
                                 DataType src, DataType dst);

Node* ConstuctCastOp(Graph* graph, const Node* base, int port,
                     DataType src, DataType dst, string cast_name);

Status UpdateAllEdge(Graph* graph, Node* new_src_node, Node* old_dst_node);

void DumpModelFile(const GraphDef& graph, string file);


}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ORIGINAL_DELIVERY_COMMON_H_
