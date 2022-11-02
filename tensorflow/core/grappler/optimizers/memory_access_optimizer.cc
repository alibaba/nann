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

#include "tensorflow/core/grappler/optimizers/memory_access_optimizer.h"

#include <fstream>
#include <queue>
#include <map>
#include <algorithm>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

namespace {

Status OptimizePatternFunction(const NodeDef& compute_node,
                               const NodeDef& gather_node,
                               const NodeDef& other_node,
                               const NodeDef& gather_input_node,
                               const NodeDef& gather_ind_node,
                               const NodeDef& gather_axis_node,
                               std::vector<NodeDef>* new_nodes,
                               int input_port,
                               int& counter) {
  bool invalid = false;
  if (gather_ind_node.name().find("user_creative_indicator") == string::npos) {
    VLOG(1) << "gather input indicator placeholder not match:" << gather_ind_node.name();
    invalid = true;
  }
  if (compute_node.op() == "MatMul") {
    bool transpose_a = compute_node.attr().at("transpose_a").b();
    bool transpose_b = compute_node.attr().at("transpose_b").b();
    if (transpose_a || transpose_b) {
      invalid = true;
    }
  }

  Tensor gather_axis_tensor = GetNodeTensorAttr(gather_axis_node, "value");
  auto const_dtype = gather_axis_node.attr().at("dtype").type();
  if (const_dtype == DT_INT32) {
    auto gather_axis_value = gather_axis_tensor.flat<int32>();
    if (gather_axis_value(0) != 0) {
      LOG(WARNING) << "gather axis const value not valid:"
                   << gather_axis_node.DebugString();
      invalid = true;
    }
  } else if (const_dtype == DT_INT64) {
    auto gather_axis_value = gather_axis_tensor.flat<int64>();
    if (gather_axis_value(0) != 0) {
      LOG(WARNING) << "gather axis const value not valid:"
                   << gather_axis_node.DebugString();
      invalid = true;
    }
  } else {
    LOG(WARNING) << "gather axis const dtype is not int:" << const_dtype;
    invalid = true;
  }
  // 将Gather替换为Identity，dependency optimization会优化掉
  DataType output_type = DT_FLOAT;
  if (gather_node.attr().count("Tindices") != 0) {
    output_type = gather_node.attr().at("Tindices").type();
  } else {
    invalid = true;
  }
  if (invalid) {
    static int index = 0;
    // 不替换时，直接返回，会导致之后每次都匹配到这个不满足条件的pattern，
    // 其余pattern无法继续匹配，因此给这部分子图增加一个Identity，改变图结构
    NodeDef new_identity_node;
    new_identity_node.set_name(gather_ind_node.name() + "_identity_" + std::to_string(index));
    new_identity_node.set_op("Identity");
    new_identity_node.set_device(gather_node.device());
    new_identity_node.clear_attr();
    (*new_identity_node.mutable_attr())["T"].set_type(output_type);
    *(new_identity_node.mutable_input()->Add()) = gather_ind_node.name();

    NodeDef new_gather_node;
    new_gather_node.CopyFrom(gather_node);
    *(new_gather_node.mutable_input(1)) = new_identity_node.name();
    VLOG(1) << "insert Identity before Gather" << new_identity_node.DebugString();
    new_nodes->push_back(compute_node);
    new_nodes->push_back(other_node);
    new_nodes->push_back(new_gather_node);
    new_nodes->push_back(new_identity_node);
    new_nodes->push_back(gather_input_node);
    new_nodes->push_back(gather_ind_node);
    new_nodes->push_back(gather_axis_node);
    index++;
    return Status::OK();
  }
  NodeDef new_compute_node;
  new_compute_node.CopyFrom(compute_node);
  *(new_compute_node.mutable_input(input_port)) = gather_node.input(0);
  new_nodes->push_back(new_compute_node);
  new_nodes->push_back(other_node);
  new_nodes->push_back(gather_node);
  new_nodes->push_back(gather_input_node);
  new_nodes->push_back(gather_ind_node);
  new_nodes->push_back(gather_axis_node);
  counter++;

  return Status::OK();
}

bool OptimizeGatherPattern(GraphDef &input_graph_def, GraphDef* output_graph_def,
                           bool& is_changed, int& counter, bool radical) {
  VLOG(1) << "start to optimize gather pattern, " << gather_pattern1.DebugString();
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      gather_pattern1,
      [&is_changed, &counter](const NodeMatch& match, const std::set<string>& input_nodes,
                    const std::set<string>& output_nodes,
                    std::vector<NodeDef>* new_nodes) {
        const NodeDef& compute_node = match.node;
        const NodeDef& gather_node = match.inputs[0].node;
        const NodeDef& other_node = match.inputs[1].node;
        const NodeDef& gather_input_node = match.inputs[0].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[0].inputs[1].node;
        const NodeDef& gather_axis_node = match.inputs[0].inputs[2].node;
        VLOG(1) << match.DebugString();
        is_changed = true;
        
        return OptimizePatternFunction(compute_node, gather_node, other_node,
                                       gather_input_node, gather_ind_node,
                                       gather_axis_node, new_nodes, 0, counter);
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize gather failed " << status;
    return false;
  }
  input_graph_def = *output_graph_def;
  VLOG(1) << "start to optimize gather pattern, " << gather_pattern2.DebugString();
  status = ReplaceMatchingOpTypes(
      input_graph_def,
      gather_pattern2,
      [&is_changed, &counter](const NodeMatch& match, const std::set<string>& input_nodes,
                    const std::set<string>& output_nodes,
                    std::vector<NodeDef>* new_nodes) {
        const NodeDef& compute_node = match.node;
        const NodeDef& other_node = match.inputs[0].node;
        const NodeDef& gather_node = match.inputs[1].node;
        const NodeDef& gather_input_node = match.inputs[1].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[1].inputs[1].node;
        const NodeDef& gather_axis_node = match.inputs[1].inputs[2].node;
        VLOG(1) << match.DebugString();
        is_changed = true;
        return OptimizePatternFunction(compute_node, gather_node, other_node,
                                       gather_input_node, gather_ind_node,
                                       gather_axis_node, new_nodes, 1, counter);
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize gather failed " << status;
    return false;
  }
  if (!radical) return true;
  input_graph_def = *output_graph_def;
  // gather->MatMul pattern没有利用broadcast语义，直接去掉gather会导致输出形状变化
  // 可以通过环境变量关闭优化来避免问题，默认开启
  VLOG(1) << "start to optimize gather pattern, " << gather_pattern3.DebugString();
  status = ReplaceMatchingOpTypes(
      input_graph_def,
      gather_pattern3,
      [&is_changed, &counter](const NodeMatch& match, const std::set<string>& input_nodes,
                    const std::set<string>& output_nodes,
                    std::vector<NodeDef>* new_nodes) {
        const NodeDef& compute_node = match.node;
        const NodeDef& gather_node = match.inputs[0].node;
        const NodeDef& other_node = match.inputs[1].node;
        const NodeDef& gather_input_node = match.inputs[0].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[0].inputs[1].node;
        const NodeDef& gather_axis_node = match.inputs[0].inputs[2].node;
        VLOG(1) << match.DebugString();
        is_changed = true;
        return OptimizePatternFunction(compute_node, gather_node, other_node,
                                       gather_input_node, gather_ind_node,
                                       gather_axis_node, new_nodes, 0, counter);
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize gather failed " << status;
    return false;
  }
  return true;
}

bool OptimizeMemoryAccess(GraphDef& input_graph, GraphDef* optimized_graph, bool radical) {
  int counter = 0;
  while(1) {
    bool graph_changed = false;
    bool result = OptimizeGatherPattern(input_graph, optimized_graph, graph_changed, counter, radical);
    if (!result) return false;
    if (!graph_changed) break;
    std::swap(input_graph, *optimized_graph);
  }
  VLOG(0) << "remove " << counter << " useless Gather node";
  return true;
}

}  // end namespace

Status MemoryAccessOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool optimize = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &optimize);
  if (!optimize) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  VLOG(0) << "MemoryAccessOptimizer is on.";
  bool radical = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE_RADICAL", true, &radical);

  GraphDef input_graph_def = item.graph;
  if (!OptimizeMemoryAccess(input_graph_def, optimized_graph, radical)) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void MemoryAccessOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
