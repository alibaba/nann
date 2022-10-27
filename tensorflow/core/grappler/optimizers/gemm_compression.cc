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

#include "tensorflow/core/grappler/optimizers/gemm_compression.h"
#include "tensorflow/core/grappler/optimizers/original_delivery_common.h"

#include <fstream>
#include <queue>
#include <map>
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

bool OptimizeGatherConcatPattern(GraphDef &input_graph_def, GraphDef* output_graph_def,
                                 bool& is_changed) {
  VLOG(1) << "start to optimize gather pattern, " << gemm_compression_pattern.DebugString();
  Status status = ReplaceMatchingOpTypes(
      input_graph_def,
      gemm_compression_pattern,
      [&is_changed](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // 1. 匹配到pattern
        // 2. 获取有用的节点
        const NodeDef& matmul_node = match.node;
        const NodeDef& concat_node = match.inputs[0].node;
        const NodeDef& weight_node = match.inputs[1].node;
        const NodeDef& gather_node = match.inputs[0].inputs[0].node;
        const NodeDef& gather_input_node = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_ph_node = match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& gather_ind_node = match.inputs[0].inputs[0].inputs[1].node;
        const NodeDef& gather_axis_node = match.inputs[0].inputs[0].inputs[2].node;
        VLOG(1) << match.DebugString();
        
        Status status;

        // 3. 检查placeholder和gather axis const 值
        bool invalid = false;
        if (gather_ind_node.name().find("user_creative_indicator") == string::npos) {
          LOG(WARNING) << "gather input indicator placeholder not match:" << gather_ind_node.name();
          invalid = true;
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
        if (concat_node.op() != "ConcatV2") {
          LOG(WARNING) << "concat node is not ConcatV2:" << concat_node.DebugString();
          invalid = true;
        }
        // 4. 将concat进行分拆，后续MatMul的权重也要分拆
        DataType output_type = DT_FLOAT;
        string type_key = "T";
        if (weight_node.op() == "Const") type_key = "dtype";
        if (weight_node.attr().count(type_key) != 0) {
          output_type = weight_node.attr().at(type_key).type();
        } else {
          invalid = true;
        }
        if (invalid) {
          // 不替换时，直接返回，会导致之后每次都匹配到这个不满足条件的pattern，
          // 其余pattern无法继续匹配，因此给这部分子图增加一个Identity，改变图结构
          NodeDef new_identity_node;
          new_identity_node.set_name(concat_node.name() + "/identity");
          new_identity_node.set_op("Identity");
          new_identity_node.clear_attr();
          (*new_identity_node.mutable_attr())["T"].set_type(output_type);
          *(new_identity_node.mutable_input()->Add()) = gather_node.name();

          NodeDef new_concat_node;
          new_concat_node.CopyFrom(concat_node);
          *(new_concat_node.mutable_input(0)) = new_identity_node.name();
          VLOG(1) << "insert Identity before Gather" << new_identity_node.DebugString();
          // 5. 保留匹配的节点
          new_nodes->push_back(matmul_node);
          new_nodes->push_back(weight_node);
          new_nodes->push_back(new_concat_node);
          new_nodes->push_back(new_identity_node);
          new_nodes->push_back(gather_node);
          new_nodes->push_back(gather_ph_node);
          new_nodes->push_back(gather_input_node);
          new_nodes->push_back(gather_ind_node);
          new_nodes->push_back(gather_axis_node);

          is_changed = true;
          return Status::OK();
        }
        // 获取输入的shape
        int size = gather_ph_node.attr().at("shape").shape().dim(1).size();
        VLOG(1) << "get gaterh input dim 1 size:" << size;
        
        // 构建split，拆分权重
        Tensor t_begin(DT_INT64, TensorShape({2}));
        auto begin_data = t_begin.tensor<int64, 1>();
        begin_data(0) = 0;
        begin_data(1) = 0;
        Tensor t_size(DT_INT64, TensorShape({2}));
        auto size_data = t_size.tensor<int64, 1>();
        size_data(0) = size;
        size_data(1) = -1;
        string slice_name_part1 = weight_node.name() + "_part1";
        NodeDef begin_const_part1;
        NodeDef size_const_part1;
        NodeDef slice_part1;
        // 构建Slice
        TF_RETURN_IF_ERROR(CreateConstNodeDef(begin_const_part1,
                             slice_name_part1 + "/slice_begin", t_begin, weight_node));
        TF_RETURN_IF_ERROR(CreateConstNodeDef(size_const_part1,
                             slice_name_part1 + "/slice_size", t_size, weight_node));
        TF_RETURN_IF_ERROR(ConstructSliceNodeDef(slice_part1, weight_node,
                             begin_const_part1, size_const_part1,
                             slice_name_part1 + "/slice", output_type));
        begin_data(0) = size;
        size_data(0) = -1;
        string slice_name_part2 = weight_node.name() + "_part2";
        NodeDef begin_const_part2;
        NodeDef size_const_part2;
        NodeDef slice_part2;
        // 构建Slice
        TF_RETURN_IF_ERROR(CreateConstNodeDef(begin_const_part2,
                             slice_name_part2 + "/slice_begin", t_begin, weight_node));
        TF_RETURN_IF_ERROR(CreateConstNodeDef(size_const_part2,
                             slice_name_part2 + "/slice_size", t_size, weight_node));
        TF_RETURN_IF_ERROR(ConstructSliceNodeDef(slice_part2, weight_node,
                             begin_const_part2, size_const_part2,
                             slice_name_part2 + "/slice", output_type));

        // 构建新的ConcatV2
        NodeDef new_concat;
        std::vector<NodeDefBuilder::NodeOut> concat_inputs;
        DataType t_idx = concat_node.attr().at("Tidx").type();
        DataType t_input = concat_node.attr().at("T").type();
        string idx_name;
        int idx = 0;
        for (string input : concat_node.input()) {
          // skip first input
          if (idx == 0) {
            idx++;
            continue;
          }
          if (idx == (concat_node.input().size() - 1)) {
            idx_name = input;
          } else {
            concat_inputs.emplace_back(input, 0, t_input);
          }
          idx++;
        }
        int concat_n = concat_node.attr().at("N").i();
        NodeDefBuilder::NodeOut concat_idx(idx_name, 0, t_idx);
        TF_RETURN_IF_ERROR(ConstuctConcatNodeDef(new_concat, matmul_node,
                                     matmul_node.name() + "/concat",
                                     concat_inputs, concat_idx, concat_n - 1,
                                     t_input, t_idx));
        // 构建新的MatMul
        NodeDef matmul_part1;
        matmul_part1.CopyFrom(matmul_node);
        *(matmul_part1.mutable_input(0)) = gather_input_node.name();
        *(matmul_part1.mutable_input(1)) = slice_part1.name();
        matmul_part1.set_name(matmul_node.name() + "_part1");
        NodeDef matmul_part2;
        matmul_part2.CopyFrom(matmul_node);
        *(matmul_part2.mutable_input(0)) = new_concat.name();
        *(matmul_part2.mutable_input(1)) = slice_part2.name();
        matmul_part2.set_name(matmul_node.name() + "_part2");
        // 结果做加法
        std::vector<NodeDefBuilder::NodeOut> add_inputs;
        add_inputs.emplace_back(matmul_part1.name(), 0, output_type);
        add_inputs.emplace_back(matmul_part2.name(), 0, output_type);
        NodeDef add_node;
        TF_RETURN_IF_ERROR(ConstuctAddNodeDef(add_node, matmul_node, matmul_node.name(),
                                              add_inputs, output_type));

        // 5. 保留匹配的节点
        new_nodes->push_back(add_node);
        new_nodes->push_back(matmul_part1);
        new_nodes->push_back(matmul_part2);
        new_nodes->push_back(new_concat);
        // 为了使concat的其他输入能再次匹配到这个pattern，保留原gather和concat节点
        // 最终会在图中留下无后续依赖的concat节点
        new_nodes->push_back(concat_node);
        new_nodes->push_back(gather_node);
        new_nodes->push_back(slice_part1);
        new_nodes->push_back(slice_part2);
        new_nodes->push_back(begin_const_part1);
        new_nodes->push_back(size_const_part1);
        new_nodes->push_back(begin_const_part2);
        new_nodes->push_back(size_const_part2);
        new_nodes->push_back(weight_node);
        new_nodes->push_back(gather_ph_node);
        new_nodes->push_back(gather_input_node);
        new_nodes->push_back(gather_ind_node);
        new_nodes->push_back(gather_axis_node);

        is_changed = true;
        return Status::OK();
      },
      {}, output_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "optimize gather concat failed " << status;
    return false;
  }
  return true;
}

bool OptimizeGemmCompression(GraphDef& input_graph, GraphDef* optimized_graph) {
  
  int count = 0;
  while(1) {
    bool graph_changed = false;
    bool result = OptimizeGatherConcatPattern(input_graph, optimized_graph, graph_changed);
    if (!result) return false;
    if (!graph_changed) break;
    count++;
    std::swap(input_graph, *optimized_graph);
  }
  VLOG(0) << "Compress " << count << " Gather->Concat gemm structure";
  return true;
}

}  // end namespace

Status GemmCompressionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool optimize = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &optimize);
  if (!optimize) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  VLOG(0) << "GemmCompressionOptimizer is on.";

  GraphDef input_graph_def = item.graph;
  if(!OptimizeGemmCompression(input_graph_def, optimized_graph)) {
    LOG(INFO) << "optimize gemm compression failed";
    *optimized_graph = item.graph;
    return Status::OK();
  }
  *optimized_graph->mutable_versions() = item.graph.versions();
  return Status::OK();
}

void GemmCompressionOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow
