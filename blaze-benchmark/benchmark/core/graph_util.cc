#include "benchmark/core/graph_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

#include "benchmark/core/node_util.h"

#if USE_CUDA
#include <nvml.h>
#endif  // USE_CUDA

using namespace tensorflow;

namespace benchmark {

const char* const kGPUDevice = "/device:GPU:";
const char* const kCPUDevice = "/device:CPU:";
const char* const kBlazeAttrConf = "blaze_option_path";
const char* const kBlazeAttrGraph = "graph_def";
const char* const kBlazeAttrInputNames = "input_names";
const char* const kBlazeAttrOutputNames = "output_names";
const char* const kBlazeKernelName = "BlazeXlaOp";
const char* const kBlazeRealDevice = "_blaze_real_device";
const char* const kDeepCopySuffix = "_deep_copy";
const char* const kBlazeCounter = "_blaze_counter";

template <class T>
inline void SetNodeAttr(const std::string& key, const T& value, NodeDef* node) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  auto* attr_map = node->mutable_attr();
  (*attr_map)[key] = attr_value;
}


Status SetGraphDevice(GraphDef& graph_def, int cpuid, int gpuid) {
  if (cpuid < 0) {
    return errors::InvalidArgument("Invalid cpuid: ", cpuid);
  }
  for (int i = 0; i < graph_def.node_size() ;i++) {
    NodeDef* node = graph_def.mutable_node(i);
    std::string device_name;
    std::string cpu_device = kCPUDevice + std::to_string(cpuid);
    std::string gpu_device = kGPUDevice + std::to_string(gpuid);
    if (gpuid < 0) {
      // if gpu is not present, place all nodes to cpu
      device_name = cpu_device;
    } else if (node->device().empty()) {
      // place nodes to gpu by default
      device_name = gpu_device;
    } else {
      // change cpu/gpu id based on device numbers
      DeviceNameUtils::ParsedName device;
      if (!DeviceNameUtils::ParseFullName(node->device(), &device)) {
        return errors::InvalidArgument("invalid device ", node->name(), " ", node->device());
      }
      if (device.type == "GPU") {
        device.id = gpuid;
      } else if (device.type == "CPU") {
        device.id = cpuid;
      }
      device_name = DeviceNameUtils::ParsedNameToString(device);
    }
    node->set_device(device_name);
    if (node->op() == kBlazeKernelName) {
      std::string temp = gpu_device;
      if (gpuid == -1) temp = cpu_device;
      *(((*(node->mutable_attr()))[kBlazeRealDevice]).mutable_s()) = temp;
    }
  }
  return Status::OK();
}

Status SetBlazeOpAttributes(const std::string& folder_path, const ConfigProto& config, GraphDef* graph_def, int gpu_count) {
  for (int i = 0; i < graph_def->node_size(); i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->op() == kBlazeKernelName) {
      auto attr = node->mutable_attr()->find(kBlazeAttrGraph);
      if (attr == node->mutable_attr()->end()) {
        return errors::Internal("Blaze node ", node->DebugString(),
                                " do not have attr ", kBlazeAttrGraph);
      }
      std::string graph_path = node->attr().at(kBlazeAttrGraph).s();
      SetNodeAttr(kBlazeAttrGraph, graph_path, node);
      attr = node->mutable_attr()->find(kBlazeAttrConf);
      if (attr == node->mutable_attr()->end()) {
        return errors::Internal("Blaze node ", node->DebugString(),
                                " do not have attr ", kBlazeAttrConf);
      }
      //auto options = config.blaze_options();
      //SetNodeAttr(kBlazeAttrConf, options.DebugString(), node);
      int count = gpu_count > 0 ? gpu_count : 0;
      SetNodeAttr(kBlazeCounter, gpu_count, node);
    }
  }
  return Status::OK();
}

void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (selector(node)) {
      *output_graph_def->mutable_node()->Add() = node;
    }
  }
}

void RedirectEdgesInGraphDef(const std::set<std::string>& input_nodes,
                             std::vector<std::vector<std::pair<std::string, Tensor>>>* inputs,
                             GraphDef* graph_def) {
  for (auto& temp : *inputs) {
    for (auto& iter : temp) {
      iter.first = NewNodeNameFromInput(iter.first);
    }
  }
  for (NodeDef& node : *(graph_def->mutable_node())) {
    for (std::string& input_name : *(node.mutable_input())) {
      std::string input_node = NodeNameFromInput(input_name);
      std::string input_copy = NewNodeNameFromInput(input_name) + kDeepCopySuffix;
      if (input_nodes.count(input_node) && node.name() != input_copy) {
        input_name = input_copy;
      }
    }
  }
}

Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const std::vector<std::string>& output_names,
                        std::vector<std::vector<std::pair<std::string, Tensor>>>* inputs,
                        GraphDef* output_graph_def) {
  std::set<std::string> required_nodes;
  std::set<std::string> input_nodes;
  std::unordered_map<std::string, std::vector<std::pair<std::string, DataType>>> node_tensor_map;
  for (auto& iter : (*inputs)[0]) {
    std::string tensor_name = iter.first;
    std::string node_name = NodeNameFromInput(iter.first);
    required_nodes.insert(node_name);
    input_nodes.insert(node_name);
    if (node_tensor_map.find(node_name) == node_tensor_map.end()) {
      std::vector<std::pair<std::string, DataType>> temp;
      node_tensor_map[node_name] = temp;
    }
    node_tensor_map[node_name].emplace_back(std::make_pair(tensor_name, iter.second.dtype()));
  }
  for (const std::string& output : output_names) {
    required_nodes.insert(output);
  }

  std::map<std::string, const NodeDef*> node_lookup;
  MapNamesToNodes(input_graph_def, &node_lookup);

  std::vector<std::string> current_inputs;
  for (const std::string& output_name : output_names) {
    current_inputs.push_back(NodeNameFromInput(output_name));
  }

  while (!current_inputs.empty()) {
    std::set<std::string> next_inputs;
    for (const std::string& current_input : current_inputs) {
      required_nodes.insert(current_input);
      if (input_nodes.count(current_input)) {
        continue;
      }
      if (!node_lookup.count(current_input)) {
        return errors::InvalidArgument("Input node ", current_input,
                                       " not found in graph");
      }
      const NodeDef* current_node = node_lookup[current_input];
      for (const std::string& input_name : current_node->input()) {
        std::string input_node_name = NodeNameFromInput(input_name);
        if (!required_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs =
        std::vector<std::string>(next_inputs.begin(), next_inputs.end());
  }

  GraphDef filtered_graph_def;

  FilterGraphDef(input_graph_def,
                 [&](const NodeDef& node) {
                   return required_nodes.count(node.name()) > 0;
                 },
                 &filtered_graph_def);

  output_graph_def->Clear();
  for (const NodeDef& node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      for (auto iter : node_tensor_map[node.name()]) {
        NodeDef placeholder_node;
        placeholder_node.set_op("Placeholder");
        placeholder_node.set_name(NewNodeNameFromInput(iter.first));
        SetNodeAttr("dtype", iter.second, &placeholder_node);
        *(output_graph_def->mutable_node()->Add()) = placeholder_node;

        // Optimize GPU memcpy: add DeepCopy after input to allocate pinned memory and do async h2d memcpy
        NodeDef copy_node;
        copy_node.set_op("DeepCopy");
        copy_node.set_name(NewNodeNameFromInput(iter.first) + kDeepCopySuffix);
        SetNodeAttr("T", iter.second, &copy_node);
        copy_node.add_input(placeholder_node.name());
        copy_node.set_device("/device:CPU:0");
        *(output_graph_def->mutable_node()->Add()) = copy_node;
      }
    } else {
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  RedirectEdgesInGraphDef(input_nodes, inputs, output_graph_def);

  for (const NodeDef& node : output_graph_def->node()) {
    if (kDenseInputs.count(node.op()) != 0 && node.op() != "Placeholder") {
      return errors::Internal(node.op(),
          " is not stripped (may because runmeta is invalid and does not include the op's output tensors).");
    }
  }

  return Status::OK();
}

Status FlatBlazeOp(GraphDef* graph_def, std::unordered_map<std::string, std::string>* output_map) {
  output_map->clear();
  GraphDef temp_graph(std::move(*graph_def));
  graph_def->Clear();

  for (int i = 0; i < temp_graph.node_size(); i++) {
    NodeDef* temp = temp_graph.mutable_node(i);
    // copy non BlazeXlaOp into dst graph
    if (temp->op() != kBlazeKernelName) {
      *(graph_def->mutable_node()->Add()) = *temp;
      continue;
    }

    // get links between in and out graphs
    NodeDef* blaze = temp;
    VLOG(1) << "FlatBlazeOp: " << blaze->name();
    std::vector<std::string> blaze_graph_inputs;
    std::vector<std::string> blaze_graph_outputs;
    Status s;
    s = GetNodeAttr(*blaze, kBlazeAttrInputNames, &blaze_graph_inputs);
    if (!s.ok()) {
      return errors::Internal("FlatBlazeOp: parse attr ", kBlazeAttrInputNames, " failed, ", s.ToString());
    }
    s = GetNodeAttr(*blaze, kBlazeAttrOutputNames, &blaze_graph_outputs);
    if (!s.ok()) {
      return errors::Internal("FlatBlazeOp: parse attr ", kBlazeAttrOutputNames, " failed, ", s.ToString());
    }
    if (blaze->input_size() != blaze_graph_inputs.size()) {
      return errors::Internal("FlatBlazeOp: input numbers mismatch, ",
                              blaze->input_size(), " vs ", blaze_graph_inputs.size());
    }

    std::unordered_map<std::string, std::string> input_map;
    int idx = 0;
    for (const std::string& blaze_op_input : *(blaze->mutable_input())) {
      input_map[blaze_graph_inputs[idx++]] = blaze_op_input;
    }
    idx = 0;
    for (const std::string& blaze_graph_output : blaze_graph_outputs) {
      std::string blaze_op_output = blaze->name();
      if (idx == 0) {
        (*output_map)[blaze_op_output] = blaze->name() + "/" + blaze_graph_output;
      }
      blaze_op_output += ":" + std::to_string(idx);
      (*output_map)[blaze_op_output] = blaze->name() + "/" + blaze_graph_output;
      idx++;
    }

    // new inner graph based on blazexlaop attr
    std::string path = blaze->attr().at(kBlazeAttrGraph).s();
    GraphDef blaze_graph;

    if (!ReadBinaryProto(Env::Default(), path, &blaze_graph).ok()) {
      if (!ReadTextProto(Env::Default(), path, &blaze_graph).ok()) {
        return errors::Internal("FlatBlazeOp: parse proto from ", path,  " failed");
      }
    }
    graph_def->mutable_library()->MergeFrom(blaze_graph.library());

    // flat inner graph into outer graph
    for (int j = 0; j < blaze_graph.node_size(); j++) {
      NodeDef* node = blaze_graph.mutable_node(j);
      std::string name = node->name();
      // add namescope to avoid duplicate names
      std::string new_name = blaze->name() + "/" + name;
      node->set_name(new_name);
      if (node->op() == "Placeholder") {
        // link outer graph to inputs of inner graph
        NodeDef identity;
        identity.set_op("Identity");
        identity.set_name(new_name);
        SetNodeAttr("T", node->attr().at("dtype"), &identity);
        if (input_map.find(name) == input_map.end()) {
          return errors::Internal("FlatBlazeOp failed: Placeholder ", name,
                                  " is not in BlazeXlaOp input_names attr.");
        }
        identity.add_input(input_map[name]);
        *(graph_def->mutable_node()->Add()) = identity;
      } else {
        // copy non-input nodes
        for (std::string& input : *(node->mutable_input())) {
          // add namescope to avoid duplicate names
          input = blaze->name() + "/" + input;
        }
        *(graph_def->mutable_node()->Add()) = *node;
      }
    }
  }

  // link outputs of inner graph to outer graph
  for (int i = 0; i < graph_def->node_size(); i++) {
    NodeDef* temp = graph_def->mutable_node(i);
    for (std::string& input : *(temp->mutable_input())) {
      if (output_map->find(input) != output_map->end()) {
        input = (*output_map)[input];
      }
    }
  }

  graph_def->mutable_library()->MergeFrom(temp_graph.library());
  return Status::OK();
}

}  // namespace benchmark
