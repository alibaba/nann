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

#include "tensorflow/core/grappler/optimizers/multi_dnn_switch_optimizer.h"
#include "tensorflow/core/grappler/optimizers/original_delivery_common.h"

#include <fstream>
#include <queue>
#include <map>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace grappler {

namespace {

struct MultiDNNInfo {
  std::set<Node*> dynamic_partition_a; // 可能有多个DynamicPartion交叉计算
  Node* dynamic_partition_b;
  Node* dynamic_stitch;
  Node* partition;
  MultiDNNInfo() {
    dynamic_partition_b = nullptr;
    dynamic_stitch = nullptr;
    partition = nullptr;
  }
};

bool OpInputOrderInsensitive(string op) {
  static std::unordered_set<string> op_set = {
      "Add", "Mul",
      "Maximum", "Minimum",
      "Merge"};
  if (op_set.find(op) != op_set.end()) {
    return true;
  }
  return false;
}

bool SkipVisitInputOps(string op) {
   static std::unordered_set<string> op_set = {
       "NoOp",
       "Const"};
  if (op_set.find(op) != op_set.end()) {
    return true;
  }
  return false;
}

template <class Tkey, class Tvalue>
bool EqualProtoMap(const ::tensorflow::protobuf::Map<Tkey, Tvalue>& a,
                   const ::tensorflow::protobuf::Map<Tkey, Tvalue>& b,
                   const std::function<string(const Tkey&)>& key_to_string,
                   const std::function<string(const Tvalue&)>& value_to_string,
                   const std::function<bool(const Tkey&, const Tvalue&,
                                            const Tvalue&)>& compare,
                   const string& map_name, string* diff) {
  for (const auto& elt_a : a) {
    const auto iter = b.find(elt_a.first);
    if (iter == b.end()) {
      if (diff) {
        *diff = absl::StrCat(map_name, " expected: contains element with key '",
                             key_to_string(elt_a.first),
                             "' got: map has no such element");
      }
      return false;
    }
    if (!compare(elt_a.first, elt_a.second, iter->second)) {
      if (diff) {
        *diff = absl::StrCat(map_name, " expected: element with key '",
                             key_to_string(elt_a.first), "' has value '",
                             value_to_string(elt_a.second), "' got: '",
                             value_to_string(iter->second), "'");
      }
      return false;
    }
  }
  for (const auto& elt_b : b) {
    const auto iter = a.find(elt_b.first);
    if (iter == a.end()) {
      if (diff) {
        *diff = absl::StrCat(map_name, " got: contains element with key '",
                             key_to_string(elt_b.first),
                             "' expected: map has no such element");
      }
      return false;
    }
  }
  return true;
}

bool EqualNodeDef(const NodeDef& a, const NodeDef& b, string* diff) {
  if (a.op() != b.op()) {
    if (diff) {
      *diff = absl::StrCat(" mismatch for node ", a.name(),
                           ", expected op '", a.op(), "' got '", b.op());
    }
    return false;
  }
  if (a.device() != b.device()) {
    if (diff) {
      *diff = absl::StrCat( " mismatch for node ", a.name(),
                           ", expected device '", a.device(), "' got '",
                           b.device());
    }
    return false;
  }
  if (a.input_size() != b.input_size()) {
    if (diff) {
      *diff = absl::StrCat( " mismatch for node ", a.name(),
                           ", expected ", a.input_size(), " inputs got ",
                           b.input_size(), " expected:\n", a.DebugString(),
                           "\ngot:\n", b.DebugString());
    }
    return false;
  }
  for (int i = 0; i < a.input_size(); ++i) {
    if (absl::StartsWith(a.input(i), "^")) {
      if (!absl::StartsWith(b.input(i), "^")) {
        if (diff) {
          *diff = absl::StrCat( " mismatch for node ", a.name(),
                               " input ", i, ", expected control input ",
                               a.input(i), " got ", b.input(i), " expected:\n",
                               a.DebugString(), "\ngot:\n", b.DebugString());
        }
        return false;
      }
    }
  }
  return EqualProtoMap<string, AttrValue>(
      a.attr(), b.attr(), [](const string& s) { return s; },
      [](const AttrValue& v) { return v.DebugString(); },
      [](const string& key, const AttrValue& av, const AttrValue& bv) {
        if (key == "ancestors") {
          // The ancestors are added from a set so the order is unpredictable;
          // just compare set equality not list equality.
          std::unordered_set<string> a_set(av.list().s().begin(),
                                           av.list().s().end());
          std::unordered_set<string> b_set(bv.list().s().begin(),
                                           bv.list().s().end());
          return a_set == b_set;
          } else if (key == "_class") {
            return true;
          } else {
          return av.DebugString() == bv.DebugString();
        }
      },
      absl::StrCat(" attr mismatch for node ", a.name()), diff);
}

bool CheckNodeMatch(const Node* n1, const Node* n2, string* diff,
                    std::set<tensorflow::DataType> check_const_type,
                    bool force_check_value = false) {
  if (n1->type_string() != n2->type_string()) {
    if (diff) {
      *diff = absl::StrCat("input op type not match: ", n1->type_string(),
                           " VS ", n2->type_string());
    }
    return false;
  }
  if (n1->type_string() == "Const") {
    auto type1 = n1->def().attr().at("dtype");
    auto type2 = n2->def().attr().at("dtype");
    if (type1.DebugString() != type2.DebugString()) {
      if (diff) {
        *diff = absl::StrCat("const op data type not match: ",
                             n1->DebugString(), " VS ", n2->DebugString());
      }
      return false;
    }
    // 指定类型const要保证值一致
    if (check_const_type.find(type1.type()) != check_const_type.end()) {
      if(!EqualNodeDef(n1->def(), n2->def(), diff)) {
        return false;
      }
      return true;
    }
    auto& shape1 = n1->def().attr().at("value").tensor().tensor_shape();
    auto& shape2 = n2->def().attr().at("value").tensor().tensor_shape();
    if (shape1.DebugString() != shape2.DebugString()) {
      if (diff) {
        *diff = absl::StrCat("input const op shape not match: ",
                  shape1.DebugString(), " VS ", shape2.DebugString());
       }
       return false;
    }
  } else {
    if(!EqualNodeDef(n1->def(), n2->def(), diff)) {
      return false;
    }
  }
  return true;
}

int GetNodeConstInputCount(const Node* n) {
  int const_num = 0;
  for (auto e : n->in_edges()) {
    if (e->src()->type_string() == "Const") {
      const_num++;
    }
  }
  return const_num;
}

class SubGraphCollection {
 public:
  SubGraphCollection(Graph* graph, std::vector<Node*> switch_n, Node* merge, int branch_num) :
    graph_(graph), switch_n_(switch_n), merge_(merge), branch_num_(branch_num) {
      index_ = nullptr;
  }

  struct BranchNodesCollection {
    bool const_input_has_controlflow;
    std::map<int, Node*> branch_nodes;
    std::vector<std::map<int, const Edge*>> inputs;
    int reserve_branch;
    Node* reserve_node;

    BranchNodesCollection(std::map<int, Node*>& nodes) {
      const_input_has_controlflow = false;
      branch_nodes = nodes;
      reserve_branch = branch_nodes.begin()->first;
      reserve_node = branch_nodes.begin()->second;
    }

    void SetBranchNodesConstInput() {
      // 将节点的const输入信息统计到inputs中
      int const_num = GetNodeConstInputCount(reserve_node);
      inputs.resize(const_num);
      for (auto iter : branch_nodes) {
        int input_index = 0;
        for (auto e : iter.second->in_edges()) {
          if (e->src()->type_string() == "Const") {
            if (e->IsControlEdge()) const_input_has_controlflow = true;
            inputs[input_index][iter.first] = e;
            input_index++;
          }
        }
      }
    }

    bool HasConstInput() {
      return !inputs.empty();
    }

    bool IsSwitchN() {
      return reserve_node->type_string() == "_SwitchN";
    }

    int GetReserveBranch() {
      return reserve_branch;
    }

    Node* GetReserveNode() {
      return reserve_node;
    }

    string DebugString() {
      string s = absl::StrCat("branch nodes collection has controlflow weight: ",
                   const_input_has_controlflow, ", nodes: \n");
      for (auto iter : branch_nodes) {
        s = absl::StrCat(s, "branch index ", iter.first, ", ", iter.second->name(),
                         ", ", iter.second->type_string(), "\n");
      }
      s = absl::StrCat(s, "const input:\n");
      for (auto& input : inputs) {
        for (auto  iter: input) {
          s = absl::StrCat(s, "branch index ", iter.first, ": ",
                           iter.second->src()->name(), ":",
                           iter.second->dst_input(), "\n");
        }
      }
      return s;
    }
  };

  int GetBranchNum() {return branch_num_;}

  bool InputsAreSameConst(std::map<int, const Edge*>& inputs) {
    auto begin = inputs.begin();
    if (begin->second->src()->type_string() != "Const") return false;
    // 1.如果不是同一个const节点，则继续判断const值是否相同
    bool same_node = true;
    auto it = begin;
    for (it++; it != inputs.end(); ++it) {
      if (begin->second->src()->name() != it->second->src()->name()) same_node = false;
    }
    if (same_node) return true;
    std::set<tensorflow::DataType> check_value_set = {DT_BOOL, DT_INT32, DT_INT64};
    auto type = begin->second->src()->def().attr().at("dtype");
    if (check_value_set.find(type.type()) == check_value_set.end()) return false;
    // 2.如果const值都一致，也认为是相同const输入
    it = begin;
    for (it++; it != inputs.end(); ++it) {
      string diff;
      if (!CheckNodeMatch(begin->second->src(), it->second->src(), &diff,
                          {DT_BOOL, DT_INT32, DT_INT64})) {
        return false;
      }
    }
    return true;
  }

  bool TryUpdateSwitchNInput(Node* node) {
    for (auto e : node->in_edges()) {
      for (Node* n : switch_n_) {
        if (e->src() == n) {
          const Edge* input;
          n->input_edge(0, &input);
          Status status = graph_->UpdateEdge(input->src(), input->src_output(),
                                      e->dst(), e->dst_input());
          TF_RETURN_FALSE_IF_ERROR(status, "update edge")
        }
      }
    }
    return true;
  }
  bool AnyQueueEmpty(std::map<int, std::queue<Node*>>& branch_unvisited_queue) {
    for (auto iter:branch_unvisited_queue) {
      if (iter.second.empty()) {
        return true;
      }
    }
    return false;
  }

  bool CheckBranchNodesMatch(std::map<int, Node*>& nodes) {
    auto begin = nodes.begin();
    auto it = begin++;
    for (; it != nodes.end(); ++it) {
      string diff;
      if (!CheckNodeMatch(begin->second, it->second, &diff, {})) {
        LOG(ERROR) << "graph a node not match with graph b node: " << diff;
        return false;
      }
    }
    return true;
  }

  bool PushInputsWithSameOrder(std::vector<Node*>& base_order,
                               std::vector<Node*>& same_order, Node* node) {
    if ((node->in_edges().size() - GetNodeConstInputCount(node))
        != base_order.size()) {
      LOG(ERROR) << "branch nodes input not equal: " << base_order.size()
                 << " VS " << node->num_inputs() << ", " << node->DebugString();
      return false;
    }
    if (!OpInputOrderInsensitive(node->type_string())) {
      for (auto e : node->in_edges()) {
        if(SkipVisitInputOps(e->src()->type_string())) continue;
        same_order.push_back(e->src());
      }
    } else {
      // 防止输入中有match 的，但name不同，从而漏掉排在后边的input
      std::set<string> visited;
      for (Node* n : base_order) {
        bool match = false;
        for (auto e : node->in_edges()) {
          string diff;
          if (visited.find(e->src()->name()) != visited.end()) {
            continue;
          }
          if(SkipVisitInputOps(e->src()->type_string())) continue;
          if (CheckNodeMatch(n, e->src(), &diff, {})) {
            VLOG(1) << "node " << n->name() << " match with " << e->src()->name();
            same_order.push_back(e->src());
            visited.insert(e->src()->name());
            match = true;
            break;
          } else {
            VLOG(1) << n->name() << " not match with " << e->src()->name() << ", " << diff;
          }
        }
        if (!match) {
          LOG(ERROR) << "cant find any match input to base: " << n->name()
                     << " | " << n->type_string()
                     << ", " << node->def().DebugString();
          return false;
        }
      }
    }
    return true;
  }

  bool IsSkipBranch (int index) {
    return skip_branchs_.find(index) != skip_branchs_.end();
  }

  bool CollecteBranchNodes(const string& skip_branchs_str) {
    for (auto x : str_util::Split(skip_branchs_str, ",")) {
      int index = atoi(x.c_str());
      if (index >= branch_num_) {
        LOG(ERROR) << "set skip branch index beyound branch num:"
                   << index << " VS " << branch_num_;
      } else {
        skip_branchs_.insert(index);
      }
    }
    if (skip_branchs_.size() == branch_num_) {
      LOG(ERROR) << "skip all branch!? branch num:" << branch_num_
                 << "skip string: " << skip_branchs_str;
    }
    Status status;
    std::map<int, std::queue<Node*>> branch_unvisited_queue;
    if (merge_->num_inputs() != branch_num_) {
      LOG(ERROR) << "merge input count not equals to branch num:"
                 << merge_->num_inputs() << " VS " << branch_num_;
    }
    for (auto e : merge_->in_edges()) {
      if (IsSkipBranch(e->dst_input())) {
        VLOG(1) << "skip branch " << e->dst_input();
        continue;
      }
      branch_unvisited_queue[e->dst_input()].push(e->src());
      VLOG(1) << "merge input edge: " << e->dst_input() << ":" << e->src()->name();
    }
    std::set<string> visited;
    while(!AnyQueueEmpty(branch_unvisited_queue)) {
      std::map<int, Node*> nodes;
      std::vector<Node*> base_order;
      std::vector<Node*> same_order;
      for (auto& iter:branch_unvisited_queue) {
        Node* top = iter.second.front();
        iter.second.pop();
        if (visited.find(top->name()) != visited.end() &&
          top->type_string() != "_SwitchN") {
          continue;
        }
        visited.insert(top->name());
        nodes[iter.first] = top;
        // SwitchN为界，不再追溯前序节点
        if (top->type_string() == "_SwitchN") {
          bool valid = false;
          for (Node* n : switch_n_) {
            if (n == top) {
              valid = true;
              break;
            }
          }
          if (!valid) {
            LOG(ERROR) << "find strange SwitchN node! " << top->DebugString();
            return false;
          }
          continue;
        }
        same_order.clear();
        VLOG(1) << "visit node: " << top->name() << ", " << top->type_string()
                << ", has input num: " << top->num_inputs();
        if (base_order.empty()) {
          for (auto e : top->in_edges()) {
            if(SkipVisitInputOps(e->src()->type_string())) continue;
            VLOG(1) << iter.first << " push input: " << e->src()->name()
                    << ", " << e->src()->type_string();
            base_order.push_back(e->src());
            iter.second.push(e->src());
          }
        } else {
          if (!PushInputsWithSameOrder(base_order, same_order, top)) {
            return false;
          }
          for (Node* n : same_order) {
            VLOG(1) << iter.first << " push input: " << n->name()
                    << ", " << n->type_string();
            iter.second.push(n);
          }
        }
      }
      if (nodes.empty()) continue;
      if (nodes.size() != (branch_num_ - skip_branchs_.size())) {
        LOG(ERROR) << "visit subgraph layer nodes not equal to branch number:"
                   << nodes.size() << " VS " << branch_num_;
        for (auto iter : nodes) {
          VLOG(0) << "branch index " << iter.first << ": "
                  << iter.second->name() << ", " << iter.second->type_string();
        }
        return false;
      }
      std::unique_ptr<BranchNodesCollection> temp(new BranchNodesCollection(nodes));
      branch_collection_.push_back(std::move(temp));
      if (!CheckBranchNodesMatch(nodes)) {
        LOG(ERROR) << "not all nodes match: ";
        VLOG(0) << branch_collection_.back()->DebugString();
        return false;
      }
      branch_collection_.back()->SetBranchNodesConstInput();
      VLOG(1) << branch_collection_.back()->DebugString();
    }
    for (auto iter:branch_unvisited_queue) {
      if (!iter.second.empty()) {
        LOG(ERROR) << "unvisited queue not empty, maybe switch branch subgraph not equal";
        return false;
      }
    }
    VLOG(1) << "got SwitchN size: " << switch_n_.size();
    for (Node* n : switch_n_) {
      if (index_ == nullptr) {
        status = n->input_edge(1, &index_);
        TF_RETURN_FALSE_IF_ERROR(status, "get index of SwitchN")
      } else {
        const Edge* temp = nullptr;
        status = n->input_edge(1, &temp);
        TF_RETURN_FALSE_IF_ERROR(status, "get index of SwitchN")
        if(index_->src() != temp->src()) {
          LOG(ERROR) << "index of SwitchN diff: " << index_->src()->name()
                     << " VS " << temp->src()->name();
          return false;
        }
      }
    }
    return true;
  }

  bool ConvertToSwitchWeight() {
    std::unordered_set<string> converted;
    std::unordered_set<Node*> remove_set;
    Status status;
    string switch_name = switch_n_[0]->name() + "/merge_switch_subgraph/switch";
    NodeDefBuilder::NodeOut switch_input(index_->src()->name(),
                                         index_->src_output(),
                                         index_->src()->output_type(0));
    Node* switch_n = NodeConstructor(graph_, switch_name, switch_n_[0],
              [&](NodeDef& def) {
                return NodeDefBuilder(switch_name, "_SwitchN")
                      .Input(switch_input)
                      .Input(switch_input)
                      .Attr("num_outs", branch_num_)
                      .Attr("T", index_->src()->output_type(0))
                      .Finalize(&def);
              });
    TF_RETURN_FALSE_IF_NULL(switch_n, "construct SwitchN failed")
    VLOG(1) << switch_n->DebugString();
    std::map<int, Node*> no_ops;
    std::map<int, Node*> identity_ops;
    for (int i = 0; i < branch_num_; ++i) {
      if (IsSkipBranch(i)) continue;
      string no_name = switch_n_[0]->name() + "/merge_switch_subgraph/input_control_node_"
                       + std::to_string(i);
      string identity_name = switch_n_[0]->name() + "/merge_switch_subgraph/pivot_"
                       + std::to_string(i);
      Node* no = NodeConstructor(graph_, no_name, switch_n_[0], [&](NodeDef& def) {
        return NodeDefBuilder(no_name, "NoOp")
                              .Finalize(&def);
      });
      TF_RETURN_FALSE_IF_NULL(no, "construct NoOp failed")
      no_ops[i] = no;
      Node* identity = NodeConstructor(graph_, identity_name, switch_n_[0],
            [&](NodeDef& def) {
        return NodeDefBuilder(identity_name, "Identity")
                              .Input(switch_n->name(), i, switch_n->output_type(0))
                              .Finalize(&def);
      });
      TF_RETURN_FALSE_IF_NULL(identity, "construct Identity failed")
      identity_ops[i] = identity;
    }
    // index-|          |->Identity->NoOp
    // index-|->SwitchN-|->Identity->NoOp
    //                  |->Identity->NoOp
    // 由于_SwitchN输出类型(这里是int32)与分支数据类型不相同，因此无法直接连接
    // 控制边不能指定输出index，只能节点到节点，因此需要先增加n个Identity，接收_SwitchN多个输出
    // NoOp不能接受输入，也没有数据类型，因此可以当做控制边，作为桥梁解决数据类型不匹配的问题
    // 1.首先将SwitchN每个分支连接到一个Identity节点;
    // 2.然后Identity与NoOp连接控制边，NoOp再与后续节点连接控制边，达到选择分支的目的
    graph_->AddEdge(index_->src(), index_->src_output(), switch_n, 0);
    graph_->AddEdge(index_->src(), index_->src_output(), switch_n, 1);
    for (int i = 0; i < branch_num_; ++i) {
      if (IsSkipBranch(i)) continue;
      graph_->AddEdge(switch_n, i, identity_ops[i], 0);
      graph_->AddControlEdge(identity_ops[i], no_ops[i]);
    }

    for (auto& collection : branch_collection_) {
      if (converted.count(collection->GetReserveNode()->name()) != 0) {
        continue;
      }
      if (collection->IsSwitchN()) {
        continue;
      }
      // 如果是节点输入有SwitchN，则直接去掉，将其输入连接到SwitchN的第一个输出
      if (!TryUpdateSwitchNInput(collection->GetReserveNode())) {
        LOG(ERROR) << "try update switch input failed";
        return false;
      }
      // 如果节点有const输入，则构造结构,其中SwitchN、Identity和NoOp共享
      //        |->Identity->NoOp->const-|
      // SwitchN|->Identity->NoOp->const-|->Merge->Node
      //        |->Identity->NoOp->const-|
      if (collection->HasConstInput()) {
        int input_idx = 0;
        for (auto& input : collection->inputs) {
          if (InputsAreSameConst(input)) continue;
          string merge_name = collection->GetReserveNode()->name()
                              + "/merge_switch_subgraph/merge_"
                              + std::to_string(input_idx);
          input_idx++;
          std::vector<NodeDefBuilder::NodeOut> merge_inputs;
          for (auto iter : input) {
            const Edge* e = iter.second;
            merge_inputs.emplace_back(e->src()->name(), e->src_output(),
                                      e->src()->output_type(0));
          }
          int merge_attr_n = branch_num_ - skip_branchs_.size();
          std::function<Status(NodeDef&)> merge_builder = [&](NodeDef& def) {
            return NodeDefBuilder(merge_name, "Merge")
                                  .Input(merge_inputs)
                                  .Attr("T", input[0]->src()->output_type(0))
                                  .Attr("N", merge_attr_n)
                                  .Finalize(&def);
          };
          Node* merge = NodeConstructor(graph_, merge_name, switch_n_[0], merge_builder);
          TF_RETURN_FALSE_IF_NULL(merge, "construct Merge failed")
          int port = 0;

          for (int i = 0; i < branch_num_; ++i) {
            if (IsSkipBranch(i)) continue;
            graph_->AddControlEdge(no_ops[i], input[i]->src());
            graph_->AddEdge(input[i]->src(), input[i]->src_output(), merge, port);
            port++;
          }
          int reserve = collection->GetReserveBranch();
          status = graph_->UpdateEdge(merge, 0, input[reserve]->dst(),
                                      input[reserve]->dst_input());
          TF_RETURN_FALSE_IF_ERROR(status, "update merge edge")
        }
      }
      converted.insert(collection->GetReserveNode()->name());
    }
    const Edge* merge_in;
    status = merge_->input_edge(0, &merge_in);
    TF_RETURN_FALSE_IF_ERROR(status, "get merge input edge")
    for (auto e : merge_->out_edges()) {
      status = graph_->UpdateEdge(merge_in->src(), merge_in->src_output(),
                                  e->dst(), e->dst_input());
      TF_RETURN_FALSE_IF_ERROR(status, "update merge input edge")
    }
    // 删除其他分支节点
    VLOG(1) << "start remove node ";
    if (merge_->out_edges().empty()) {
      graph_->RemoveNode(merge_);
    }
    bool deleted = false;
    do {
      deleted = false;
      for (auto& collection : branch_collection_) {
        for (auto iter : collection->branch_nodes) {
          Node* n = iter.second;
          if (n == nullptr) continue;
          if (remove_set.find(n) != remove_set.end()) continue;
          if (n == collection->GetReserveNode()) {
            if (!collection->IsSwitchN()) {
              continue;
            }
          }
          if (n->out_edges().empty()) {
            deleted = true;
            VLOG(1) << "remove node " << n->name();
            graph_->RemoveNode(n);
            remove_set.insert(n);
          }
       }
      }
    } while(deleted);
    return true;
  }

  void DebugBranchNodesCollection() {
    string s = "Debug branch nodes collection\n_SwitchN nodes:\n";
    for (Node* n : switch_n_) {
      s = absl::StrCat(s, n->name(), "\n");
    }
    s = absl::StrCat(s, "merge node:\n", merge_->name(), "\n");
    VLOG(0) << s;
    for (auto& collection : branch_collection_) {
      VLOG(0) << collection->DebugString();
    }
  }
 private:
  Graph* graph_;
  const Edge* index_;
  std::vector<Node*> switch_n_;
  Node* merge_;
  //        |-->target1
  // SwitchN|-->target2
  //        |-->target3
  int branch_num_;
  std::vector<std::unique_ptr<BranchNodesCollection>> branch_collection_;
  std::set<int> skip_branchs_;
};

void DebugMultiDNNInfo(MultiDNNInfo &multi_dnn_info) {
  VLOG(0) << "Dynamic partition a: " << multi_dnn_info.dynamic_partition_a.size();
  for (Node* n : multi_dnn_info.dynamic_partition_a) {
    VLOG(0) << n->DebugString();
  }
  VLOG(0) << "Dynamic partition b:" << multi_dnn_info.dynamic_partition_b->DebugString();
  VLOG(0) << "DynamicStitch:" << multi_dnn_info.dynamic_stitch->DebugString();
  VLOG(0) << "partition:" << multi_dnn_info.partition->DebugString();
  VLOG(0) << "-------------------------";
}

void SearchTargetDynamicPartition(Graph *graph, MultiDNNInfo& info,
                                   std::set<Node*> &candidate_node) {
  std::queue<Node*> unvisited_queue;
  std::unordered_set<string> visited;
  unvisited_queue.push(info.dynamic_stitch);
  while(!unvisited_queue.empty()) {
    Node* top = unvisited_queue.front();
    unvisited_queue.pop();
    if (visited.count(top->name()) != 0) continue;
    visited.insert(top->name());
    for (auto e : top->in_edges()) {
      if (visited.count(e->src()->name()) != 0) continue;
      if (e->src() == info.dynamic_partition_b) continue;
      if (e->src()->type_string() == "DynamicPartition") {
        if (candidate_node.count(e->src())) {
          info.dynamic_partition_a.insert(e->src());
        } else {
          VLOG(0) << "find strange DynamicPartition node:" << e->src()->name();
        }
      }
      else if (e->src()->type_string() == "Placeholder" ||
               e->src()->type_string() == "PlaceholderWithDefault" ||
               e->src()->type_string() == "DynamicStitch") {
        // set search boundry
        continue;
      }
      else {
        unvisited_queue.push(e->src());
      }
    }
  }
}

// 两个DynamicPartition，一个可以替换为_SwitchN，另一个与DynamicStitch一起替换为Merge
bool DynamicPartitionToSwitch(Graph* graph, std::vector<std::shared_ptr<
                                            SubGraphCollection>>& sub_graph_group) {
  Status status;
  std::vector<Node*> nodes(graph->num_nodes());
  std::map<string, Node*> index_cache;
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  std::vector<MultiDNNInfo> multi_dnn_info;
  for (Node* node : nodes) {
    // 1.找到DynamicStitch和输入DynamicPartition
    // 2.找到DynamicPartition的输入partition(应该是Placeholder)
    // 3.partition有两种输出，一种是切分shape用的DynamicPartition_b
    //   第二种是切分输入数据用的DynamicPartition_a
    if (node->type_string() != "DynamicStitch") continue;
    VLOG(1) << "find dynamicstitch " << node->DebugString();
    MultiDNNInfo info;
    info.dynamic_stitch = node;
    info.dynamic_stitch->input_node(0, &(info.dynamic_partition_b));
    VLOG(1) << "find dynamic partition b " << info.dynamic_partition_b->DebugString();
    info.dynamic_partition_b->input_node(1, &(info.partition));
    VLOG(1) << "partition " << info.partition->DebugString();
    std::set<Node*> candidate_node;
    for (auto n:info.partition->out_nodes()) {
      if (n == info.dynamic_partition_b) continue;
      else if(n->type_string() == "DynamicPartition") {
        candidate_node.insert(n);
      } else {
        VLOG(0) << "partition output find unknown node:" << n->DebugString();
      }
    }
    SearchTargetDynamicPartition(graph, info, candidate_node);
    if (info.dynamic_partition_a.size() < 1) {
      VLOG(0) << "not found dynamic partition a node!!";
      return false;
    }
    multi_dnn_info.push_back(info);
    if (VLOG_IS_ON(1)) DebugMultiDNNInfo(info);
  }
  if (multi_dnn_info.empty()) {
    VLOG(0) << "not found multi dnn structure";
    return true;
  } else {
    VLOG(0) << "found " << multi_dnn_info.size() << " multi dnn structure";
  }
  for (MultiDNNInfo& info:multi_dnn_info) {
    VLOG(1) << "start to replace nodes";
    int switch_branch_num = -1;
    // 构建reduction indices const
    string squeeze_name = info.partition->name() + "/multi_dnn/squeeze";
    Node *squeeze = nullptr;
    if (index_cache.find(squeeze_name) == index_cache.end()) {
      // 构建begin const
      Tensor t_begin(DT_INT64, TensorShape({1}));
      auto begin_data = t_begin.tensor<int64, 1>();
      begin_data(0) = 0;
      // 构建size const
      Tensor t_size(DT_INT64, TensorShape({1}));
      auto size_data = t_size.tensor<int64, 1>();
      size_data(0) = 1;
      string slice_name = info.partition->name() + "/multi_dnn/slice";
      Node* slice = ConstructSliceOp(graph, info.partition, 0, slice_name,
                                     t_begin, t_size);
      TF_RETURN_FALSE_IF_NULL(slice, "construct slice")

      // squeeze
      string squeeze_name = info.partition->name() + "/multi_dnn/squeeze";
      squeeze = NodeConstructor(graph, squeeze_name, slice,
                      [&](NodeDef& def) {
                        return NodeDefBuilder(squeeze_name, "Squeeze")
                             .Input(slice->name(), 0, slice->output_type(0))
                             .Attr("T", slice->output_type(0))
                             .Attr("squeeze_dims", {0})
                             .Finalize(&def);
      });
      TF_RETURN_FALSE_IF_NULL(squeeze, "construct squeeze")
      graph->AddEdge(slice, 0, squeeze, 0);
      index_cache.insert(std::pair<string, Node*>(squeeze_name, squeeze));
    } else {
      squeeze = index_cache.find(squeeze_name)->second;
      VLOG(1) << "find cache squeeze:" << squeeze->DebugString();
    }
    // 4.DynamicPartition_a替换为_SwitchN
    std::vector<Node*> switch_vector;
    for (Node* dynamic_partition_a : info.dynamic_partition_a) {
      string switch_name = dynamic_partition_a->name() + "/multi_dnn/switch";
      std::vector<NodeDefBuilder::NodeOut> switch_inputs;
      const Edge *input_edge = nullptr;
      dynamic_partition_a->input_edge(0, &input_edge);
      switch_inputs.emplace_back(input_edge->src()->name(), 0,
                                 input_edge->src()->output_type(0));
      switch_inputs.emplace_back(squeeze->name(), 0, squeeze->output_type(0));
      std::set<int> src_output;
      for (auto e:dynamic_partition_a->out_edges()) {
        src_output.insert(e->src_output());
      }
      int num_partitions = dynamic_partition_a->def().attr().at("num_partitions").i();
      if (src_output.size() != num_partitions) {
        LOG(ERROR) << "DynamicPartition output num not equal num_partitions attr: "
                   << src_output.size() << " VS " << num_partitions;
        return false;
      }
      if (switch_branch_num == -1 ? false : switch_branch_num != num_partitions) {
        LOG(ERROR) << "not all dynamic partition output num equal: "
                   << switch_branch_num << " VS " << num_partitions;
        return false;
      }
      switch_branch_num = num_partitions;
      Node* switch_n = NodeConstructor(graph, switch_name, dynamic_partition_a,
              [&](NodeDef& def) {
        return NodeDefBuilder(switch_name, "_SwitchN")
                             .Input(switch_inputs[0])
                             .Input(switch_inputs[1])
                             .Attr("num_outs", dynamic_partition_a->num_outputs())
                             .Attr("T", dynamic_partition_a->output_type(0))
                             .Finalize(&def);
      });
      TF_RETURN_FALSE_IF_NULL(switch_n, "construct _SwitchN")
      graph->AddEdge(input_edge->src(), input_edge->src_output(), switch_n, 0);
      graph->AddEdge(squeeze, 0, switch_n, 1);
      status = UpdateAllEdge(graph, switch_n, dynamic_partition_a);
      TF_RETURN_FALSE_IF_ERROR(status, "update DynamicPartition output edge")
      graph->RemoveNode(dynamic_partition_a);
      switch_vector.push_back(switch_n);
    }
    // 5.DynamicPartition_b+DynamicStitch替换为Merge
    string merge_name = info.dynamic_stitch->name() + "/multi_dnn/merge";
    std::vector<NodeDefBuilder::NodeOut> merge_inputs;
    std::vector<const Edge*> stitch_inputs;
    for (auto e:info.dynamic_stitch->in_edges()) {
      if (e->src()->type_string() != "DynamicPartition") {
        stitch_inputs.emplace_back(e);
      }
    }
    // fix: 遍历stitch输入边，没有按照输入端口保序，
    // 导致switch权重与实际场景不匹配，且每次顺序都不同，通过主动sort来保序
    // 但第二阶场景数比较少时一直能够保序
    std::sort(stitch_inputs.begin(), stitch_inputs.end(),
        [](const Edge* a, const Edge* b) {
          return a->dst_input() < b->dst_input();
        });
    for (auto e:stitch_inputs) {
      merge_inputs.emplace_back(e->src()->name(), e->src_output(),
                                e->src()->output_type(0));
    }
    int input_size = stitch_inputs.size();
    Node* merge = NodeConstructor(graph, merge_name, info.dynamic_stitch,
              [&](NodeDef& def) {
      return NodeDefBuilder(merge_name, "Merge")
                            .Input(merge_inputs)
                            .Attr("T", info.dynamic_stitch->output_type(0))
                            .Attr("N", input_size)
                            .Finalize(&def);
    });
    TF_RETURN_FALSE_IF_NULL(merge, "construct merge")
    int port = 0;
    for (auto e:stitch_inputs) {
      graph->AddEdge(e->src(), e->src_output(), merge, port++);
    }
    status = UpdateAllEdge(graph, merge, info.dynamic_stitch);
    if (!status.ok()) {
      LOG(ERROR) << "update DynamicStitch output edge failed " << status;
      return false;
    }
    graph->RemoveNode(info.dynamic_partition_b);
    graph->RemoveNode(info.dynamic_stitch);
    sub_graph_group.push_back(std::make_shared<SubGraphCollection>(
                              graph, switch_vector, merge, switch_branch_num));
  }
  return true;
}

bool SkipUselessControlflowEdge(const Edge* e) {
  if (e->src()->IsSource()) {
    return true;
  }
  return false;
}

bool HasMultiControlflowEdge(Node* node) {
  bool has_controlflow = false;
  for (auto e:node->in_edges()) {
    if (SkipUselessControlflowEdge(e)) continue;
    if (e->IsControlEdge()) {
      if (!has_controlflow) {
        has_controlflow = true;
      } else {
        return true;
      }
    }
  }
  return false;
}

bool ReplaceControlflowToMergeNode(Graph* graph, Node* node) {
  std::vector<const Edge*> no_ops;
  std::vector<const Edge*> identity_ops;
  for (auto e:node->in_edges()) {
    if (SkipUselessControlflowEdge(e)) continue;
    if (e->src()->type_string() != "NoOp") {
      LOG(WARNING) << "const node input controlflow is not NoOp:" << e->DebugString();
      return false;
    }
    Node* no = e->src();
    if (no->num_inputs() > 1) {
      LOG(WARNING) << "NoOp node must have 1 input controlflow edge:" << no->DebugString();
      return false;
    }
    no_ops.push_back(e);
    const Edge* identity;
    for (auto ex:no->in_edges()) {
      if (SkipUselessControlflowEdge(e)) continue;
      if (ex->src()->type_string() != "Identity") {
        LOG(WARNING) << "NoOp input controlflow is not Identity:" << ex->DebugString();
        return false;
      }
      identity_ops.push_back(ex);
    }
  }
  int in_size = identity_ops.size();
  if (no_ops.size() != in_size) {
    LOG(WARNING) << "NoOp nodes not equal to Identity nodes:"
                 << no_ops.size() << " VS " << in_size;
    VLOG(0) << node->DebugString();
    for (auto e:no_ops) {
      VLOG(0) << e->DebugString();
    }
    for (auto e:identity_ops) {
      VLOG(0) << e->DebugString();
    }
    return false;
  }
  string merge_name = node->name() + "/merge_switch_subgraph/merge_const";
  std::vector<NodeDefBuilder::NodeOut> merge_inputs;
  for (auto e:identity_ops) {
    merge_inputs.emplace_back(e->src()->name(), 0,
                              e->src()->output_type(0));
  }
  std::function<Status(NodeDef&)> merge_builder = [&](NodeDef& def) {
    return NodeDefBuilder(merge_name, "Merge")
                          .Input(merge_inputs)
                          .Attr("T", identity_ops[0]->src()->output_type(0))
                          .Attr("N", in_size)
                          .Finalize(&def);
  };
  Node* merge = NodeConstructor(graph, merge_name,
                                identity_ops[0]->src(), merge_builder);
  TF_RETURN_FALSE_IF_NULL(merge, "construct merge")
  int merge_in_port = 0;
  for (auto e:identity_ops) {
    graph->AddEdge(e->src(), 0, merge, merge_in_port);
    merge_in_port++;
  }
  graph->AddControlEdge(merge, node);
  for (auto e:no_ops) {
    graph->RemoveEdge(e);
  }
  return true;
}

bool RefineControlflowForMergedConst(Graph* graph) {
  Status status;
  std::vector<Node*> nodes(graph->num_nodes());
  std::map<string, Node*> index_cache;
  int i = 0;
  for (Node* node : graph->nodes()) {
    nodes[i++] = node;
  }
  int counter = 0;
  for (Node* node : nodes) {
    if (node->type_string() == "Const") {
      if (HasMultiControlflowEdge(node)) {
        if (!ReplaceControlflowToMergeNode(graph, node)) {
          LOG(WARNING) << "refine controlflow for merged const failed";
          return false;
        }
        counter++;
      }
    }
  }
  VLOG(0) << "convert " << counter << " multi controlflow const to merge structure";
  return true;
}

bool SwitchSubGraphToSwitchWeight(Graph* graph, const string& skip_branchs_str,
                std::vector<std::shared_ptr<SubGraphCollection>>& sub_graph_group) {
  Status status;
  VLOG(0) << "skip branchs " << skip_branchs_str;
  for (std::shared_ptr<SubGraphCollection>& collection : sub_graph_group) {
    // 1.先将_SwitchN到Merge之间的branch子图遍历，收集到计算节点和Const信息
    if (!collection->CollecteBranchNodes(skip_branchs_str)) {
      collection->DebugBranchNodesCollection();
      return false;
    }
    // 2.根据收集到的信息转化Switch子图为Switch const，复用第一个branch的计算节点
    if (!collection->ConvertToSwitchWeight()) {
      return false;
    }
    VLOG(0) << "convert subgraph to switch weight done";
  }
  // arithmetic optimization可能会将相同的const合并
  // 导致多个switch分支出来的控制边连接到同一个const
  // 从而无法正常调度，导致后续节点状态变成dead
  // 有相同const被共享，需要增加merge来保证节点正常被调度到
  if (!RefineControlflowForMergedConst(graph)) {
    return false;
  }
  return true;
}

}  // end namespace

Status MultiDNNSwitchOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool multi_dnn_switch = true;
  ReadBoolFromEnvVar("TF_ENABLE_ORIGINAL_DELIVERY_OPTIMIZE", true, &multi_dnn_switch);
  if (!multi_dnn_switch) {
    *optimized_graph = item.graph;
    return Status::OK();
  }
  static int pass = 0;
  VLOG(0) << "MultiDNNSwitchOptimizer is on." << pass;
  if (VLOG_IS_ON(1)) {
    string file = "before_multi_dnn_switch_" + std::to_string(pass) + ".pb";
    DumpModelFile(item.graph, file);
  }

  // convert graphdef to graph
  FunctionLibraryDefinition flib(OpRegistry::Global(), item.graph.library());
  Graph graph(flib);
  Status status = ConvertGraphDefToGraph(GraphConstructorOptions(),
                                         item.graph, &graph);
  if (!status.ok()) {
    LOG(WARNING) << "ConvertGraphDefToGraph failed: " << status.ToString();
    *optimized_graph = item.graph;
    return Status::OK();
  }

  std::vector<std::shared_ptr<SubGraphCollection>> sub_graph_group;
  bool result = DynamicPartitionToSwitch(&graph, sub_graph_group);
  if (!result || sub_graph_group.empty()) {
    if (!result) {
      LOG(WARNING) << "optimized multi dnn DynamicPartition to Switch failed";
    }
    *optimized_graph = item.graph;
    return Status::OK();
  }
  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (!SwitchSubGraphToSwitchWeight(&graph, skip_branchs_str_, sub_graph_group)) {
    LOG(WARNING) << "optimized multi dnn switch subgraph to switch weight failed";
    return Status::OK();
  }

  // convert graph to graphdef
  graph.ToGraphDef(optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();

  if (VLOG_IS_ON(1)) {
    string file = "after_multi_dnn_switch_" + std::to_string(pass) + ".pb";
    DumpModelFile(*optimized_graph, file);
  }
  pass++;
  return Status::OK();
}

void MultiDNNSwitchOptimizer::Feedback(tensorflow::grappler::Cluster *cluster,
                             const tensorflow::grappler::GrapplerItem &item,
                             const tensorflow::GraphDef &optimized_graph, double result) {
  // no-op
}

}  // end namespace grappler
}  // end namespace tensorflow

