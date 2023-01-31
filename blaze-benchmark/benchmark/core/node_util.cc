#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/str_util.h"

using namespace tensorflow;
namespace benchmark {

void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<std::string, const NodeDef*>* result) {
  for (const NodeDef& node : graph_def.node()) {
    (*result)[node.name()] = &node;
  }
}

void NodeNamePartsFromInput(const std::string& input_name, std::string* prefix,
                            std::string* node_name, std::string* suffix) {
  std::vector<std::string> input_parts = str_util::Split(input_name, ':');
  if (input_parts.size() < 2) {
    *suffix = "";
  } else {
    *suffix = ":" + input_parts[1];
  }
  StringPiece node_name_piece(input_parts[0]);
  if (absl::ConsumePrefix(&node_name_piece, "^")) {
    *prefix = "^";
  } else {
    *prefix = "";
  }
  *node_name = std::string(node_name_piece);
}

std::string NodeNameFromInput(const std::string& input_name) {
  std::string prefix;
  std::string node_name;
  std::string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  return node_name;
}

std::string NewNodeNameFromInput(const std::string& input_name) {
  std::vector<std::string> input_parts = str_util::Split(input_name, ':');
  if (input_parts.size() < 2) {
    return input_parts[0];
  } else {
    if (input_parts[1] == "0") return input_parts[0];
    return input_parts[0] + "/" + input_parts[1];
  }
}

}  // namespace benchmark
