#pragma once
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace tensorflow;
namespace benchmark {

void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<std::string, const NodeDef*>* result);

void NodeNamePartsFromInput(const std::string& input_name, std::string* prefix,
                            std::string* node_name, std::string* suffix);

std::string NodeNameFromInput(const std::string& input_name);

std::string NewNodeNameFromInput(const std::string& input_name);

}  // namespace benchmark
