#pragma once
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace tensorflow;
namespace benchmark {

Status GetDeviceCount(const ConfigProto& config, int* cpu_count, int* gpu_count);

}  // namespace benchmark
