#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA
#include <cuda_runtime.h>
#endif  // GOOGLE_CUDA

using namespace tensorflow;
namespace benchmark {

static int GetGpuCount() {
  int gpu_count = 0;
#if GOOGLE_CUDA
  cudaError_t error_id = cudaGetDeviceCount(&gpu_count);
  if (error_id != cudaSuccess) {
    LOG(ERROR) << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
               << " -> " << cudaGetErrorString(error_id);
    return 0;
  }
#endif  // GOOGLE_CUDA
  return gpu_count;
}

// Parse 'visible_device_list' into a list of platform GPU ids.
Status ParseVisibleDeviceList(const std::string& visible_device_list,
                                          std::vector<int>* visible_gpu_order, int gpu_count) {
  visible_gpu_order->clear();

  // If the user wants to remap the visible to virtual GPU mapping,
  // check for that here.
  if (visible_device_list.empty()) {
    visible_gpu_order->resize(gpu_count);
    // By default, visible to virtual mapping is unchanged.
    int deviceNo = 0;
    std::generate(visible_gpu_order->begin(), visible_gpu_order->end(),
                  [&deviceNo] { return deviceNo++; });
  } else {
    const std::vector<std::string> order_str =
        str_util::Split(visible_device_list, ',');
    for (const std::string& platform_gpu_id_str : order_str) {
      int platform_gpu_id;
      if (!strings::safe_strto32(platform_gpu_id_str, &platform_gpu_id)) {
        return errors::InvalidArgument(
            "Could not parse entry in 'visible_device_list': '",
            platform_gpu_id_str,
            "'. visible_device_list = ", visible_device_list);
      }
      if (platform_gpu_id < 0 ||
          platform_gpu_id >= gpu_count) {
        return errors::InvalidArgument(
            "'visible_device_list' listed an invalid GPU id '", platform_gpu_id,
            "' but visible device count is ", gpu_count);
      }
      visible_gpu_order->push_back(platform_gpu_id);
    }
  }

  // Validate no repeats.
  std::set<int> visible_device_set(visible_gpu_order->begin(),
                                   visible_gpu_order->end());
  if (visible_device_set.size() != visible_gpu_order->size()) {
    return errors::InvalidArgument(
        "visible_device_list contained a duplicate entry: ",
        visible_device_list);
  }
  return Status::OK();
}

Status GetDeviceCount(const ConfigProto& config, int* cpu_count, int* gpu_count) {
  // get cpu count from config.device_count
  *cpu_count = 1;
  auto iter = config.device_count().find("CPU");
  if (iter != config.device_count().end()) {
    *cpu_count = iter->second;
    if (*cpu_count < 0) *cpu_count = 1;
  }

  // get system gpu count
  int total_gpus = GetGpuCount();
  *gpu_count = total_gpus;
  // get gpu count from config.device_count
  auto iter1 = config.device_count().find("GPU");
  if (iter1 != config.device_count().end()) {
    int temp = iter1->second;
    if (temp >= 0 && temp < *gpu_count) *gpu_count = temp;
  }
  if (*gpu_count == 0) return Status::OK();

  // parse gpu_options.visible_device_list
  const GPUOptions& gpu_options = config.gpu_options();
  std::vector<int> visible_gpu_order;
  TF_RETURN_IF_ERROR(ParseVisibleDeviceList(gpu_options.visible_device_list(),
                                            &visible_gpu_order, total_gpus));
  int visible_gpus = visible_gpu_order.size();
  if (*gpu_count > visible_gpus) *gpu_count = visible_gpus;
  if (*gpu_count == 0) return Status::OK();

  // parse gpu_options.experimental().virtual_devices
  const auto& virtual_devices = gpu_options.experimental().virtual_devices();
  if (virtual_devices.empty()) return Status::OK();
  if (total_gpus < virtual_devices.size()) {
    return errors::Unknown(
        "Not enough GPUs to create virtual devices."
        " total_gpus: ", total_gpus, " #virtual_devices: ", virtual_devices.size());
  }
  if (!gpu_options.visible_device_list().empty() &&
      visible_gpu_order.size() != virtual_devices.size()) {
    return errors::InvalidArgument(
        "The number of GPUs in visible_device_list doesn't match the number "
        "of elements in the virtual_devices list.",
        " #GPUs in visible_device_list: ", visible_gpu_order.size(),
        " virtual_devices.size(): ", virtual_devices.size());
  }
  int next_tf_gpu_id = 0;
  for (int i = 0; i < *gpu_count; ++i) {
    if (virtual_devices.empty() || virtual_devices.Get(i).memory_limit_mb_size() == 0) {
      next_tf_gpu_id++;
    } else {
      next_tf_gpu_id += virtual_devices.Get(i).memory_limit_mb_size();
    }
  }
  *gpu_count = next_tf_gpu_id;
  return Status::OK();
}

}  // namespace benchmark
