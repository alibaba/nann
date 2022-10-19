#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#endif
#include "./blaze_predictor.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
using tensorflow::se::Event;
#endif
namespace tensorflow {
const int kBlazeStartStepId = 1024;
const std::string kCpuDeviceName = "/job:localhost/replica:0/task:0/device:CPU:0";

mutex BlazePredictor::session_mu_;
BlazePredictor::SessionMap BlazePredictor::session_map_;
mutex BlazePredictor::log_mu_;

BlazePredictor::BlazePredictor(OpKernelConstruction* ctx) : device_type_(ctx->device_type().type()) {
  ReadInt64FromEnvVar("BLAZE_LOG_LEVEL", 0, &log_level_);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_def", &graph_def_str_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(ctx, ParseAttr(ctx->def().device()));
  ctx_ = ctx;
}

BlazePredictor::BlazePredictor(const std::vector<std::string>& input_names,
                          const std::vector<std::string>& output_names,
                          const GraphDef& graph_def, const std::string& device,
                          const BlazeKernelOptions& options, const string& device_string,
                          const std::vector<DataType>& input_types,
                          OpKernelConstruction* ctx) :
    input_names_(input_names), output_names_(output_names),
    graph_def_(graph_def), request_device_(device),
    blaze_run_options_(options), device_type_(device_string),
    input_types_(input_types), ctx_(ctx) {
  ReadInt64FromEnvVar("BLAZE_LOG_LEVEL", 0, &log_level_);
  // rewrite HugeConst
  std::string root_path;
  auto status = ctx_->GetAttr("_extra_conf_root_path", &root_path);
 	if (status.ok()) {
    LOG(INFO) << "get _extra_conf_root_path" << root_path;
 	  for (int i = 0; i < graph_def_.node_size() ;i++) {
      auto node = graph_def_.mutable_node(i);
      if (node->op() == "HugeConst") {
        auto* attr_map = node->mutable_attr();
        if (attr_map != nullptr) {
          auto it = attr_map->find("path");
          if (it != attr_map->end()) {
            std::string ad_vec_path = root_path + '/' + it->second.s();
            it->second.set_s(ad_vec_path);
            LOG(INFO) << "Update path attr of HugeConst op " << node->name() << " to " << ad_vec_path;
          }
        }
      }
    }
  } { LOG(INFO) << "not get _extra_conf_root_path" << root_path; }
}

BlazePredictor::~BlazePredictor() {
  if (!session_key_.empty()) {
    mutex_lock l(session_mu_);
    auto it = session_map_.find(session_key_); 
    if (it == session_map_.end()) {
      LOG(ERROR) << "release session not found in static map, should not happen";
      return;
    }
    if (--(it->second.count) == 0) {
      session_map_.erase(it);
    }
 	}
}

Status BlazePredictor::ParseAttr(const std::string& device) {
  if (!ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    VLOG(0) << "Parse blaze options from file failed, try as readable string";
  } else {
    if (!::tensorflow::protobuf::TextFormat::ParseFromString(
            blaze_option_path_, &blaze_run_options_)) {
      return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
    }
  }

  if (!ReadTextProto(Env::Default(), graph_def_str_, &graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_def_str_, &graph_def_).ok()) {
      LOG(ERROR) << "Parse graph from " << graph_def_str_ << " failed";
      return errors::Internal("Parse graph from ", graph_def_str_, " failed");
    }
  }
  
  if (device.size() == 0) {
    return errors::Internal("ctx device not set");
  }
  request_device_ = device;

  return Status::OK();
}

Status BlazePredictor::GenSessionOptions(SessionOptions& options) {
  options.config.MergeFrom(blaze_run_options_.config_proto());
  //disable caller thread
  options.config.set_force_run_in_caller_thread(false);
  options.config.set_is_blaze(true);
  return Status::OK();
}

Status BlazePredictor::PrepareGraph(GraphDef& graph_def) {
  const char* const kDevicePrefix = "/job:localhost/replica:0/task:0";
  const char* const kBlazeDevicePrefix = "/device:";
  auto st = ctx_->GetAttr(kBlazeRealDevice, &blaze_real_deive_);
  if (!st.ok()) {
    device_ = kDevicePrefix + request_device_;
  } else {
    if (blaze_real_deive_.rfind(kBlazeDevicePrefix, 0) != 0) {
      return errors::Internal("node attr ", kBlazeRealDevice,
          " : ", blaze_real_deive_, " not startswith ", kBlazeDevicePrefix);
    }
    device_ = kDevicePrefix + blaze_real_deive_;
  }

  LOG(INFO) << "BlazePredictor will use device " << device_;
  graph_def = graph_def_;
  SetDeviceInGraphDef(device_, &graph_def);
  SetCPUDeviceInGraphDef(kCpuDeviceName, &graph_def);

  return Status::OK();
}

Status BlazePredictor::MakeCallable() {
  CallableOptions callable_options;
  TF_RETURN_IF_ERROR(PrepareCallableOptions(callable_options));
  LOG(INFO) << "create session with callable options " <<
        callable_options.DebugString();
  return session_->MakeCallable(callable_options, &handle_);
}

Status BlazePredictor::PrepareCallableOptions(CallableOptions &callable_options) {
  std::set<std::string> cpu_inputs;
  for (const auto& node : graph_def_.node()) {
    if (node.op() == "Placeholder") {
      auto it = node.attr().find("dtype");
      if (it != node.attr().end()) {
        if (it->second.type() == DT_INT32 || it->second.type() == DT_UINT32) {
          cpu_inputs.insert(node.name());
        }
      }
    }
  }
  for (const auto& input : input_names_) {
    if (cpu_inputs.find(input) == cpu_inputs.end()) {
      callable_options.add_feed(input);
      callable_options.mutable_feed_devices()->insert({input, device_});
      copyable_.push_back(true);
    } else {
      callable_options.add_feed(input);
      callable_options.mutable_feed_devices()->insert({input, kCpuDeviceName});
      copyable_.push_back(false);
    }
  }

  for (const auto& output : output_names_) {
    callable_options.add_fetch(output);
    callable_options.mutable_fetch_devices()->insert({output, device_});
  }
  callable_options.set_fetch_skip_sync(true);
  return Status::OK();
}

Status BlazePredictor::Warmup() {
  return Status::OK();
}

Status BlazePredictor::InitSession() {
  TF_RETURN_IF_ERROR(PrepareData());

  SessionOptions options;
  *(options.config.mutable_gpu_options()) = BlazeConfSingleton::GetInstance()
      ->GetConfig().gpu_options();

  options.config.MergeFrom(blaze_run_options_.config_proto());
  TF_RETURN_IF_ERROR(GenSessionOptions(options));
  auto status = ctx_->GetAttr("_session_key", &session_key_);
 	if (status.ok()) {
 	  LOG(INFO) << "_session_key detected, static Session enabled";
 	} else {
 	  LOG(INFO) << "_session_key not detected, won't use static predictor map";
 	  session_key_.clear();
  }
  {
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(PrepareGraph(graph_def));
 	  mutex_lock l(session_mu_);
    auto it = session_map_.find(session_key_);
    if (it == session_map_.end()) {
      VLOG(0) << "create session with config " << options.config.DebugString();
      session_ = std::shared_ptr<Session>(NewSession(options));
      if (session_ == nullptr) {
        LOG(ERROR) << "create session failed";
        return errors::Internal("Create session failed");
      }
      auto dir_session = reinterpret_cast<DirectSession*>(session_.get());
      dir_session->SetStepInitId(kBlazeStartStepId);
      LOG(INFO) << "Blaze start with step id " << kBlazeStartStepId;
      LOG(INFO) << "Creat session succ " << this;

      auto status = session_->Create(graph_def);
      if (!status.ok()) {
        LOG(ERROR) << "create session with GraphDef failed " << status.ToString();
        return status;
      }
      TF_RETURN_IF_ERROR(MakeCallable());
      LOG(INFO) << "MakeCallable succ " << this;
      if (!session_key_.empty()) {
        session_map_.emplace(session_key_, SessionTuple(session_, 1, handle_));
      }
    } else {
      CallableOptions callable_options;
      TF_RETURN_IF_ERROR(PrepareCallableOptions(callable_options));
      session_ = it->second.session;
      it->second.count++;
      handle_ = it->second.handle;
    }
    TF_RETURN_IF_ERROR(SetDeviceInfo(ctx_));

    auto warm_status = Warmup();
    if (warm_status != Status::OK()) {
      return warm_status;
    }
    return  Status::OK();
  }
}

Status BlazePredictor::Compute(OpKernelContext* ctx) {

  int num_inputs = ctx->num_inputs();
  if (num_inputs != input_names_.size()) {
    return errors::Internal("ctx input size ", num_inputs,
        " != ", input_names_.size());
  }
  if (ctx->num_outputs() != output_names_.size()) {
    return errors::Internal("ctx output size ", ctx->num_outputs(),
        " != ", output_names_.size());
  }

  std::vector<Tensor> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(ctx->input(i));
  }

  std::vector<Tensor> outputs;

  std::vector<Tensor> real_inputs(inputs.size());
  TF_RETURN_IF_ERROR(PrepareInputs(inputs, &real_inputs, ctx));

  TF_RETURN_IF_ERROR(session_->RunCallable(handle_, real_inputs, &outputs, nullptr));

  std::vector<Tensor> real_outputs(outputs.size());
  TF_RETURN_IF_ERROR(PrepareOutputs(outputs, &real_outputs, ctx));
  for (int i = 0; i < real_outputs.size(); ++i) {
    ctx->set_output(i, real_outputs[i]);
  }
  return Status::OK();
}

void BlazePredictor::SetDeviceInGraphDef(const std::string device_name,
                                         GraphDef* graph_def) {
  VLOG(2) << "Before setting device: \n" << graph_def->DebugString();
  int node_size = graph_def->node_size();
  for (int i = 0; i < node_size; i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->device() == "/device:CPU:0") {
      VLOG(1) << "node " << node->name() << " device /device:CPU:0, do not overwrite to GPU";
      continue;
    }
    node->set_device(device_name);
  }
  VLOG(2) << "After setting device: \n" << graph_def->DebugString();
}

void BlazePredictor::SetCPUDeviceInGraphDef(const std::string device_name,
                                            GraphDef* graph_def) {
  VLOG(2) << "Before setting device: \n" << graph_def->DebugString();
  int node_size = graph_def->node_size();
  for (int i = 0; i < node_size; i++) {
    NodeDef* node = graph_def->mutable_node(i);
    if (node->device() == "/device:CPU:0") {
      VLOG(1) << "node " << node->name() << " device /device:CPU:0, overwrite to " << device_name;
      node->set_device(device_name);
    }
  }
  VLOG(2) << "After setting device: \n" << graph_def->DebugString();
}

Status BlazePredictor::SetDeviceInfo(OpKernelConstruction* ctx) {
  auto st = ctx->GetAttr(kBlazeRealDevice, &blaze_real_deive_);
  if (!st.ok()) {
    VLOG(0) << "Blaze not set device, using " << request_device_;
    same_device_ = true;
    blaze_device_ = nullptr;
    return Status::OK();
  } else {
    DeviceNameUtils::ParsedName req_name;
    DeviceNameUtils::ParsedName blaze_name;
    if (!DeviceNameUtils::ParseFullName(request_device_, &req_name)) {
      return errors::Internal(request_device_, " parse failed");
    }

    if (!DeviceNameUtils::ParseFullName(blaze_real_deive_, &blaze_name)) {
      return errors::Internal(blaze_real_deive_, " parse failed");
    }
    
    auto req_dev = DeviceNameUtils::LocalName(request_device_);
    auto blaze_dev = DeviceNameUtils::LocalName(blaze_real_deive_);

    vgpu_id_ = blaze_name.id;
    if (req_dev != blaze_dev) {
      VLOG(0) << "req_dev: " << req_dev << "; blaze_dev: " << blaze_dev;
      if (req_name.type == "cpu" || blaze_name.type == "CPU") {
        return errors::Internal("req_dev.type: ", req_name.type,
            ", blaze_dev.type: ", blaze_name.type, " not supported");
      }
      same_device_ = false;
      const DeviceMgr* mgr = nullptr;
      TF_RETURN_IF_ERROR(session_->LocalDeviceManager(&mgr));
      if (mgr == nullptr) {
        return errors::Internal("DeviceMgr not found");
      }
      TF_RETURN_IF_ERROR(mgr->LookupDevice(blaze_dev, &blaze_device_));
      auto* dev_info = blaze_device_->tensorflow_gpu_device_info();
      if (!dev_info) {
        return errors::Internal("get gpu device info failed");
      }
      AllocatorAttributes alloc_attrs;
      alloc_attrs.set_on_host(false);
      blaze_allocator_ = blaze_device_->GetAllocator(alloc_attrs);
      if (!blaze_allocator_) {
        return errors::Internal("get gpu allocator failed");
      }
      stream_ = GetStream();
      if (!stream_) {
        return errors::Internal("get stream_ for ", blaze_device_, " failed" );
      }
    }
    return Status::OK();
  }
}

stream_executor::Stream* BlazePredictor::GetStream() const {
  #if GOOGLE_CUDA
  TfGpuId tf_gpu_id(vgpu_id_);
  auto* se = GpuIdUtil::ExecutorForTfGpuId(tf_gpu_id).ValueOrDie();

  if (!se) { return nullptr; }
  static tensorflow::GPUOptions gpu_options;
  auto sg = tensorflow::StreamGroupFactory::Global().GetOrCreate(
      tf_gpu_id, 0, se, gpu_options);
  if (!sg) {
    VLOG(0) << "get stream group failed";
    return nullptr;
  }
  return sg->compute;

  #else
    return nullptr;
  #endif
}

Status BlazePredictor::PrepareInputs(const std::vector<Tensor>& inputs,
  std::vector<Tensor>* real_inputs, OpKernelContext* ctx) {
  if (!same_device_) {
    return CopyTensorCPUToGPU(inputs, real_inputs, ctx);
  }
  *real_inputs = inputs;
  return Status::OK();
}

Status BlazePredictor::CopyTensorCPUToGPU(const std::vector<Tensor>& inputs,
    std::vector<Tensor>* real_inputs,
    OpKernelContext* ctx) {

  for (int i = 0; i < inputs.size(); ++i) {
    if (!copyable_[i]) {
      (*real_inputs)[i] = inputs[i];
      continue;
    }
    Tensor copyed_tensor(blaze_allocator_, inputs[i].dtype(), inputs[i].shape());
    (*real_inputs)[i] = copyed_tensor;

    const uint8* input_ptr = (uint8*)GetTensorAddress(&inputs[i]);
    uint8* real_ptr = (uint8*)GetTensorAddress(&(*real_inputs)[i]);
    uint64 input_size = GetTensorSize(&inputs[i]);
    uint64 real_size = GetTensorSize(&(*real_inputs)[i]);
    if (input_ptr == nullptr || real_ptr == nullptr) {
      return errors::Internal(
          "Error when getting input address or size");
    }
#if GOOGLE_CUDA
      auto real_dev_ptr = AsDeviceMemory(real_ptr, real_size);
      bool copy_status =
          GetStream()->ThenMemcpy(&real_dev_ptr, input_ptr, input_size).ok();
      if (!copy_status) {
        return errors::Internal("MemcpyH2D for padding inputs failed.");
      }
#else
      return errors::Internal("CUDA not suaported");
#endif
  }
  return Status::OK();
}

Status BlazePredictor::PrepareOutputs(const std::vector<Tensor>& outputs,
  std::vector<Tensor>* real_outputs, OpKernelContext* ctx) {
  if (!same_device_) {
    return CopyTensorGPUToCPU(outputs, real_outputs, ctx);
  }
  *real_outputs = outputs;
  return Status::OK();
}

Status BlazePredictor::CopyTensorGPUToCPU(const std::vector<Tensor>& gpu_tensors,
    std::vector<Tensor>* cpu_tensors,
    OpKernelContext* ctx) {
  for (int i = 0; i < gpu_tensors.size(); ++i) {
#if GOOGLE_CUDA
    TensorShape slice_to_shape = gpu_tensors[i].shape();
    const auto& tmp_tensor = gpu_tensors[i];
    uint8* tmp_ptr = (uint8*)GetTensorAddress(&tmp_tensor);
    uint64 tmp_size = GetTensorSize(&tmp_tensor);
    auto tmp_dev_ptr = AsDeviceMemory(tmp_ptr, tmp_size);
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_on_host(true);
    alloc_attrs.set_gpu_compatible(true);
    TF_RETURN_IF_ERROR(ctx->allocate_temp(tmp_tensor.dtype(),
          tmp_tensor.shape(), &((*cpu_tensors)[i]), alloc_attrs));
    uint8* host_add = (uint8*)GetTensorAddress(&((*cpu_tensors)[i]));
    auto stream = GetStream();
    stream->ThenMemcpy(host_add, tmp_dev_ptr, tmp_size);
    auto event = std::make_shared<Event>(stream->parent());
    if (!event->Init()) {
      LOG(ERROR) << "event init failed!";
      return errors::Internal("SliceToDynamic GPU2CPU failed event init");
    }
    stream->ThenRecordEvent(event.get());
    stream->ThenSynchronizeEvent(event.get());
#else
    return errors::Internal("cuda not supported");
#endif
  }
  return Status::OK();
}
}
