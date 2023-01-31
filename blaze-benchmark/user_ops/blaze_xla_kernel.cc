/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/kernels/benchmark_helper.h"
#include "tensorflow/core/kernels/blaze_predictor.h"
#include "tensorflow/core/kernels/blaze_xla_predictor.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/compiler/jit/flags.h"

namespace tensorflow {

REGISTER_OP("BlazeXlaOp")
    .Attr("InT: list({int8,int64,float16,float32,int32})")
    .Attr("OutT: list({int8,int64,float16,float32,int32})")
    .Attr("input_names: list(string) >= 0")
    .Attr("output_names: list(string) >= 0")
    .Attr("graph_def: string")
    .Attr("blaze_option_path: string")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

class BlazeXlaOp : public AsyncOpKernel {
 public:
  explicit BlazeXlaOp(OpKernelConstruction* context);
  ~BlazeXlaOp() { total_running_counter_ -= running_counter_; }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

  Status ParseRunOptions(BlazeKernelOptions& run_options);
 private:
  Status ParseAttr();
  void InitPredictor(OpKernelConstruction* context);
  void TraceTensors(OpKernelContext* ctx);
  void CopyTensor(MemoryType, OpKernelContext* ctx,
                  const string& name, const Tensor& tensor);

  void Schedule(OpKernelContext* ctx, const DoneCallback& done, uint64 begin);
  void Running(OpKernelContext* ctx, const DoneCallback& done, uint64 begin);
  void ComputeNormal(OpKernelContext* context, const DoneCallback& done);
  void ComputeBenchmark(OpKernelContext* context, const DoneCallback& done);
  void ComputeNull(OpKernelContext* context, const DoneCallback& done);
  int GetBatchSizeUnsafe(OpKernelContext* context);

  DeviceType device_type_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string blaze_option_path_;
  std::string graph_def_path_;
  std::string graph_def_str_;
  string device_string_;
  std::vector<DataType> input_types_;
  
  std::string device_;
  GraphDef graph_def_;
  std::unique_ptr<BlazePredictor> predictor_;
  BlazeKernelOptions blaze_run_options_;
  Env* env_;
  mutex tracing_mu_;
  mutex benchmark_mu_;
  mutex running_mu_;
  std::atomic<int> benchmark_counter_;
  int wait_ns_;

  tensorflow::thread::ThreadPool pool_;
  tensorflow::thread::ThreadPool schedule_pool_;
  std::atomic<int> running_counter_;
  const int kBlazeRunningCount_;
  const int kScheduleFactor_ = 2;
  const int kSleepMicroSeconds = 50;

  std::atomic<int> waiting_counter_;
  const int kMaxWaitingCount_;
  
  static std::atomic<int> total_waiting_counter_;
  static std::atomic<int> total_running_counter_;

  int counter_per_session_;
};

std::atomic<int> BlazeXlaOp::total_waiting_counter_(0);
std::atomic<int> BlazeXlaOp::total_running_counter_(0);

int BlazeThreadsCount() {
  const int kDefaultDenseThreadsNum = 2;
  int64 dense_threads_num;
  ReadInt64FromEnvVar("BLAZE_THREADS_NUM", kDefaultDenseThreadsNum, &dense_threads_num);
  VLOG(0) << "blaze set thread pool size " << dense_threads_num;
  return (int)dense_threads_num;
}

int BlazeWatingCount() {
  const int kDefaultWaitingCount = 10;
  int64 dense_waiting_num;
  ReadInt64FromEnvVar("DENSE_MAX_WAITING_COUNT", kDefaultWaitingCount, &dense_waiting_num);
  VLOG(0) << "blaze max waiting count " << dense_waiting_num;
  return (int)dense_waiting_num;
}

void BlazeXlaOp::InitPredictor(OpKernelConstruction* context) {
  auto config = blaze_run_options_.mutable_config_proto();
  config->set_allow_soft_placement(true);
  config->mutable_gpu_options()->set_allow_growth(true);

  if (blaze_run_options_.use_single_threaded_executor()) {
    config->mutable_experimental()->set_executor_type("SINGLE_THREADED_EXECUTOR");
  }
  LOG(INFO) << "Blaze create with options " << blaze_run_options_.DebugString();
  if (blaze_run_options_.xla_compilation()) {
    auto jitLevel = OptimizerOptions::ON_1;
    {
      tensorflow::BuildXlaOpsPassFlags* flags =
          tensorflow::GetBuildXlaOpsPassFlags();
      flags->tf_xla_enable_lazy_compilation = false;
    }
    {
      tensorflow::MarkForCompilationPassFlags* flags =
          tensorflow::GetMarkForCompilationPassFlags();
      flags->tf_xla_cpu_global_jit = true;
      flags->tf_xla_min_cluster_size = 1;
    }
    config->mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(jitLevel);
    predictor_ = absl::make_unique<BlazeXlaPredictor>(input_names_, output_names_,
                                       graph_def_, device_, blaze_run_options_,
                                       device_string_, input_types_, context, blaze_run_options_.config_proto().enable_xla_auto_padding());
  } else {
    predictor_ = absl::make_unique<BlazePredictor>(input_names_, output_names_,
                                    graph_def_, device_, blaze_run_options_,
                                    device_string_, input_types_, context);
  }
}

BlazeXlaOp::BlazeXlaOp(OpKernelConstruction* context)
    : AsyncOpKernel(context), device_type_(context->device_type().type()), 
    pool_(Env::Default(), "blaze_kernel", BlazeThreadsCount() * 2),
    schedule_pool_(Env::Default(), "blaze_schedule_kernel", 2), running_counter_(0),
    kBlazeRunningCount_(BlazeThreadsCount()), waiting_counter_(0),
    kMaxWaitingCount_(BlazeWatingCount()) {
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(context, context->GetAttr("graph_def", &graph_def_path_));
  OP_REQUIRES_OK(context, context->GetAttr("blaze_option_path", &blaze_option_path_));
  OP_REQUIRES_OK(context, context->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(context, ParseAttr());
  device_string_ = context->device_type().type_string();
  device_ = context->def().device();
  InitPredictor(context);
  OP_REQUIRES_OK(context, predictor_->InitSession());
  env_ = Env::Default();
  benchmark_counter_ = 0;
  wait_ns_ = blaze_run_options_.wait_ms() * 1000000;

  const char* const kBlazeCounter = "_blaze_counter";
  int session_counter;
  OP_REQUIRES_OK(context, context->GetAttr(kBlazeCounter, &session_counter));
  int total_count = BlazeThreadsCount();
  int per_counter = session_counter > 0 ?
                    total_count / session_counter : total_count;
  counter_per_session_ = per_counter > 0 ? per_counter : total_count;
  VLOG(0) << "blaze session count: " << session_counter <<
             ", predictor per session: " << counter_per_session_;
}

Status BlazeXlaOp::ParseAttr() {
  if (ReadTextProto(Env::Default(), blaze_option_path_,
                     &blaze_run_options_).ok()) {
    VLOG(0) << "Parse blaze options from file succ";
  } else {
    if (!::tensorflow::protobuf::TextFormat::ParseFromString(
            blaze_option_path_, &blaze_run_options_)) {
      LOG(ERROR) << "Parse proto from " << blaze_option_path_ << " failed";
      return errors::Internal("parse proto from ", blaze_option_path_,  " failed");
    }
    LOG(INFO) << "Parse blaze options succ " << blaze_run_options_.DebugString();
  }

  if (!ReadTextProto(Env::Default(), graph_def_path_,
                     &graph_def_).ok()) {
    if (!ReadBinaryProto(Env::Default(), graph_def_path_,
                       &graph_def_).ok()) {
      return errors::Internal("parse proto from ", graph_def_path_,  " failed");
    }
  }

  // graph_def_str_ = graph_def_.DebugString();
  
  return Status::OK();
}

void BlazeXlaOp::ComputeNormal(OpKernelContext* ctx, const DoneCallback& done) {
  auto begin = env_->NowNanos();
  Schedule(ctx, done, begin);
}

void BlazeXlaOp::ComputeBenchmark(OpKernelContext* ctx, const DoneCallback& done) {
  if (benchmark_counter_ < 200) {
    // out from warmup
    ComputeNormal(ctx, done);
    ++benchmark_counter_;
  } else {
    auto& helper = BenchmarkHelper::GetInstance();
    helper.Start();
    mutex_lock l(benchmark_mu_);
    while(1) {
      auto start_ns = env_->NowNanos();
      predictor_->Compute(ctx);
      auto end_ns = env_->NowNanos();
      helper.RecordTM((end_ns - start_ns) / 1000000.0f);
      helper.Add();
      for (int i = 0; i < ctx->num_outputs(); ++i) {
        ctx->release_output(i);
      }
    }
  }
  done();
}

void BlazeXlaOp::ComputeNull(OpKernelContext* context, const DoneCallback& done) {
  auto batch_size = GetBatchSizeUnsafe(context);
  Tensor *output;
  context->allocate_output(0, {batch_size, 2}, &output);
  done();
}

void BlazeXlaOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  switch(blaze_run_options_.run_mode()) {
    case BlazeKernelOptions::DEFAULT: {
      ComputeNormal(ctx, std::move(done));
      break;
    }
    case BlazeKernelOptions::BENCHMARK: {
      ComputeBenchmark(ctx, std::move(done));
      break;
    }
    case BlazeKernelOptions::SKIP: {
      ComputeNull(ctx, std::move(done));
      break;
    }
    default: {
      ComputeNormal(ctx, std::move(done));
    }
  }
}

void BlazeXlaOp::TraceTensors(OpKernelContext* ctx) {
  if (ctx->status().ok()) {
    int num_inputs = ctx->num_inputs();
    for (int i = 0; i < num_inputs; ++i) {
      const auto& tensor = ctx->input(i);
      const auto& name = input_names_[i];
      CopyTensor(ctx->input_memory_type(i), ctx, name, tensor);
    }

    for (int i = 0; i < ctx->num_outputs(); ++i) {
      const auto tensor = ctx->mutable_output(i);
      const auto& name = output_names_[i];
      CopyTensor(ctx->output_memory_type(i), ctx, name, *tensor);
    }
  }
}

void BlazeXlaOp::CopyTensor(MemoryType mtype, OpKernelContext* ctx,
                            const string& name, const Tensor& tensor) {
  if (device_type_ == DEVICE_GPU && mtype == DEVICE_MEMORY) {
    DeviceContext* device_ctxt = ctx->op_device_context();
    Device* device = static_cast<Device*>(ctx->device());

    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_gpu_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* cpu_allocator = device->GetAllocator(host_alloc_attrs);
    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, tensor.dtype(), tensor.shape());
    auto info = ctx->traced_infos();
    device_ctxt->CopyDeviceTensorToCPU(
        &tensor, "TensorTrace", device, cpu_tensor,
        [this, cpu_tensor, ctx, name, info](const Status& s) {
          ctx->SetStatus(s);
          if (s.ok()) {
            mutex_lock l(tracing_mu_);
            if (info) {
            if (info->traced_tensors) {
            auto name_tensor = info->traced_tensors->add_name_tensors();
            name_tensor->set_name(name);
            cpu_tensor->AsProtoField(name_tensor->mutable_tensor());
            } else {
              LOG(ERROR) << "blaze xla kernel trace error, something wrong1";
            }
            } else {
            LOG(ERROR) << "blaze xla kernel trace error, something wrong";
            }
          }
          delete cpu_tensor;
        });
  } else {
    auto name_tensor = ctx->traced_infos()->traced_tensors->
      mutable_name_tensors()->Add();
    name_tensor->set_name(name);
    tensor.AsProtoField(name_tensor->mutable_tensor());
  }
}

//unsafe func to infer batchsize, just for testing
int BlazeXlaOp::GetBatchSizeUnsafe(OpKernelContext* context) {
  for (int i = 0; i < context->num_inputs(); ++i) {
    const auto& shape = context->input(i).shape();
    if (shape.dims() != 0 && shape.dim_size(0) != 1) {
      return shape.dim_size(0);
    }
  }
  return 1;
}

void BlazeXlaOp::Schedule(OpKernelContext* ctx, const DoneCallback& done, uint64 begin) {
  if (running_counter_ >= counter_per_session_) {
    auto schedule_func = [this, ctx, done, begin] {
      while (running_counter_ >= counter_per_session_) {
        env_->SleepForMicroseconds(kSleepMicroSeconds);
        auto schedule_time = env_->NowNanos();
        if (wait_ns_ > 0) {
          bool time_out = schedule_time - begin > wait_ns_ ? true : false;
          if (TF_PREDICT_FALSE(time_out)) {
            --total_waiting_counter_;
          }
          OP_REQUIRES_ASYNC(ctx, !time_out,
              errors::Internal("blaze wait too long ", schedule_time - begin,
                "counter: ", running_counter_.load()),
              done);
        }
      }
      --total_waiting_counter_;
      ++running_counter_;
      ++total_running_counter_;
      auto running_func = [this, ctx, done, begin] {
        this->Running(ctx, done, begin);
      };
      pool_.Schedule(std::move(running_func));
    };
    ++total_waiting_counter_;
    schedule_pool_.Schedule(std::move(schedule_func));
  } else {
    ++running_counter_;
    ++total_running_counter_;
    auto running_func = [this, ctx, done, begin] {
      this->Running(ctx, done, begin);
    };
    pool_.Schedule(std::move(running_func));
  };
}

void BlazeXlaOp::Running(OpKernelContext* ctx, const DoneCallback& done, uint64 begin) {
  auto schedule_time = env_->NowNanos();
  Status status;
  if (!ctx->traced_infos()) {
    status = predictor_->Compute(ctx);
  } else {
    auto start_ns = env_->NowNanos();
    status = predictor_->Compute(ctx);
    auto end_ns = env_->NowNanos();
    if (ctx->traced_infos()->enable_prof_stats) {
      ctx->traced_infos()->prof_stats->blaze_latency_ms = ((end_ns - start_ns) / 1000000.0f);
      ctx->traced_infos()->prof_stats->blaze_wait_ms = ((start_ns - begin) / 1000000.0f);
      auto output = ctx->mutable_output(0);
      if (output) {
        ctx->traced_infos()->prof_stats->batch_size = output->dim_size(0);
        // check memory
        if (device_type_ != DEVICE_GPU) {
          //check type, only support fp32 now
          if (output->dtype() == DT_FLOAT) {
            bool is_nan = false;
            int nan_counter = 0;
            auto fp32_v = output->flat<float>();
            //2dim will be ok
            int index = 0;
            auto inner_size = output->NumElements() / output->dim_size(0);
            for (int i = 0; i < output->dim_size(0); ++i) {
              for (int j = 0; j < inner_size; ++j) {
                if (std::isnan(fp32_v(index + j))) {
                  ++nan_counter;
                  is_nan = true;
                  break;
                }
              }
              index += inner_size;
            }
            ctx->traced_infos()->prof_stats->blaze_nan = is_nan ? 1 : 0;
            ctx->traced_infos()->prof_stats->blaze_nan_counter = nan_counter;
          }
        }
      }
      ctx->traced_infos()->prof_stats->blaze_running_counter += total_running_counter_;
      ctx->traced_infos()->prof_stats->blaze_waiting_counter += total_waiting_counter_;
    }

    if (ctx->traced_infos()->enable_trace_tensors) {
      TraceTensors(ctx);
    }
  }
  --running_counter_;
  --total_running_counter_;
  OP_REQUIRES_ASYNC(ctx, status.ok(), status, done);
  done();
}

REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_CPU), BlazeXlaOp);
REGISTER_KERNEL_BUILDER(Name("BlazeXlaOp").Device(DEVICE_GPU), BlazeXlaOp);
}  // namespace tensorflow
