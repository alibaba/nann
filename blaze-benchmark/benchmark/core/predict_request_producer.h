#pragma once

#include "benchmark/common/common_defines.h"
#include "benchmark/core/metrics.h"
#include "benchmark/proto/bench_conf.pb.h"

namespace benchmark {

class PredictRequestProducer {
 public:
  PredictRequestProducer(PredictRequestQueue* predict_queue, Metrics* metrics);

  bool Init(const benchmark::BenchmarkConfig& bench_config, int parallelism);

  void Start();

 protected:
  PredictRequestQueue* predict_queue_;
  Metrics* metrics_;
  std::vector<std::vector<PredictRequest>> all_request_;
  std::vector<int> qps_distribution_;
  int total_qps_;
};

}  // namespace benchmark
