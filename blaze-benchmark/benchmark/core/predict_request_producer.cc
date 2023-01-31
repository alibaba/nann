#include "benchmark/core/predict_request_producer.h"

#include <random>
#include <thread>
#include <chrono>

namespace benchmark {

PredictRequestProducer::PredictRequestProducer(benchmark::PredictRequestQueue* predict_queue,
                                               benchmark::Metrics *metrics) {
  predict_queue_ = predict_queue;
  metrics_ = metrics;
  total_qps_ = 0;
}

bool PredictRequestProducer::Init(const benchmark::BenchmarkConfig& bench_config,
                                  int parallelism) {
  int model_count = bench_config.bench_model_config().size();
  all_request_.reserve(model_count);
  for (int idx = 0; idx < model_count; ++idx) {
    // init model input
    const auto& bench_model_config = bench_config.bench_model_config()[idx];
    std::vector<PredictRequest> predict_request_list;
    predict_request_list.emplace_back(idx);
    all_request_.emplace_back(predict_request_list);

    // init multi model qps distribution vector
    int qps = bench_model_config.qps();

    // qps <= 0, bench max qps
    if (qps <= 0) {
      if (model_count > 1) {
        std::cerr << "bad config: qps <= 0 && model count > 1";
        return false;
      }
      total_qps_ = 0;
      qps_distribution_.emplace_back(0);
    } else {
      total_qps_ += qps;
      for (int j = 0; j < qps; ++j) {
        qps_distribution_.emplace_back(idx);
      }
    }
  }
  total_qps_ /= parallelism;
  return true;
}

void PredictRequestProducer::Start() {
  std::vector<size_t> pos;
  pos.resize(all_request_.size(), 0);
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_int_distribution<> dist(0, qps_distribution_.size() - 1);
  while (!metrics_->IsStopped()) {
    if (total_qps_ > 0) {
      auto time1 = std::chrono::high_resolution_clock::now();
      uint64_t interval = (800 * 1000) / total_qps_;
      for (int i = 0; i < total_qps_; ++i) {
        int ids = qps_distribution_[dist(eng)];
        if (pos[ids] == all_request_[ids].size()) {
          pos[ids] = 0;
        }
        predict_queue_->Enqueue(&all_request_[ids][pos[ids]++]);
        std::this_thread::sleep_for(std::chrono::microseconds(interval));
      }
      auto time2 = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
      if (diff <= 1000 * 1000) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000 * 1000 - diff));
      }
    } else {
      if (pos[0] == all_request_[0].size()) {
        pos[0] = 0;
      }
      predict_queue_->Enqueue(&all_request_[0][pos[0]++]);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
}

}  // namespace benchmark
