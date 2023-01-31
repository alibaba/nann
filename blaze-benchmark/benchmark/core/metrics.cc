#include "benchmark/core/metrics.h"

namespace benchmark {

bool Metrics::Init() {
  auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
  if (!reporter_) {
    reporter_.reset(new cppmetrics::core::ConsoleReporter(
        registry, std::cout, boost::chrono::seconds(1)));
  }
  return true;
}

void Metrics::Start() { reporter_->start(boost::chrono::seconds(3)); }

void Metrics::Stop() {
  std::unique_lock <std::mutex> l(mu_);
  if (!stopped_) {
    stopped_ = true;
    reporter_->stop();
  }
}

void Metrics::UpdateThroughput(const std::string& name) {
  cppmetrics::core::MeterPtr thr;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (throughputs_.find(name) == throughputs_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      throughputs_[name] = registry->meter(name + "_throughput");
    }
    thr = throughputs_[name];
  }
  thr->mark();
}

void Metrics::UpdateLatency(const std::string& name, int latency) {
  cppmetrics::core::HistogramPtr lat;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (latencies_.find(name) == latencies_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      latencies_[name] = registry->histogram(name + "_latency");
    }
    lat = latencies_[name];
  }
  lat->update(latency);
}

void Metrics::UpdateBatchsize(const std::string& name, int batchsize) {
  cppmetrics::core::HistogramPtr bs;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (batchsizes_.find(name) == batchsizes_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      batchsizes_[name] = registry->histogram(name + "_batchsize");
    }
    bs = batchsizes_[name];
  }
  bs->update(batchsize);
}

void Metrics::UpdateGetPredictorFailures(const std::string& name) {
  cppmetrics::core::MeterPtr fail;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (get_predictor_failures_.find(name) == get_predictor_failures_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      get_predictor_failures_[name] = registry->meter(name + "_get_predictor_failures");
    }
    fail = get_predictor_failures_[name];
  }
  fail->mark();
}

void Metrics::UpdateFailures(const std::string& name) {
  cppmetrics::core::MeterPtr fail;
  {
    std::unique_lock <std::mutex> l(mu_);
    if (failures_.find(name) == failures_.end()) {
      auto registry = cppmetrics::core::MetricRegistry::DEFAULT_REGISTRY();
      failures_[name] = registry->meter(name + "_failures");
    }
    fail = failures_[name];
  }
  fail->mark();
}

bool Metrics::IsStopped() {
  std::unique_lock <std::mutex> l(mu_);
  return stopped_;
}

}  // namespace benchmark
