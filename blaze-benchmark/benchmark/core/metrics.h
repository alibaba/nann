#pragma once

#include <mutex>
#include "cppmetrics/cppmetrics.h"

namespace benchmark {

class Metrics {
 public:
  bool Init();

  void Start();

  void UpdateThroughput(const std::string& name);
  void UpdateLatency(const std::string& name, int latency);
  void UpdateBatchsize(const std::string& name, int batchsize);
  void UpdateFailures(const std::string& name);
  void UpdateGetPredictorFailures(const std::string& name);

  void Stop();

  bool IsStopped();

  std::mutex mu_;
  std::unique_ptr<cppmetrics::core::ConsoleReporter> reporter_;
  std::map<std::string, cppmetrics::core::HistogramPtr> latencies_;
  std::map<std::string, cppmetrics::core::HistogramPtr> batchsizes_;
  std::map<std::string, cppmetrics::core::MeterPtr> throughputs_;
  std::map<std::string, cppmetrics::core::MeterPtr> failures_;
  std::map<std::string, cppmetrics::core::MeterPtr> get_predictor_failures_;
  bool stopped_ = false;
};

}  // namespace benchmark
