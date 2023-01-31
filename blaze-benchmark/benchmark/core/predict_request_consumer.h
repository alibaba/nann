#pragma once

#include "benchmark/common/common_defines.h"
#include "benchmark/core/model.h"
#include "benchmark/core/metrics.h"

namespace benchmark {

class PredictRequestConsumer {
 public:
  PredictRequestConsumer(ModelSelector* model_selector,
                         PredictRequestQueue* predict_queue,
                         Metrics* metrics, int max_queue_size);

  void Start();

 protected:
  bool PredictImpl(Model::PredictContext* predict_context, int* batchsize);

 private:
  ModelSelector* model_selector_;
  PredictRequestQueue* predict_queue_;
  Metrics* metrics_;
  int max_queue_size_;
};

}  // namespace benchmark
