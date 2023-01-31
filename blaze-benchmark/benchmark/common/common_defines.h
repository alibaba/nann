#pragma once

#include "benchmark/common/queue.h"

namespace benchmark {

struct PredictRequest {
  int model_idx;
  PredictRequest(int idx) {
    model_idx = idx;
  }
  PredictRequest() {
    model_idx = 0;
  }
};

typedef Queue<PredictRequest> PredictRequestQueue;

}  // namespace benchmark
