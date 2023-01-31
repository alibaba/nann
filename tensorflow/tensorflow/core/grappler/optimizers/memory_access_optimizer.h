/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_ACCESS_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_ACCESS_OPTIMIZER_H_

#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

using namespace tensorflow::graph_transforms;

namespace tensorflow {
namespace grappler {

namespace {

static const OpTypePattern gather_pattern1 =
       {"BatchMatMulV2|Mul",
         { // input
           {"GatherV2",
             {
               {"*"},
               {"Placeholder|Tile"},
               {"Const"},
             }
           },
           {"*"},
         }
       };
static const OpTypePattern gather_pattern2 =
       {"BatchMatMulV2|Mul",
         { // input
           {"*"},
           {"GatherV2",
             {
               {"*"},
               {"Placeholder|Tile"},
               {"Const"},
             }
           },
         }
       };
static const OpTypePattern gather_pattern3 =
       {"MatMul",
         { // input
           {"GatherV2",
             {
               {"*"},
               {"Placeholder|Tile"},
               {"Const"},
             }
           },
           {"*"},
         }
       };
}  // end namespace

class MemoryAccessOptimizer : public GraphOptimizer {
 public:
  MemoryAccessOptimizer() {}
  ~MemoryAccessOptimizer() override {}

  string name() const override { return "memory_access"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_ACCESS_OPTIMIZER_H_
