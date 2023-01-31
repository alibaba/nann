#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <functional>

using namespace tensorflow;

template <typename T>
int ValidateRaggedTensor(const typename TTypes<T>::ConstVec& values, 
                         const typename TTypes<int64>::ConstVec& row_splits) {
  if (row_splits.dimension(0) == 0) return 1;
  if (row_splits(0) != 0) return 2;
  if (row_splits(row_splits.dimension(0)-1) != values.dimension(0)) return 3;
  return 0;
}


REGISTER_OP("BatchGatherOnRT")
    .Input("params_values: T")
    .Input("params_row_splits: int64")
    .Input("indices_values: int64")
    .Input("indices_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle params_values, params_row_splits, indices_values, indices_row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &params_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &params_row_splits));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &indices_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &indices_row_splits));
      if (c->Value(c->Dim(params_row_splits, 0)) == 1 ||
          c->Value(c->Dim(indices_row_splits, 0)) == 1 ) {
        c->set_output(0, c->MakeShape({0}));
        c->set_output(1, c->MakeShape({1}));
        return Status::OK();
      }
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      return Status::OK();
    });

template <typename T>
class BatchGatherOnRT: public OpKernel {
 public:
  explicit BatchGatherOnRT(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto params_values = context->input(0).vec<T>();
    const auto params_row_splits = context->input(1).vec<int64>();

    const auto indices_values = context->input(2).vec<int64>();
    const auto indices_row_splits = context->input(3).vec<int64>();

    int valid = ValidateRaggedTensor<T>(params_values, params_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input0 params, code: ", valid));
    valid = ValidateRaggedTensor<int64>(indices_values, indices_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 indices, code: ", valid));

    //handle void inputs
    if (params_row_splits.dimension(0) == 1 || indices_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void inputs "
        <<params_row_splits.dimension(0)<<":"<<indices_row_splits.dimension(0);
      Tensor* t;
      OP_REQUIRES_OK(context, context->allocate_output(0, {0}, &t));
      OP_REQUIRES_OK(context, context->allocate_output(1, {1}, &t));
      (t->vec<int64>())(0) = 0;
      return;
    }

    OP_REQUIRES(context, params_row_splits.dimension(0) == indices_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        params_row_splits.dimension(0) ,"!=", indices_row_splits.dimension(0)));

    
    int num_groups = indices_row_splits.dimension(0) - 1;
    
    Tensor *ret_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {indices_values.dimension(0)}, 
                                                     &ret_values_tensor));
    auto ret_values = ret_values_tensor->vec<T>();
    context->set_output(1, context->input(3));

    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; i++) {
        for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
          int idx = params_row_splits(i) + indices_values(j);
          ret_values(j) = params_values(idx);
        }
      }
    };
    Move(0, num_groups);

  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchGatherOnRT").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BatchGatherOnRT<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
