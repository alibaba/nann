
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace tensorflow;

template <typename T>
int ValidateRaggedTensor(const typename TTypes<T>::ConstVec& values, 
                         const typename TTypes<int64>::ConstVec& row_splits) {
  if (row_splits.dimension(0) == 0) return 1;
  if (row_splits(0) != 0) return 2;
  if (row_splits(row_splits.dimension(0)-1) != values.dimension(0)) return 3;
  return 0;
}


REGISTER_OP("SplitsGather")
    .Input("splits: T")
    .Input("indices_values: int64")
    .Input("indices_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(2));
      return Status::OK();
    });


template <typename T>
class SplitsGather: public OpKernel {
 public:
  explicit SplitsGather(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto splits = context->input(0).vec<T>();

    const auto indices_values = context->input(1).vec<int64>();
    const auto indices_row_splits = context->input(2).vec<int64>();

    int valid = ValidateRaggedTensor<int64>(indices_values, indices_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 indices, code: ", valid));

    OP_REQUIRES(context, splits.dimension(0) == 0 || splits(0) == 0, 
                errors::InvalidArgument("input splits should NOT contain less than ONE element."));
    if (splits.dimension(0) <= 1 || indices_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void inputs "
        <<splits.dimension(0)<<":"<<indices_row_splits.dimension(0);
      Tensor* t;
      OP_REQUIRES_OK(context, context->allocate_output(0, {0}, &t));
      OP_REQUIRES_OK(context, context->allocate_output(1, {1}, &t));
      (t->vec<int64>())(0) = 0;
      return;
    }

    int num_groups = indices_row_splits.dimension(0) - 1;

    Tensor *ret_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {indices_row_splits.dimension(0)}, 
                                                     &ret_row_splits_tensor));
    auto ret_row_splits = ret_row_splits_tensor->vec<int64>();

    int sum = 0;
    ret_row_splits(0) = 0;
    for (int i = 0; i < num_groups; ++i) {
      for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
        int idx = indices_values(j);
        int len = splits(idx+1) - splits(idx);
        sum += len;
      }
      ret_row_splits(i+1) = sum;
    }

    Tensor *ret_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {sum}, &ret_values_tensor));
    auto ret_values = ret_values_tensor->vec<T>();

    // can be parallelize on i
    for (int i = 0; i < num_groups; ++i) {
      int value_idx = ret_row_splits(i);
      for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
        int group_idx = indices_values(j);
        for (int k = splits(group_idx); k < splits(group_idx+1); ++k) {
          ret_values(value_idx) = k;
          value_idx++;
        }
      }
      //assert values_idx == ret_row_splits(i+1)
      OP_REQUIRES(context, value_idx == ret_row_splits(i+1),
                errors::InvalidArgument("Number of collected values does NOT match number in row_splits: ", 
                                        value_idx ,"!=", ret_row_splits(i+1), " @group ", i));
    }
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("SplitsGather").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SplitsGather<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
