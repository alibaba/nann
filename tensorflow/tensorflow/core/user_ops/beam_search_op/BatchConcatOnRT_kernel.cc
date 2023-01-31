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



REGISTER_OP("BatchConcatOnRT")
    .Input("left_values: T")
    .Input("left_row_splits: int64")
    .Input("right_values: T")
    .Input("right_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {half, float, double, int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle left_values, left_row_splits, right_values, right_row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &left_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &left_row_splits));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &right_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &right_row_splits));
      c->set_output(0, c->MakeShape( {c->Value(c->Dim(left_values, 0)) + 
                                      c->Value(c->Dim(right_values, 0))} ));
      if (c->Value(c->Dim(left_row_splits, 0)) == 1)
        c->set_output(1, c->input(3));
      else 
        c->set_output(1, c->input(1));
      return Status::OK();
    });

template <typename T>
class BatchConcatOnRT: public OpKernel {
 public:
  explicit BatchConcatOnRT(OpKernelConstruction* context) : OpKernel(context) {
  }
  void Compute(OpKernelContext* context) override {
    const auto left_values = context->input(0).vec<T>();
    const auto left_row_splits = context->input(1).vec<int64>();

    const auto right_values = context->input(2).vec<T>();
    const auto right_row_splits = context->input(3).vec<int64>();

    int valid = ValidateRaggedTensor<T>(left_values, left_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input0 left, code: ", valid));
    valid = ValidateRaggedTensor<T>(right_values, right_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 right, code: ", valid));

    //handle void inputs
    if (left_row_splits.dimension(0) == 1){
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void input0 "
        <<left_row_splits.dimension(0)<<":"<<right_row_splits.dimension(0);
      context->set_output(0, context->input(2));
      context->set_output(1, context->input(3));
      return;
    } else if (right_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void input1 "
        <<left_row_splits.dimension(0)<<":"<<right_row_splits.dimension(0);
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(1));
      return;
    }

    OP_REQUIRES(context, left_row_splits.dimension(0) == right_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        left_row_splits.dimension(0) ,"!=", right_row_splits.dimension(0)));

    int num_groups = left_row_splits.dimension(0) - 1;

    Tensor *ret_values_tensor;
    Tensor *ret_row_splits_tensor;

    OP_REQUIRES_OK(context, context->allocate_output(0, {left_values.dimension(0) + right_values.dimension(0)},
                                                     &ret_values_tensor));

    OP_REQUIRES_OK(context, context->allocate_output(1, {left_row_splits.dimension(0)},
                                                     &ret_row_splits_tensor));

    auto ret_values = ret_values_tensor->vec<T>();
    auto ret_row_splits = ret_row_splits_tensor->vec<int64>();

    ret_row_splits(0) = 0;

    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; i++) {
        int left_begin = left_row_splits(i);
        int left_end = left_row_splits(i+1);
        int right_begin = right_row_splits(i);
        int right_end = right_row_splits(i+1);
        std::copy(left_values.data()+left_begin, left_values.data()+left_end, ret_values.data()+right_begin+left_begin);
        std::copy(right_values.data()+right_begin, right_values.data()+right_end, ret_values.data()+right_begin+left_end);
        ret_row_splits(i+1) = left_end + right_end;
      }
    };
    Move(0, num_groups);
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchConcatOnRT").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BatchConcatOnRT<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(Eigen::half);
