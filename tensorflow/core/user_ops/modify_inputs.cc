#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

REGISTER_OP("ModifyInput")
    .Input("input: Ref(int32)")
    .Output("output: Ref(int32)")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape);

class ModifyInputOp : public tensorflow::OpKernel {
 public:
  explicit ModifyInputOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    auto input = context->mutable_input(0, false);
    auto value = input.flat<tensorflow::int32>();
    value(0) += 1;
    context->forward_ref_input_to_ref_output(0, 0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ModifyInput").Device(tensorflow::DEVICE_CPU), ModifyInputOp);

