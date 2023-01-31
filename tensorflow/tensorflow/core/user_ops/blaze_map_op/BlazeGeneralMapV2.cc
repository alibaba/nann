#include <map>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/numbers.h"

using namespace tensorflow;

REGISTER_OP("BlazeGeneralMapV2")
    .Input("key: Tin")
    .Output("output: Tout")
    .Attr("Tin: {int32, int64} = DT_INT64")
    .Attr("Tout:{int32, int64} = DT_INT32")
    .Attr("keys: list(string) >= 0")
    .Attr("values: list(string) >= 0")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
    c->set_output(0, c->MakeShape({c->UnknownDim()}));
    return Status::OK();
});

template<typename IN, typename OUT>
class BlazeGeneralMapV2 : public OpKernel {
 public:
  explicit BlazeGeneralMapV2(OpKernelConstruction* ctx) : OpKernel(ctx) 
  {
    std::vector<string> keys;
    std::vector<string> values;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keys", &keys));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("values", &values));
    OP_REQUIRES_OK(ctx, InitMap(keys, values));
  }
  void Compute(OpKernelContext* ctx) {
    const auto& input = ctx->input(0).vec<IN>();
    int num_elems = input.dimension(0);
    std::vector<IN> keys(input.data(), input.data()+num_elems);
    std::vector<OUT> values;
    values.reserve(num_elems);

    for (int i = 0; i < keys.size(); ++i) {
      IN key = keys[i]; 
      auto iter = map_.find(key);
      if (iter != map_.end()) {
          values.push_back(map_[key]);
      }
    }

    TensorShape output_shape({values.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<OUT>();
    std::copy(values.begin(), values.end(), output.data());
  }

 private:
  Status InitMap(const std::vector<string>& keys, const std::vector<string>& values) {
    if (keys.size() != values.size()) {
      return errors::Internal("key size ", keys.size(), " not euqal to value size ",
                              values.size());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      IN in;
      StringPiece sp(keys[i]);
      if (!strings::SafeStringToNumeric<IN>(sp, &in)) {
        return errors::Internal("Parse ", keys[i], " failed" );
      }
      StringPiece outsp(values[i]);
      OUT out;
      if (!strings::SafeStringToNumeric<OUT>(outsp, &out)) {
        return errors::Internal("Parse ", values[i], " failed");
      }
      map_[in] = out;
    }
    return Status::OK();
  }
 private:
  std::map<IN, OUT> map_;
  OUT default_val_;
};


#define REGISTER_KERNEL_GENERAL_MAP(in_type, out_type)       \
  REGISTER_KERNEL_BUILDER(Name("BlazeGeneralMapV2")               \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<in_type>("Tin")       \
                              .TypeConstraint<out_type>("Tout"), \
                          BlazeGeneralMapV2<in_type, out_type>);
REGISTER_KERNEL_GENERAL_MAP(int64, int32);

