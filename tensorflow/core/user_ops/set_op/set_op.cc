#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <math.h>
#include "tensorflow/core/platform/fingerprint.h"



using namespace tensorflow;

/*
REGISTER_OP("SetUnion")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle a_values, a_row_splits, b_values, b_row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &a_row_splits));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &b_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &b_row_splits));
      if (c->Value(c->Dim(a_row_splits, 0)) == 1 ) {
        c->set_output(0, c->input(2));
        c->set_output(1, c->input(3));
        return Status::OK();
      } else if (c->Value(c->Dim(b_row_splits, 0)) == 1 ) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
      }
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("SetIntersection")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle a_values, a_row_splits, b_values, b_row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &a_row_splits));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &b_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &b_row_splits));
      if (c->Value(c->Dim(a_row_splits, 0)) == 1  || c->Value(c->Dim(b_row_splits, 0)) == 1) {
        c->set_output(0, c->MakeShape({0}));
        c->set_output(1, c->MakeShape({1}));
        return Status::OK();
      } 
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("SetDifference")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle a_values, a_row_splits, b_values, b_row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &a_row_splits));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &b_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &b_row_splits));
      if (c->Value(c->Dim(a_row_splits, 0)) == 1 ) {
        c->set_output(0, c->MakeShape({0}));
        c->set_output(1, c->MakeShape({1}));
        return Status::OK();
      } else if (c->Value(c->Dim(b_row_splits, 0)) == 1 ) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
      }
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });
*/

template <typename T>
int ValidateRaggedTensor(const typename TTypes<T>::ConstVec& values, 
                         const typename TTypes<int64>::ConstVec& row_splits) {
  if (row_splits.dimension(0) == 0) return 1;
  if (row_splits(0) != 0) return 2;
  if (row_splits(row_splits.dimension(0)-1) != values.dimension(0)) return 3;
  return 0;
}

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
class SetUnion: public OpKernel {
 public:
  explicit SetUnion(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    int valid = ValidateRaggedTensor<T>(a_values, a_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input0 a, code: ", valid));
    valid = ValidateRaggedTensor<T>(b_values, b_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 b, code: ", valid));

    //handle void inputs
    if (a_row_splits.dimension(0) == 1){
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void input0 "
        <<a_row_splits.dimension(0)<<":"<<b_row_splits.dimension(0);
      context->set_output(0, context->input(2));
      context->set_output(1, context->input(3));
      return;
    } else if (b_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void input1 "
        <<a_row_splits.dimension(0)<<":"<<b_row_splits.dimension(0);
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(1));
      return;
    }

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Union = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        sets[i].insert(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1));
        sets[i].insert(b_values.data()+b_row_splits(i), b_values.data()+b_row_splits(i+1));
      }
    };
    Union(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

template <typename T>
class SetIntersection: public OpKernel {
 public:
  explicit SetIntersection(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    int valid = ValidateRaggedTensor<T>(a_values, a_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input0 a, code: ", valid));
    valid = ValidateRaggedTensor<T>(b_values, b_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 b, code: ", valid));

    if (a_row_splits.dimension(0) == 1 || b_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void inputs "
        <<a_row_splits.dimension(0)<<":"<<b_row_splits.dimension(0);
      Tensor* t;
      OP_REQUIRES_OK(context, context->allocate_output(0, {0}, &t));
      OP_REQUIRES_OK(context, context->allocate_output(1, {1}, &t));
      (t->vec<int64>())(0) = 0;
      return;
    }

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Intersect = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        std::unordered_set<T> temp(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1),
                                   a_row_splits(i+1) - a_row_splits(i));
        auto& set = sets[i];
        set.reserve(b_row_splits(i+1) - b_row_splits(i));
        for (int j = b_row_splits(i); j < b_row_splits(i+1); ++j) {
          if (temp.count(b_values(j)))
            set.insert(b_values(j));
        }
      }
    };
    Intersect(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

template <typename T>
class SetDifference: public OpKernel {
 public:
  explicit SetDifference(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    int valid = ValidateRaggedTensor<T>(a_values, a_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input0 a, code: ", valid));
    valid = ValidateRaggedTensor<T>(b_values, b_row_splits);
    OP_REQUIRES(context, valid == 0, 
                errors::InvalidArgument("Invalid RaggedTensor input1 b, code: ", valid));

    //handle void inputs
    if (a_row_splits.dimension(0) == 1){
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void inputs "
        <<a_row_splits.dimension(0)<<":"<<b_row_splits.dimension(0);
      Tensor* t;
      OP_REQUIRES_OK(context, context->allocate_output(0, {0}, &t));
      OP_REQUIRES_OK(context, context->allocate_output(1, {1}, &t));
      (t->vec<int64>())(0) = 0;
      return;
    } else if (b_row_splits.dimension(0) == 1) {
      if (VLOG_IS_ON(1)) LOG(WARNING) << this->name() << " void inputs "
        <<a_row_splits.dimension(0)<<":"<<b_row_splits.dimension(0);
      context->set_output(0, context->input(0));
      context->set_output(1, context->input(1));
      return;
    }

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Differ = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        auto& set = sets[i];
        set.insert(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1));
        for (int j = b_row_splits(i); j < b_row_splits(i+1); ++j) {
          set.erase(b_values(j));
        }
      }
    };
    Differ(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("SetUnion").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetUnion<T>); \
  REGISTER_KERNEL_BUILDER(Name("SetIntersection").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetIntersection<T>); \
  REGISTER_KERNEL_BUILDER(Name("SetDifference").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetDifference<T>); 

//REGISTER_CPU(int32);
//REGISTER_CPU(int64);

