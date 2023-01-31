#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


REGISTER_OP("GetChildren")
    .Input("nodes: int32")
    .Input("tree: int32")
    .Output("children: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("GetParents")
    .Input("nodes: int32")
    .Input("tree: int32")
    .Output("parents: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("FirstLevel")
    .Input("tree: int32")
    .Output("first_level: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });


// ===============================================
// ParentIndicator
// an indicator showing the parent of each node, must be monotonically increasing
// e.g. a tree such as
// 0
// 1      2    3   4
// 5 6 7  8 9  10  11 12
// , gives  -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4
// ================================================

const int avg_node_degree = 1024;

class GetChildren_ParentIndicator: public OpKernel {
 public:
  explicit GetChildren_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto& nodes = context->input(0).vec<int>();
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    int tree_size = tree.dimension(0);
    std::vector<int> parents(nodes.data(), nodes.data()+num_nodes);
    std::sort(parents.begin(), parents.end());

    std::vector<int> children;
    children.reserve(num_nodes*avg_node_degree);

    int child = 0;
    for (int i = 0; i < parents.size(); ++i) {
      int parent = parents[i];
      OP_REQUIRES(context, 0 <= parent && parent < tree_size,
                  errors::InvalidArgument("tree_size is ", tree_size, ", but input node is ", parent));
      while (tree(child) <= parent) {
        if (tree(child) == parent) {
          children.push_back(child);
        }
        child++;
      }
    }
 
    //Allocate Output
    TensorShape output_shape({children.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    std::copy(children.begin(), children.end(), output.data());
  };
};

class GetParents_ParentIndicator: public OpKernel {
 public:
  explicit GetParents_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose parents will be returned
    const auto& nodes = context->input(0).vec<int>();
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    int tree_size = tree.dimension(0);

    //Allocate Output
    TensorShape output_shape({num_nodes});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    for (int i = 0; i < num_nodes; ++i) {
      int node = nodes(i);
      OP_REQUIRES(context, 0 <= node && node < tree_size,
                  errors::InvalidArgument("tree_size is ", tree_size, ", but input node is ", node));
      int parent = tree(node);
      OP_REQUIRES(context, parent >= 0,
                  errors::InvalidArgument("the node ", node, " is already root."));
      output(i) = parent;
    }
  };
};

class FirstLevel_ParentIndicator: public OpKernel {
 public:
  explicit FirstLevel_ParentIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto& tree = context->input(0).vec<int>();

    std::vector<int> first_level;
    first_level.reserve(avg_node_degree);

    for (int i = 0; i < tree.dimension(0); ++i) {
      if (tree(i) < 0) {
        first_level.push_back(i);
      } else {
        break;
      }
    }
 
    //Allocate Output
    TensorShape output_shape({first_level.size()});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    std::copy(first_level.begin(), first_level.end(), output.data());
  };
};


// ================================
// RangeIndicator
// an indicator showing the splits of level order traversal of a complete tree
// e.g. a tree such as
// 0
// 1      2    3   4      
// 5 6 7  8 9  10  11 12
// , whose level order traversal is  0 | 1 2 3 4 | 5 6 7 ; 8 9 ; 10 ; 11 12 | ...
// will be represented as  1, 5, 8, 10, 11, 13 ..., such that [a_i, a_i+i) is the children of i-th node
// it also represents a bunch of trees (a forest, or start from mid layer), e.g.
// 0         1         2
// 3     4   5         6      7   8
// 9 10  11  12 13 14  15 16  17  18 19 20
// represents as  3, 5, 6, 9, 11, 12, 15, 17, 18, 21
// thous we know that the "first level" (roots of trees) is [0,3)=0, 1, 2, since the first element is 3.
// =================================


class GetChildren_RangeIndicator: public OpKernel {
 public:
  explicit GetChildren_RangeIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto& nodes = context->input(0).vec<int>();
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    int num_ranges = tree.dimension(0)-1;
    int tree_size = tree(num_ranges);

    int num_children = 0;
    for (int i = 0; i < num_nodes; ++i) {
      int node = nodes(i);
      OP_REQUIRES(context, 0 <= node && node < tree_size, 
                  errors::InvalidArgument("tree_size is ", tree_size, ", but input node is ", node ));
      if (node >= num_ranges) {
        continue;
      }
      num_children += tree(node+1) - tree(node);
    }

    //Allocate Output
    TensorShape output_shape({num_children});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    int idx = 0;
    for (int i = 0; i < num_nodes; ++i) {
      int node = nodes(i);
      if (node >= num_ranges) {
        continue;
      }
      for (int j = tree(node); j < tree(node+1); ++j) {
        output(idx) = j;
        ++idx;
      }
    }
    OP_REQUIRES(context, idx == num_children,
                errors::InvalidArgument("number of found children mismatch the pre counted number"));
  };
};

class GetParents_RangeIndicator: public OpKernel {
 public:
  explicit GetParents_RangeIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto& nodes = context->input(0).vec<int>();
    const auto& tree = context->input(1).vec<int>();

    int num_nodes = nodes.dimension(0);
    int num_ranges = tree.dimension(0)-1;
    int tree_size = tree(num_ranges);

    std::vector<int> sorted_idx(num_nodes);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&nodes](int a, int b){return nodes(a) < nodes(b);});

    //Allocate Output
    TensorShape output_shape({num_nodes});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    for (int i = 0; i < num_nodes; ++i) {
      LOG(INFO) << sorted_idx[i] << ":" << nodes(sorted_idx[i]);
    }

    int parent = 0;
    for (int i = 0; i < num_nodes; ++i) {
      int node_idx = sorted_idx[i];
      int node = nodes(node_idx);
      OP_REQUIRES(context, 0 <= node && node < tree_size,
                  errors::InvalidArgument("tree_size is ", tree_size, ", but input node is ", node ));
      while (tree(parent+1) <= node)
        parent += 1;
      int range_begin = tree(parent);
      int range_end = tree(parent+1);
      OP_REQUIRES(context, range_begin <= node && node < range_end,
                  errors::InvalidArgument("Node", node, " is NOT child of Node", parent,
                                          ":range[", range_begin, ",", range_end, "), which could be root already"));
      output(node_idx) = parent;
    }
  };
};

class FirstLevel_RangeIndicator: public OpKernel {
 public:
  explicit FirstLevel_RangeIndicator(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const auto& tree = context->input(0).vec<int>();

    int first_level_size = tree(0);
 
    //Allocate Output
    TensorShape output_shape({first_level_size});
    Tensor *output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->vec<int>();

    for (int i = 0; i < first_level_size; ++i) {
      output(i) = i;
    }
  };
};

REGISTER_KERNEL_BUILDER(Name("GetChildren").Device(DEVICE_CPU), GetChildren_RangeIndicator);
REGISTER_KERNEL_BUILDER(Name("GetParents").Device(DEVICE_CPU), GetParents_RangeIndicator);
REGISTER_KERNEL_BUILDER(Name("FirstLevel").Device(DEVICE_CPU), FirstLevel_RangeIndicator);
