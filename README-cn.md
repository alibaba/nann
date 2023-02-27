[English Version](README.md)

## 项目简介

NANN (**N**eural **A**pproximate **N**earest **N**eighbor Search，又名“二向箔”)是一个基于纯TensorFlow的灵活、高性能的大规模检索框架。


## 背景
NANN是一种任意神经网络相似性度量下的近似近邻检索算法，于2021年在阿里巴巴内被提出并进行了深度优化，后在淘宝展示广告、神马搜索等众多业务上得到了广泛的应用。
NANN对GPU和CPU上的推理性能进行了深度优化，使其能服务于搜索、推荐、广告等对精度和性能有极高要求的场景。
此外，NANN基于原生TensorFlow，易于使用和部署。

## 主要优点
简单地说，NANN主要优点可以分为模型训练、性能优化和用户友好性三方面。以下逐一介绍：

### 模型训练

- 任意复杂模型：模型训练与索引构建解耦，因此对模型结构几乎没有任何限制，也避免了索引和模型训练绑定带来的高额训练负担。
- 对抗训练：我们采用对抗训练来保证复杂模型下的优越检索性能。

### 性能优化

- 高效检索：我们使用 TensorFlow Custom Ops 实现了HNSW 检索过程。就在线检索而言，重写后的 HNSW 检索比 Faiss 原生版本更加高效。
- 运行时优化：我们支持 GPU Multi-Streaming with Multi-Contexts，这极大地增强了并行性。
- 编译优化：我们支持 XLA 并加速了其JIT 过程； 此外，我们还将 XLA 应用到了大规模检索场景，其中batch-size始终是动态的。
- 图级优化：我们针对推荐、搜索和广告领域中常用的一些常见的模型结构，基于 TensorFlow Grappler 提供了一些图级优化。

### 用户友好性

- 原生TensorFlow：NANN 的后端服务和前端实现完全基于 TensorFlow 生态系统。
- 模型推理和检索解耦：训练-检索解耦使得用户能专注优化深度模型，无需额外考虑检索流程。
- 性能测试：我们提供了一个简单的基准测试工具，可用于分析延迟、吞吐量等推理性能。

## 安装

### 代码准备

```bash
git clone git@github.com:alibaba/nann.git 
```

### 进入Docker

`docker pull alinann/nann_devel:10.2-cudnn7-devel-ubuntu18.04`

| tag | TensorFlow | Python | CUDA | OS | Bazel |
| --- | --- | --- | --- | --- | --- |
| 10.2-cudnn7-devel-ubuntu18.0 | 1.15.5 | 3.7.4 | 10.2 | Ubuntu18.04 | 0.24.1 |

该docker镜像基于 `nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04` 构建，在`/opt/conda`路径下安装了TensorFlow runtime和编译的必要依赖，也包含`Faiss`等相关依赖包。

```bash
sudo yum -y install systemd-devel systemd-libs libseccomp device-mapper-libs
sudo mknod /dev/nvidia-modeset c 195 254

DOCKER_PATH=alinann/nann_devel:10.2-cudnn7-devel-ubuntu18.04
## pay attention to the nvidia-driver version, here is 460.73.01
sudo docker run -ti  --net=host --volume $HOME:$HOME -w $HOME  --volume=nvidia_driver_460.73.01:/usr/local/nvidia:ro --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --device=/dev/nvidia0 --device=/dev/nvidia1 --name=tf_whl_open_source_nann $DOCKER_PATH /bin/bash
```

### 从源码编译Tensorflow
```bash
cd tensorflow

./configure # only cuda support is needed

## build whl package for python frontend
bazel build --copt=-mavx2  -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -DGOOGLE_CUDA=1 //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

## compile tf lib for c++ backend and benchmarking
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=cuda //tensorflow:libtensorflow_framework.so
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=cuda //tensorflow:libtensorflow_cc.so

## install python frontend
pip install /tmp/tensorflow_pkg/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl


# build blaze-benchmark, depend on Tensorflow so files
cd blaze-benchmark
./Build.sh

```
## 模型训练和部署

以下内容对NANN算法的训练、索引结构构建、效果及在线性能评测、部署等流程提供了详细的demo。

### demo数据集准备

本demo基于 [UserBehavior](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) 数据集。
```bash
cd NANN_impls;
export PYTHONPATH=${PYTHONPATH}:$(pwd);
# 将数据集转为tfrecord格式，大约需要10小时
python nann/data_provider/convert_UB_to_tfrecord.py -i path/to/UserBehavior.csv -o ./data
```
### 模型训练

我们基于 [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)实现了单机多卡并行训练。

```bash
#The eps controls the scale for Fast Gradient Signed Method (FGSM) attack. 
#num_neg is the number of negative samples for every positive sample.
eps=3e-5
batch_size=800
num_neg=200
output_root=$(pwd)/output/adv${eps}_bs${batch_size}_neg${num_neg}
python main.py --job-type train --adv-eps ${eps} --num-neg ${num_neg} --batch-size ${batch_size} --output-root ${output_root}
```

### Target embedding 提取

模型训练完毕后，我们可以提取Target embedding用于构建HNSW索引和在线部署。 `item_ids.npy, item_embs.npy`  将保存在 `${output_root}/embeddings` 目录下。
```bash
python main.py --job-type extract_feature --output-root ${output_root}
```
### 索引构建

基于上述target embedding构建HNSW索引，保存在 `${output_root}/index` 目录下。

```
python nann/delivery/build_hnsw_index.py -i ${output_root}/embeddings/item_embs.npy -o ${output_root}/index
```
### 离线评测

测试HNSW检索召回率：
```bash
python main.py --job-type test --output-root ${output_root}
```

测试全量打分召回率，用于评估模型打分召回的天花板：
```
python main.py --job-type test_all --output-root ${output_root}
```

### 用于在线模型部署的模型重写

本阶段生成几个 [GraphDef](https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphDef) protobufs 用于在线部署：

- exec.pb (二值格式)/ exec.pbtxt (文本格式)：储存模型打分和检索流程。
- frozen_graph.pb：储存模型打分过程，包含模型结构和模型参数。该文件将在`exec.pb`中以 `tf.blaze_xla_op` （我们实现的一个custom op）的形式被调用。

```bash
cd NANN_impls

export PYTHONPATH=$PYTHONPATH:nann

# export the modified ckpt for online inference, 
python main.py --job-type export --output-root ${output_root}

# generate frozen_graph.pb (freeze model and small minor changes).
python nann/delivery/convert_meta.py --model_dir ${output_root}/export --output_dir ${output_root}/export/converted_model

# generate exec.pb. the outer graph (dense part + retrieval) for online inference
# generate exec.meta.pb, only need the signature_def which contains the tensor names of model I/O
python nann/delivery/build_opt_graph.py --model-dir ${output_root}/export --index-dir ${output_root}/index --item-embs-dir ${output_root}/embeddings --output-dir ${output_root}/export
```

可通过以下脚本来验证上述生成文件的正确性：
```bash
## exec.meta.pb is only for signature_def, which is used to feed and fetch model I/O
python nann/delivery/NANN_inference_demo.py --graph-file ${output_root}/export/exec.pb --meta-path ${output_root}/export/exec.meta.pb
```
### 性能评测
我们提供了一个简单的性能评测工具`blaze-benchmark`用于分析推理性能：

- `gen_runmeta.py`：用于生成输入模型的mock数据。
- `gen_benchmark_conf.py`：用于生成性能评测所需的配置文件。

使用方式如下：

```bash
cd NANN_impls
nann_impl_dir=`pwd`

cd nann/benchmark
## gen mock data for benchmarking
python gen_runmeta.py --output-dir ${output_root}/export/

## gen benchmark conf

python gen_benchmark_conf.py --model-dir ${output_root}/export

## open nvidia mps for acceleration, https://docs.nvidia.com/deploy/mps/index.html

/usr/local/nvidia/bin/nvidia-cuda-mps-control -d

## benchmarking
## TF_XLA_CUBIN_CACHE_DIR is the env var for CUBIN cache
## TF_XLA_PTX_CACHE_DIR is the env var for PTX and HLO cache. 
## the caches are used for XLA acceleration and reused for the next warmup.

cd ${output_root}/export

TF_CPP_MIN_VLOG_LEVEL=0 TF_XLA_CUBIN_CACHE_DIR=./cubin_cache TF_XLA_PTX_CACHE_DIR=./xla_cache `dirname $nann_impl_dir`/blaze-benchmark/build/benchmark/benchmark benchmark_conf
```

### 模型部署
我们基于 TensorFlow Serving 提供了一个简单的demo。

将模型导出为SavedModel格式：
```bash
python nann/delivery/pb_to_saved_model.py --model-path ${output_root}/export/exec.pb --meta-path ${output_root}/export/exec.meta.pb --export-dir ${output_root}/export/nann/1
```

基于Docker起模型服务进程：
```bash
docker run -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source=${output_root}/export/nann,target=/models/nann \
  -e MODEL_NAME=nann -t alinann/nann_serving
```

简单的冒烟测试代码如下（`tensorflow-serving-api==1.15.0`）：
```python
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import grpc
import tensorflow as tf

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
grpc_request = predict_pb2.PredictRequest()
grpc_request.model_spec.name = "nann"
grpc_request.model_spec.signature_name = 'serving_default'
comm_seq = np.zeros((1, 3200), dtype=np.float16)
level_topn = np.array([100, 200, 200, 200, 200, 200], dtype=np.int32)
grpc_request.inputs['comm_seq'].CopyFrom(tf.make_tensor_proto(comm_seq, shape=comm_seq.shape))
grpc_request.inputs['level_topn'].CopyFrom(tf.make_tensor_proto(level_topn, shape=level_topn.shape))
result = stub.Predict(grpc_request,10)
print(result.outputs['top_k'])
```

## 参与共建

- 请参考 [贡献指南](CONTRIBUTING.md)。
- 如果本工作对您有帮助，请考虑引用 [NANN](https://dl.acm.org/doi/abs/10.1145/3511808.3557098#sec-ref)。
```
article{chen2022approximate,
  title={Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation},
  author={Chen, Rihan and Liu, Bin and Zhu, Han and Wang, Yaoxuan and Li, Qi and Ma, Buting and Hua, Qingbo and Jiang, Jun and Xu, Yunlong and Deng, Hongbo and others},
  journal={arXiv preprint arXiv:2202.10226},
  year={2022}
}
```


## 行为准则

阿里巴巴已经制定了一套行为准则，我们希望项目参与者能够遵守。
详情请参阅[阿里巴巴开源行为准则](CODE_OF_CONDUCT_zh.md) ([English Version](CODE_OF_CONDUCT.md)).