## Introduction
NANN is a flexible, high-performance framework for large-scale retrieval problems based on TensorFlow.
## Background
NANN has been deeply cultivated and widely applied since 2021 in Alibaba, which supports many businesses such as Taobao display advertising,  Taobao search advertising, and Shenma search. NANN aims to solve the large-scale retrieval problem by integrating the post-training index with arbitrarily advanced models. Model-based and heuristic methods are provided to ensure that arbitrarily advanced models can still maintain their capability during large-scale retrieval. Also, NANN includes in-depth performance optimizations for GPU and CPU, which guarantee the inference performance. Moreover, NANN lays emphasis on user-friendliness, especially for TensorFlow users. 
## Key features
Briefly speaking, key features of NANN can be categorized into model training, performance optimization, and user-friendliness.
### Model Training

- Arbitrarily Advanced Models. -- Model training is decoupled with index building, which can include arbitrarily advanced models, and meanwhile, avoid costly training budgets.
- Adversarial Attacks  -- Adversarial training is resorted to ensure arbitrarily advanced models can still take effect in large-scale retrieval scenarios. 
### Performance Optimization

- High-Efficient Retrieval -- We reinvent the HNSW search process with TensorFlow Custom Ops. The reinvented HNSW is more efficient than the off-the-shelf version in Faiss in terms of online retrieval.
- Runtime Optimization -- We support GPU Multi-Streaming, which greatly enhances parallelism.
- Compilation Optimization -- We support XLA and accelerate its Just-In-Time (JIT) process; we also extend the XLA application to our large-scale retrieval scenarios where the number of samples that are propagated through the networks is always dynamic.
- Graph Level Optimization -- We identify some common model structures in the fields of recommendation, search, and advertising and provide graph-level optimizations based on TensorFlow Grappler.
### User-Friendliness

- TensorFlow Only --   NANN is entirely based on TensorFlow ecosystem for both the backend and the frontend.  
- Model Inference and Retrieval Decoupling -- Post-training index enable users to focus on deep models. No extra attention needed to be paid to the retrieval process.
- Benchmarking --  We provide a handy benchmarking tool that can be used to profile the inference performance in terms of latency, throughput, etc.  
## Installation
### Code Preparation
```bash
git clone git@gitlab.alibaba-inc.com:alimama-displayads-match-public/NANN.git --recursive
```
### Enter Docker
`docker pull rihan19920210/nann_devel:10.2-cudnn7-devel-ubuntu18.04`

| tag | TensorFlow | Python | CUDA | OS | Bazel |
| --- | --- | --- | --- | --- | --- |
| 10.2-cudnn7-devel-ubuntu18.0 | 1.15.5 | 3.7.4 | 10.2 | Ubuntu18.04 | 0.24.1 |

The docker is built from nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 with necessary dependencies for TensorFlow runtime and compilation. Besides, we also install Faiss and its dependencies in the conda environment located in /opt/conda. 
```bash
sudo yum -y install systemd-devel systemd-libs libseccomp device-mapper-libs
sudo mknod /dev/nvidia-modeset c 195 254

DOCKER_PATH=rihan19920210/nann_devel:10.2-cudnn7-devel-ubuntu18.04
## pay attention to the nvidia-driver version, here is 460.73.01
sudo docker run -ti  --net=host --volume $HOME:$HOME -w $HOME  --volume=nvidia_driver_460.73.01:/usr/local/nvidia:ro --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --device=/dev/nvidia0 --device=/dev/nvidia1 --name=tf_whl_open_source_nann $DOCKER_PATH /bin/bash
```
### Build From Source
```bash
cd tensorflow

git checkout blaze/open_source_nann

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
git checkout open_source_nann
./Build.sh

```
## Usage
A Demo from Model Training to Model Benchmarking
### Data Preparation
We use [UserBehavior](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) dataset for demonstration.
```bash
# convert data to tfrecord, takes around 10h.
python convert_UB_to_tfrecord.py -i path/to/UserBehavior.csv -o ./data

```
### Model Training
Model training is implemented with [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy). 
```bash
#The eps controls the scale for Fast Gradient Signed Method (FGSM) attack. 
#num_neg is the number of negative samples for every positive sample.
eps=3e-5
batch_size=800
num_neg=200
output_root=$(pwd)/output/adv${eps}_bs${batch_size}_neg${num_neg}
python main.py --job-type train --adv-eps ${eps} --num-neg ${num_neg} --batch-size ${batch_size} --output-root ${output_root}
```
### Target Embedding Extraction
After Model Training, we extract the target embedding by partially propagating the network. The target embedding will be used for the HNSW index building and online deployment.
After extraction, `item_ids.npy, item_embs.npy`will be saved in `${output_root}/embeddings`directory
```bash
python main.py --job-type extract_feature --output-root ${output_root}
```
### Build Index
We build the post-training index (HNSW in this case) based on target embedding. The index-related files (in numpy npy format) will be saved in `${output_root}/index`
```
python nann/delivery/build_hnsw_index.py -i ${output_root}/embeddings/item_embs.npy -o ${output_root}/index
```
### Model Evaluation
Recall with HNSW
```bash
python main.py --job-type test --output-root ${output_root}
```

Recall without HNSW but in a brute-force way
```
python main.py --job-type test_all --output-root ${output_root}
```

### Model Rewrite for Online Deployment
In this stage, we will generate several [GraphDef](https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphDef) protobufs, which are the protocols for online serving.

- exec.pb (binary format)/ exec.pbtxt (text format)
The graph_def that takes charge of the whole picture, i.e. the overall process of model feedforward and retrieval.
- frozen_graph.pb 
The graph_def that includes the operations for the deep model; frozen_graph.pb will be used in `tf.blaze_xla_op`,i.e. a Custom Op, located in exec.pb.
```bash
cd NANN_impls

#set python path
export PYTHONPATH=$PYTHONPATH:nann

# export the modified ckpt for online inference, 
python main.py --job-type export --output-root ${output_root}

# generate frozen_graph.pb (freeze model and small minor changes).
python nann/delivery/convert_meta.py --model_dir ${output_root}/export --output_dir ${output_root}/export/converted_model

# generate exec.pb. the outer graph (dense part + retrieval) for online inference
# generate exec.meta.pb, only need the signature_def which contains the tensor names of model I/O
python nann/delivery/build_opt_graph.py --model-dir ${output_root}/export --index-dir ${output_root}/index --item-embs-dir ${output_root}/embeddings --output-dir ${output_root}/export
```
The following script is to verify the correctness of exec.pb and frozen_graph.pb by running one "step" of TensorFlow computation, and executing every operation.
```bash
## exec.meta.pb is only for signature_def, which is used to feed and fetch model I/O
python nann/delivery/NANN_inference_demo.py --graph-file ${output_root}/export/exec.pb --meta-path ${output_root}/export/exec.meta.pb
```
### Model Benchmarking
We provide a handy benchmarking tool, called blaze-benchmark, to profiling the inference performance.

- gen_runmeta.py
The script to generate the mock data for model inputs.
- gen_benchmark_conf.py
The script to generate config file that is needed by the tool. 
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

### Model Deployment
We deploy NANN with TensorFlow Serving for demonstration.
Export model as SavedModel Format
```bash
python nann/delivery/pb_to_saved_model.py --model-path ${output_root}/export/exec.pb --meta-path ${output_root}/export/exec.meta.pb --export-dir ${output_root}/export/nann/1
```
Model Serving with Docker
```bash
docker run -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source=${output_root}/export/nann,target=/models/nann \
  -e MODEL_NAME=nann -t rihan19920210/nann_serving
```
A Smoking Test
tensorflow-serving-api==1.15.0
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
## Community

- Please see Contributing Guide before your first contribution
- Please Cite [NANN](https://dl.acm.org/doi/abs/10.1145/3511808.3557098#sec-ref) in your applications if it helps
```
article{chen2022approximate,
  title={Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation},
  author={Chen, Rihan and Liu, Bin and Zhu, Han and Wang, Yaoxuan and Li, Qi and Ma, Buting and Hua, Qingbo and Jiang, Jun and Xu, Yunlong and Deng, Hongbo and others},
  journal={arXiv preprint arXiv:2202.10226},
  year={2022}
}
```
