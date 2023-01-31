# Build instructions for internal use.
1. Build tensorflow (http://gitlab.alibaba-inc.com/TargetAdvertising/tensorflow.git, blaze\_aios\_benchmark).
2. Install glog-devel if not installed.
3. Edit `Build.sh`, set cmake variables, and build the benchmark tool:
4. Prepare inputs and model.
* [Optional] Prepare runmeta, set biz\_options (after "session\_config") and send requests to biz, and copy generated runmeta files to local.
   "run\_options": {
     "traceTensorInfos": true,
     "traceLevel": "SOFTWARE\_TRACE"
   },
* Copy pbtxt and pb models to local.
5. Prepare benchmark configs (see example) and run.

# Build instructions for partners.
1. Build tensorflow (blaze/open\_source\_benchmark).
* Recommend build env and dependencies:
  - docker: nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
  - If build with gpu:
    - cuda: 10.2, or 11.2
    - cudnn: 7.6.5
    - cuda driver: 460.73.01
    - tensorrt: 8.0.3.4
  - Example .tf\_configure.bazelrc:
    build --action_env PYTHON_BIN_PATH="/opt/conda/bin/python"
    build --action_env PYTHON_LIB_PATH="/opt/conda/lib/python3.7/site-packages"
    build --python_path="/opt/conda/bin/python"
    build:xla --define with_xla_support=true
    build --config=xla
    build --config=tensorrt
    build --action_env TF_CUDA_VERSION="11"
    build --action_env TF_CUDNN_VERSION="7"
    build --action_env TF_TENSORRT_VERSION="8"
    build --action_env TF_NCCL_VERSION=""
    build --action_env TF_CUDA_PATHS="/usr/local/cuda-11.2"
    build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-11.2"
    build --action_env TF_CUDA_COMPUTE_CAPABILITIES="7.5,7.5"
    build --action_env LD_LIBRARY_PATH="/usr/local/nvidia/lib64/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
    build --config=cuda
    build:opt --copt=-march=native
    build:opt --copt=-Wno-sign-compare
    build:opt --host_copt=-march=native
    build:opt --define with_default_optimizations=true
    build:v2 --define=tf_api_version=2
    test --flaky_test_attempts=3
    test --test_size_filters=small,medium
    test --test_tag_filters=-benchmark-test,-no_oss,-oss_serial
    test --build_tag_filters=-benchmark-test,-no_oss
    test --test_tag_filters=-gpu
    test --build_tag_filters=-gpu
    build --action_env TF_CONFIGURE_IOS="0"
* Run build\_whl.sh in tensorflow dir.
2. Install glog-devel if not installed.
3. Edit `Build.sh`, set cmake variables, and run `Build.sh`.
4. Run benchmark (see example/DNN/) `run.sh`, change setting as needed.
