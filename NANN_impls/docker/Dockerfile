FROM reg.docker.alibaba-inc.com/blaze/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV BLAZE_HOME=/root

WORKDIR $BLAZE_HOME

COPY env/sources.list /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  cpio \
  file \
  flex \
  g++ \
  git \
  gdb \
  make \
  patch \
  rpm2cpio \
  unar \
  wget \
  unzip \
  xz-utils \
  cmake \
  vim \
  libboost-all-dev \
  libgoogle-glog-dev \
  && \
  rm -rf /var/lib/apt/lists/*

RUN wget http://211619.oss-cn-hangzhou-zmf.aliyuncs.com/public/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda\
    && rm -rf Miniconda3-latest-Linux-x86_64.sh
RUN wget http://211619.oss-cn-hangzhou-zmf.aliyuncs.com/public/tf_115_pkgs.tgz\
    && tar -zxvf tf_115_pkgs.tgz && cd tf_115_pkgs && ls | xargs /opt/conda/bin/pip install\
    && cd .. && rm -rf tf_115_pkgs*
RUN wget http://170090.oss-cn-hangzhou-zmf.aliyuncs.com/public/faiss_pkgs_open_source.tgz\
    && tar -zxvf faiss_pkgs_open_source.tgz\
    && cd faiss_pkgs\
    && ls | xargs /opt/conda/bin/conda install\
    && cd .. && rm -rf faiss_pkgs*
RUN wget http://170090.oss-cn-hangzhou-zmf.aliyuncs.com/public/bazel-0.24.1-installer-linux-x86_64.sh\
    && bash bazel-0.24.1-installer-linux-x86_64.sh && rm -rf bazel-0.24.1-installer-linux-x86_64.sh 
RUN wget http://170090.oss-cn-hangzhou-zmf.aliyuncs.com/public/open_source_cache.tgz\
    && mkdir -p /root/.cache/bazel/_bazel_root\
    && tar -zxvf open_source_cache.tgz -C /root/.cache/bazel/_bazel_root\
    && rm -rf open_source_cache.tgz
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64/:$LD_LIBRARY_PATH
