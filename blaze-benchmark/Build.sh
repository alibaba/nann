#!/bin/bash
set -x

pushd thirdparty/boost_1_53_0
if [ ! -d stage/lib ]; then
./bootstrap.sh --with-libraries=chrono,filesystem,system,thread && ./b2 -j32
fi
popd

if [ ! -d build ]; then
    mkdir build
fi
pushd build

LWP=`pwd`

cmake .. \
    -DTF_SRC_DIR=$(dirname "$(dirname `pwd`)")/tensorflow \
    -DUSE_CUDA=1 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DAIOS_SRC_DIR=/path/to/aios \
    -DAIOS=0
make VERBOSE=1 -j 32
popd
