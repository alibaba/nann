#!/bin/bash

set -xe

cat <<EOF
============================================
Building Index Build TF Model Image ...
============================================
EOF

IMAGE=reg.docker.alibaba-inc.com/matching/nann_opensource
TAG=10.2-cudnn7-devel-ubuntu18.04

docker build -t ${IMAGE}:${TAG} --network host -f Dockerfile .

docker push ${IMAGE}:${TAG}

