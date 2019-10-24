#!/bin/bash
. ../utils.sh

#Requires docker and cuda toolkit

git clone https://github.com/mlaico/h4d.git
cd h4d
git checkout enclave
cd ..
mkdir -p experiments/base_models/faster_rcnn
mkdir -p experiments/base_models/ssd

# Download pretrained weights:
if [ ! -f "experiments/base_models/ssd/vgg16_reducedfc.pth" ]; then
    dlcmd https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth -O experiments/base_models/ssd/vgg16_reducedfc.pth
fi

if [ ! -f "experiments/base_models/faster_rcnn/resnet101_caffe.pth" ]; then
	dlcmd https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth -O experiments/base_models/faster_rcnn/resnet101_caffe.pth
fi

sudo docker build -t=env_base -f env_base_docker .