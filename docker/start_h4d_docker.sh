#!/bin/bash

docker run -it \
 --gpus all \
 --name h4d_run \
 --mount type=bind,source="$(pwd)"/../../data,target=/h4d_root/data \
 --shm-size=8G \
 -p 6006:6006 \
 -p 6007:6007 \
 -p 6008:6008 \
 -p 6009:6009 \
 h4d_base
