#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

cd ../src
##
## train
python main.py \
    ctdet \
    --exp_id xview_dla_512_test \
    --dataset xview \
    --input_res 512 \
    --batch_size 20 \
    --num_epochs 100 \
    --lr_step 45,60 \
    --gpus 2 \
    --not_cuda_benchmark

#
# test
python \
    test.py ctdet \
    --exp_id xview_dla_512_test \
    --dataset xview \
    --input_res 512 \
    --resume \
    --gpus 2

##
## flip test
python test.py \
    ctdet \
    --exp_id xview_dla_512_test \
    --dataset xview \
    --input_res 512 \
    --resume \
    --flip_test \
    --gpus 2

##
## multi-scale
## TODO: Add multi-scale eval test
cd -
