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
    --exp_id xview_dlasigma_512 \
    --dataset xview \
    --arch dlasigma_34 \
    --input_res 512 \
    --batch_size 90 \
    --num_epochs 100 \
    --lr_step 55,65,75,85 \
    --gpus 5,6,7,8,9 \
    --not_cuda_benchmark

##
## test
python test.py \
    ctdet \
    --exp_id xview_dlasigma_512 \
    --dataset xview \
    --arch dlasigma_34 \
    --input_res 512 \
    --resume \
    --gpus 5,6,7,8,9

##
## flip test
python test.py \
    ctdet \
    --exp_id xview_dlasigma_512 \
    --dataset xview \
    --arch dlasigma_34 \
    --input_res 512 \
    --resume \
    --flip_test \
    --gpus 5,6,7,8,9

cd -
