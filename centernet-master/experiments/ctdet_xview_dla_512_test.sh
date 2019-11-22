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
    --exp_id xview_dla_512_again \
    --dataset xview \
    --lr 5e-4 \
    --batch_size 144 \
    --num_epochs 140 \
    --gpus 0,1,2,3,4,5,6,7,8,9 \
    --not_cuda_benchmark \
    --num_workers 16

##
## test
python test.py \
    ctdet \
    --exp_id xview_dla_512_again \
    --dataset xview \
    --keep_res \
    --resume

##
## flip test
python test.py \
    ctdet \
    --exp_id xview_dla_512_again \
    --dataset xview \
    --keep_res \
    --resume \
    --flip_test

##
## multi scale test
python test.py \
    ctdet \
    --exp_id xview_dla_512_again \
    --keep_res \
    --resume \
    --flip_test \
    --test_scales 0.5,0.75,1,1.25,1.5
cd -