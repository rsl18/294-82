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
    --exp_id pascal_hg_384 \
    --arch hourglass \
    --dataset pascal \
    --input_res 384 \
    --batch_size 48 \
    --master_batch 8 \
    --lr 2.5e-4 \
    --num_epochs 70 \
    --lr_step 45,60 \
    --gpus 0,1,2,3,4 \
    --not_cuda_benchmark

##
## test
python \
    test.py ctdet \
    --exp_id pascal_hg_384 \
    --arch hourglass \
    --dataset pascal \
    --input_res 384 \
    --resume

##
## flip test
python test.py \
    ctdet \
    --exp_id pascal_hg_384 \
    --arch hourglass \
    --dataset pascal \
    --input_res 384 \
    --resume \
    --flip_test

##
## multi scale test
python test.py ctdet \
    --exp_id pascal_hg_384 \
    --arch hourglass \
    --keep_res \
    --resume \
    --flip_test \
    --test_scales 0.5,0.75,1,1.25,1.5,1.75,2.0,2.5,3.0
cd -
