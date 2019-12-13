#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

DATASET="xview"
INPUT_RES=512
EXPID="dla_512"
GPUS="1,2,3,4"
BATCHSIZE=32


cd ../src
##
## train
python main.py \
    ctdet \
    --exp_id "$EXPID" \
    --dataset "$DATASET" \
    --input_res $INPUT_RES \
    --batch_size $BATCHSIZE \
    --num_epochs 100 \
    --lr_step 45,60 \
    --gpus "$GPUS" \
    --not_cuda_benchmark

#
# test
python \
    test.py ctdet \
    --exp_id "$EXPID" \
    --dataset "$DATASET" \
    --input_res $INPUT_RES \
    --resume \
    --gpus "$GPUS"

##
## flip test
python test.py \
    ctdet \
    --exp_id "$EXPID" \
    --dataset "$DATASET" \
    --input_res $INPUT_RES \
    --resume \
    --flip_test \
    --gpus "$GPUS"

##
## multi-scale
## TODO: Add multi-scale eval test
cd -
