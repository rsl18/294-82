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
    --exp_id fake \
    --dataset coco \
    --input_res 512 \
    --batch_size 1 \
    --master_batch 1 \
    --num_epochs 1 \
    --lr 5e-4 \
    --not_cuda_benchmark
