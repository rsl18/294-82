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
    --exp_id coco_dla_1x \
    --dataset coco \
    --input_res 512 \
    --batch_size 150 \
    --master_batch 9 \
    --num_epochs 70 \
    --lr 5e-4 \
    --gpus 0,1,2,3,4,5,6,7,8,9 \
    --not_cuda_benchmark

##
## test
python test.py ctdet \
    --exp_id coco_dla_1x \
    --keep_res \
    --dataset coco \
    --resume
    # --load_model ../models/pretrained/ctdet_coco_dla_1x.pth \

##
## flip test
python test.py ctdet \
    --exp_id coco_dla_1x \
    --keep_res \
    --resume \
    --flip_test

##
## multi scale test
python test.py ctdet \
    --exp_id coco_dla_1x \
    --keep_res \
    --resume \
    --flip_test \
    --test_scales 0.5,0.75,1,1.25,1.5
cd -
