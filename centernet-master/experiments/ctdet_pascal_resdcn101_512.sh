#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

cd ../src
##
## train
python main.py ctdet \
    --exp_id pascal_resdcn101_512 \
    --arch resdcn_101 \
    --dataset pascal \
    --input_res 512 \
    --num_epochs 70 \
    --lr_step 45,60 \
    --batch_size 25 \
    --master_batch_size 8
    --gpus 0,1,2,3,4,5,6,7,8,9

##
## test
python test.py ctdet \
    --exp_id pascal_resdcn101_512 \
    --arch resdcn_101 \
    --dataset pascal \
    --input_res 512 \
    --resume

##
## flip test
python test.py ctdet \
    --exp_id pascal_resdcn101_512 \
    --arch resdcn_101 \
    --dataset pascal \
    --input_res 512 \
    --resume \
    --flip_test
cd -
