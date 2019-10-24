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
    --exp_id xview_dlasigmafull_34 \
    --dataset xview \
    --arch dlasigmafull_34 \
    --input_res 512 \
    --batch_size 90 \
    --num_epochs 70 \
    --lr_step 45,60 \
    --gpus 5,6,7,8,9 \
    --not_cuda_benchmark

##
## test
python \
    test.py ctdet \
    --exp_id xview_dlasigmafull_34 \
    --dataset xview \
    --arch dlasigmafull_34 \
    --input_res 512 \
    --gpus 5,6,7,8,9 \
    --resume

##
## flip test
python test.py \
    ctdet \
    --exp_id xview_dlasigmafull_34 \
    --dataset xview \
    --arch dlasigmafull_34 \
    --input_res 512 \
    --resume \
    --gpus 5,6,7,8,9 \
    --flip_test
cd -