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
    --exp_id pascal_dlasigma_512 \
    --dataset pascal \
    --arch dlasigma_34 \
    --input_res 512 \
    --batch_size 90 \
    --num_epochs 70 \
    --lr_step 45,60 \
    --gpus 0,1,2,3,4 \
    --not_cuda_benchmark

##
## test
python \
    test.py ctdet \
    --exp_id pascal_dlasigma_512 \
    --dataset pascal \
    --arch dlasigma_34 \
    --input_res 512 --resume

##
## flip test
python test.py \
    ctdet \
    --exp_id pascal_dlasigma_512 \
    --dataset pascal \
    --arch dlasigma_34 \
    --input_res 512 \
    --resume \
    --flip_test
cd -