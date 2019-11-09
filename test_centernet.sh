#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

##
## train
# Just expect the user to activate the correct codna env before launching this script, so comment this out.
# Conda stuff on gpu machine so far has been a bit weird:
# eval "$(conda shell.bash hook)"
# conda activate h4d_centernet
cd ../src
python main.py \
    ctdet \
    --exp_id pascal_dla_512_test \
    --dataset pascal \
    --input_res 512 \
    --batch_size 90 \
    --num_epochs 70 \
    --lr_step 45,60 \
    --gpus 0,1,2,3,8 \
    --not_cuda_benchmark

##
## test
python \
    test.py ctdet \
    --exp_id pascal_dla_512_test \
    --dataset pascal \
    --input_res 512 --resume

##
## flip test
python test.py \
    ctdet \
    --exp_id pascal_dla_512_test \
    --dataset pascal \
    --input_res 512 \
    --resume \
    --flip_test
cd ..