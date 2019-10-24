#!/bin/bash

cd ../src
python check_models.py ctdet \
    --exp_id check_models \
    --arch hourglass \
    --dataset pascal \
    --input_res 384 \
    --resume
cd -