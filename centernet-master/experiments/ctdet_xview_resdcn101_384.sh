#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

cd ../src
##
## train
python main.py ctdet \
	--exp_id xview_resdcn101_384 \
	--arch resdcn_101 \
	--dataset xview \
    --input_res 384 \
	--num_epochs 70 \
	--lr_step 45,60 \
    --batch_size 90 \
    --master_batch_size 20 \
	--gpus 5,6,7,8,9

##
## test
python test.py ctdet \
	--exp_id xview_resdcn101_384 \
	--arch resdcn_101 \
	--dataset xview \
	--resume

##
## flip test
python test.py ctdet \
	--exp_id xview_resdcn101_384 \
	--arch resdcn_101 \
	--dataset xview \
	--resume \
	--flip_test
cd -
