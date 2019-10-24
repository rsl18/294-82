#!/bin/bash
if [[ -f log.txt ]]; then
    rm log.txt
fi
# exec &>> log.txt

cd ../src
##
## train
python main.py ctdet \
	--exp_id pascal_resdcn101_384 \
	--arch resdcn_101 \
	--dataset pascal \
	--num_epochs 70 \
	--lr_step 45,60 \
    --batch_size 90 \
	--gpus 0,1,2,3,4

##
## test
python test.py ctdet \
	--exp_id pascal_resdcn101_384 \
	--arch resdcn_101 \
	--dataset pascal \
	--resume

##
## flip test
python test.py ctdet \
	--exp_id pascal_resdcn101_384 \
	--arch resdcn_101 \
	--dataset pascal \
	--resume \
	--flip_test
cd -
