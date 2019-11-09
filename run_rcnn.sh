#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate h4d_env

cd faster-rcnn.pytorch

GPU_ID=0
DATASET='xview_coco' # Can also use 'xview_coco_tiny' for quicker code tests
NET='res101'
# 23 on RTX, 12 on Titan XP
BATCHSIZE=23
LR='1e-3'
MAX_EPOCH=100
DECAY_STEP=5
NUM_WORKERS=1

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py --epochs $MAX_EPOCH --dataset $DATASET --net $NET --bs $BATCHSIZE --nw $NUM_WORKERS --lr $LR --lr_decay_step $DECAY_STEP --cuda --mGPUs --use_tfb &