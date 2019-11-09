#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate h4d_env

cd faster-rcnn.pytorch

GPU_ID=0
DATASET='xview_coco'
LOAD_DIR='home/experiments/faster_rcnn/ross/'
CHECKPOINT_FILE='res101_xview_coco_1_69_935.pth'

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset "$DATASET" --load_dir "$LOAD_DIR" --load_name "$CHECKPOINT_FILE" --cuda --mGPUs &


