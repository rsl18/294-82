#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate h4d_env


# Use this command to get dataset size: ls -1q ./datasets/Xview/coco_chipped/train/ | wc -l
# xview_coco_v2_train_chipped: 10557, xview_coco_v2_tiny_train_chipped: 128, COCO:82783, VOC0712: 16551
DATASET_SIZE=82783

# batch sizes: SSD300: { TitanXp: 48, RTX: 96 }, SSD512 { TitanXp: 20, }
BATCH_SIZE=48
ITERS_PER_EPOCH=$((DATASET_SIZE/$BATCH_SIZE))
EPOCHS=10000
# MAX_ITER is what ssd-master uses to know how many training iterations to run.
# If resuming from previous training,t he iteration number picks up from where
# the previous training left off, so you'll want to hand-pick this value accordingly:
MAX_ITER=$(($ITERS_PER_EPOCH*$EPOCHS*2))
LOG_STEP=$(($ITERS_PER_EPOCH/50))
SAVE_STEP=$ITERS_PER_EPOCH
EVAL_STEP=$((ITERS_PER_EPOCH*2))

# Ensure LOG_STEP>0:
LOG_STEP=$(( LOG_STEP > 0 ? LOG_STEP : ITERS_PER_EPOCH ))

echo "DATASET_SIZE: ${DATASET_SIZE}"
echo "ITERS_PER_EPOCH: ${ITERS_PER_EPOCH}"
echo "EPOCHS: ${EPOCHS}"
echo "MAX_ITER: ${MAX_ITER}"
echo "LOG_STEP: ${LOG_STEP}"
echo "EVAL_STEP: ${EVAL_STEP}"
# exit

# COCO, Sigma at both boxheads:
CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
    --config-file vgg_ssd300_coco_trainval35k_sigma.yaml \
    --log_step $LOG_STEP \
    --save_step $SAVE_STEP \
    --eval_step $EVAL_STEP \
    SOLVER.MAX_ITER $MAX_ITER \
    SOLVER.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE 128 \
    RESUME_FROM_EXPERIMENT "/home/giscard/h4d/experiments/ssd/giscard/2019-07-31--14-27-03/best_ssd300-vgg_coco_2014_train_0_0_448240.pth" \
    TRAIN.RESCALE_FACTOR 0.85 \
    TEST.RESCALE_FACTOR 0.85 \
    SOLVER.LR 1e-6 \
    FOO 1 # Keep this as last line of script: \

# ## COCO, baseline:
# CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
#     --config-file vgg_ssd300_coco_trainval35k.yaml \
#     --log_step $LOG_STEP \
#     --save_step $SAVE_STEP \
#     --eval_step $EVAL_STEP \
#     SOLVER.MAX_ITER $MAX_ITER \
#     SOLVER.BATCH_SIZE $BATCH_SIZE \
#     TEST.BATCH_SIZE 32 \
#     RESUME_FROM_EXPERIMENT "/home/giscard/h4d/experiments/ssd/giscard/2019-08-02--12-24-54/best_ssd300-vgg_coco_2014_train_0_0_603400.pth" \
#     FOO 1 # Keep this as last line of script: \
