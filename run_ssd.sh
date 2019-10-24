#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate h4d_env


# Use this command to get dataset size: ls -1q ./datasets/Xview/coco_chipped/train/ | wc -l
# xview_coco_v2_train_chipped: 10557, xview_coco_v2_tiny_train_chipped: 128, COCO:82783, VOC0712: 16551
DATASET_SIZE=33299

# batch sizes: SSD300: { TitanXp: 48, RTX: 96 }, SSD512 { TitanXp: 20, }
BATCH_SIZE=30
ITERS_PER_EPOCH=$((DATASET_SIZE/$BATCH_SIZE))
EPOCHS=1000
# MAX_ITER is what ssd-master uses to know how many training iterations to run.
# If resuming from previous training,t he iteration number picks up from where
# the previous training left off, so you'll want to hand-pick this value accordingly:
MAX_ITER=$(($ITERS_PER_EPOCH*$EPOCHS*2))
LOG_STEP=$(($ITERS_PER_EPOCH/50))
SAVE_STEP=$ITERS_PER_EPOCH
EVAL_STEP=$((ITERS_PER_EPOCH*18))

# Ensure LOG_STEP>0:
LOG_STEP=$(( LOG_STEP > 0 ? LOG_STEP : ITERS_PER_EPOCH ))

echo "DATASET_SIZE: ${DATASET_SIZE}"
echo "ITERS_PER_EPOCH: ${ITERS_PER_EPOCH}"
echo "EPOCHS: ${EPOCHS}"
echo "MAX_ITER: ${MAX_ITER}"
echo "LOG_STEP: ${LOG_STEP}"
echo "EVAL_STEP: ${EVAL_STEP}"
# exit


## multi GPU
python -m torch.distributed.launch --nproc_per_node=2 ssd-master/train.py \
    --config-file sigma-voc/vgg_ssd300_ucb_coco.yaml \
    --log_step $LOG_STEP \
    --save_step $SAVE_STEP \
    --eval_step $EVAL_STEP \
    SOLVER.MAX_ITER $MAX_ITER \
    SOLVER.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE 12 \

## GPU COCO sigma:
# CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
#     --config-file sigma-voc/vgg_ssd300_voc0712_baseline.yaml \
#     --log_step $LOG_STEP \
#     --save_step $SAVE_STEP \
#     --eval_step $EVAL_STEP \
#     SOLVER.MAX_ITER $MAX_ITER \
#     SOLVER.BATCH_SIZE $BATCH_SIZE \
#     TEST.BATCH_SIZE 32 \
    # MODEL.BACKBONE.CONV6_SIGMA True \
    # FOO 1 & # Keep this as last line of script: \

## GPU Xview sigma:
# CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
#     --config-file vgg_ssd512_xview_sigma.yaml \
#     --log_step $LOG_STEP \
#     --save_step $SAVE_STEP \
#     --eval_step $EVAL_STEP \
#     SOLVER.MAX_ITER $MAX_ITER \
#     SOLVER.BATCH_SIZE $BATCH_SIZE \
#     DATASETS.TRAIN '("xview_coco_v2_train",)' \
#     DATASETS.TEST '("xview_coco_v2_val",)' \
#     TEST.BATCH_SIZE 48 \
#     FOO 1 & # Keep this as last line of script: \

# ## GPU (default) running:
# CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
#     --config-file vgg_ssd300_xview_sigma.yaml \
#     --log_step $LOG_STEP \
#     --save_step $SAVE_STEP \
#     --eval_step $EVAL_STEP \
#     SOLVER.MAX_ITER $MAX_ITER \
#     SOLVER.BATCH_SIZE $BATCH_SIZE \
#     DATASETS.TRAIN '("xview_coco_v2_train",)' \
#     DATASETS.TEST '("xview_coco_v2_val",)' \
#     TEST.BATCH_SIZE 32 \
#     FOO 1 #& # Keep this as last line of script: \

## Example of resuming training from previous experiment:
# CUDA_VISIBLE_DEVICES=0 python ssd-master/train.py \
#     --config-file vgg_ssd512_xview_coco.yaml \
#     --log_step $LOG_STEP \
#     --save_step $SAVE_STEP \
#     --eval_step $EVAL_STEP \
#     SOLVER.MAX_ITER $MAX_ITER \
#     SOLVER.BATCH_SIZE $BATCH_SIZE \
#     DATASETS.TRAIN '("xview_coco_v2_train",)' \
#     DATASETS.TEST '("xview_coco_v2_val",)' \
#     TEST.BATCH_SIZE 32 \
#     RESUME_FROM_EXPERIMENT "/home/experiments/ssd/gbiamby/C2019-07-21--13-06-55" \
    # FOO 1 #& # Keep this as last line of script: \


# # CPU/local running:
# python ssd-master/train.py \
#     --config-file vgg_ssd300_xview_coco.yaml \
#     MODEL.DEVICE cpu \
#     SOLVER.BATCH_SIZE 4 \
#     SOLVER.WARMUP_ITERS 1 \
#     SOLVER.MAX_ITER 5 \
#     DATASETS.TRAIN '("xview_coco_v2_tiny_train",)' \
#     DATASETS.TEST '("xview_coco_v2_tiny_val",)' \
#     FOO 1 & # Keep this as last line of script: \


## ------------------------------------------------------------------------------------
## NOTES
## ------------------------------------------------------------------------------------
## One epoch takes ~7 minutes on Titan Xp with batch size 16, test.batch_size=24, on
## SSD512, and eval takes ~14 minutes so it's best to set eval interval to at least
## once every 10 epochs, otherwise we are spending more time on eval than training.
##
## TODO 1:
## We should probably try to move all the above calculations to inside the python training
## code. It's not good to have to manually calculate the number of annotations for each
## dataset to store into the DATASET_SIZE param above. We should just be able to specify
## batch_size, the intervals for logging/eval, etc, and let the python code figure out
## the rest.
##
## SSD512: TEST.BATCH_SIZE: {Titan Xp: 32 , RTX: >=48 (higher might work, but 48 worked with xview on the RTX)}