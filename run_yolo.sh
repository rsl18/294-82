#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate h4d_env

cd yolov3

# For GPU0, ideal batch size is 32 for calls to evaluate (one call in train.py and one in test.py)

# optimal batch_size below is 32 for GPU0, (TitanXP 12GB RAM: batch_sz=16)

EPOCHS=100
BATCH_SIZE=32
IMG_SIZE=416
CHECKPOINT_INTERVAL=1
EVAL_INTERVAL=1

#CUDA_VISIBLE_DEVICES=1 python3 train.py \
#    --data_config config/xview_coco_v2.data \
#    --pretrained_weights weights/darknet53.conv.74 \
#    --epochs $EPOCHS \
#    --batch_size $BATCH_SIZE \
#    --img_size $IMG_SIZE \
#    --checkpoint_interval $CHECKPOINT_INTERVAL \
#    --evaluation_interval $EVAL_INTERVAL &

 #Run original yolo on MSCOCO:
  CUDA_VISIBLE_DEVICES=0 python3 train.py \
      --data_config config/coco.data \
      --model_def config/yolov3.cfg \
      --n_cpu 2 \
      --pretrained_weights ../experiments/yolo/adam/2019-07-16--16-20-44/last_ckpt18_yolov3.pth \
      --epochs 100 \
      --batch_size 32 \
      --img_size 416 \
      --checkpoint_interval 1 \
      --evaluation_interval 1


# ## Trying resuming training, yolo on MSCOCO (it worked):
# CUDA_VISIBLE_DEVICES=0 python3 train.py \
#     --data_config config/coco.data \
#     --model_def config/yolov3.cfg \
#     --n_cpu 2 \
#     --pretrained_weights ../experiments/yolo/gbiamby/2019-07-10--00-39-13/yolov3_ckpt_3.pth \
#     --epochs 100 \
#     --batch_size 16 \
#     --img_size 416 \
#     --checkpoint_interval 1 \
#     --evaluation_interval 1


# Run original yolo on xview_tiny:
#  CUDA_VISIBLE_DEVICES=0 python3 train.py \
#      --data_config config/xview_coco_v2_tiny.data \
#      --model_def config/yolov3_xview_coco_v2.cfg \
#      --n_cpu 2 \
#      --pretrained_weights weights/darknet53.conv.74 \
#      --epochs 3 \
#      --batch_size 2 \
#      --img_size 512 \
#      --checkpoint_interval 1 \
#      --evaluation_interval 1
