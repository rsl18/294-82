# # COCO SSD300 baseline:
# python ssd-master/test.py \
#     --config-file vgg_ssd300_coco_trainval35k.yaml \
#     --ckpt experiments/ssd/giscard/2019-08-02--12-24-54/best_ssd300-vgg_coco_2014_train_0_0_603400.pth \
#     # DATASETS.TRAIN '("coco_tiny_test",)' \
#     # DATASETS.TEST '("coco_tiny_test",)' \

# COCO, SSD300 sigma (both boxheads):
python ssd-master/test.py \
    --config-file vgg_ssd300_coco_trainval35k_sigma.yaml \
    --ckpt experiments/ssd/giscard/2019-07-31--14-27-03/best_ssd300-vgg_coco_2014_train_0_0_448240.pth 
    -- SOLVER.LR 1e-4