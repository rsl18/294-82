# #!/bin/bash

# THIS FILE IS NO LONGER NEEDED IN H4D. This is all handled by setup_coco.sh, and
# setup_yolo.sh (both of which get called by setup.sh, one setup script to rule them
# all). We should delete this from the repo.


# # CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

# # # Clone COCO API
# # git clone https://github.com/pdollar/coco
# mkdir -p ./coco
# cd ./coco

# echo "Edit this script to use either OPTION 1 (download COCO) or OPTION 2 (symlink to COCO already on this computer somewhere)"

# ##
# ## OPTION 1: Download all of MS COCO images. Use this if you don't have a copy of MS COCO
# ## images on your computer already. It's about ~20GB.
# ## /begin OPTION 1:
# # mkdir -p ./coco/images
# # cd ./images
# ## Download Images
# # wget -c https://pjreddie.com/media/files/train2014.zip wget -c
# # https://pjreddie.com/media/files/val2014.zip
# # unzip -q train2014.zip
# # unzip -q val2014.zip
# # cd ..
# ## /end OPTION 1
# ##

# ## should be in yolov3/data/coco/ now

# ##
# ## OPTION 2: Symlink option:
# ## If coco dataset already exists somewhere on this machine, set up the symlinks.
# ## /begin OPTION 2:
# COCO_IMGS="/home/gbiamby/school/datasets/coco/images"
# COCO_ANNS="/home/gbiamby/school/datasets/coco/annotations"
# # COCO_IMGS="/home/<your-path-to-coco>/coco/images"
# # COCO_ANNS="/home/<your-path-to-coco/coco/annotations"
# ln -sfnv "$COCO_IMGS" images
# ln -sfnv "$COCO_ANNS" annotations
# ## /end OPTION 2
# ##

# ## Note:
# ## Once you extract the images, or symlink them to wherever your pre-existing copy of MS
# ## COCO is, your folder structure should look like:
# ##   yolov3/data/coco/images/train2014
# ##   yolov3/data/coco/images/val2014

# # Download COCO Metadata (annotations)
# if [ ! -f instances_train-val2014.zip ]; then
#     wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
# fi
# if [ ! -f 5k.part ]; then
#     wget -c https://pjreddie.com/media/files/coco/5k.part
# fi
# if [ ! -f trainvalno5k.part ]; then
#     wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
# fi
# if [ ! -f labels.tgz ]; then
#     wget -c https://pjreddie.com/media/files/coco/labels.tgz
# fi
# tar xzf labels.tgz
# unzip -qo instances_train-val2014.zip


# # # # Set Up Image Lists
# paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
# paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
