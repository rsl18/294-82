#!/bin/bash
. utils.sh
. setup_options.sh
CENTERNET_ENV="h4d_env"

eval "$(conda shell.bash hook)"
conda activate $CENTERNET_ENV


# Expand relative paths to absolute ones (required for symlink creation to work):
if [ ! "$H4D_DATA_DIR" ] && [ ! "$H4D_EXPERIMENTS_DIR" ]; then
    echo "Are you setting up a local repo on the BID machine (y/n)"
    read REPO

    if [ "$REPO" = "y" ]; then
        #if server
        echo "BID machine, ok, using absolute paths"
        DATA_DIR="/home/data"
        EXPERIMENTS_DIR="/home/experiments"
    else
        DATA_DIR="$(abs_path "../data")"
        EXPERIMENTS_DIR="$(abs_path "../experiments")"
    fi
else
    # Env. variables prefixed with H4D_ defined in setup_options.sh take precedence over
    # everything:
    DATA_DIR="$H4D_DATA_DIR"
    EXPERIMENTS_DIR="$H4D_EXPERIMENTS_DIR"
fi

echo "Using DATA_DIR: " "$DATA_DIR"
echo "Using EXPERIMENTS_DIR: " "$EXPERIMENTS_DIR"


## Symlink data:
if [ -d centernet-master/data ]; then rm centernet-master/data; fi
ln -sfv "$DATA_DIR" centernet-master/data

# COCO:
# Delete and recreate the datasets symlinks:
# if [ -d centernet-master/data/coco/annotations ]; then rm centernet-master/data/coco/annotations; fi
# if [ -d centernet-master/data/coco/images ]; then rm centernet-master/data/coco/images; fi
# ln -sfv "$DATA_DIR"/coco/annotations/ centernet-master/data/coco/annotations
# ln -sfv "$DATA_DIR"/coco/images/ centernet-master/data/coco/images
#VOC:
# if [ -d centernet-master/data/voc ]; then rm centernet-master/data/voc; fi
# ln -sfv "$DATA_DIR"/voc/voc_combined/ centernet-master/data/voc
# #XVIEW:
# if [ -d centernet-master/data/Xview ]; then rm centernet-master/data/Xview; fi
# ln -sfv "$DATA_DIR"/Xview/coco_chipped centernet-master/data/Xview
# #COCO:
# if [ -d centernet-master/data/coco ]; then rm centernet-master/data/coco; fi
# ln -sfv "$DATA_DIR"/coco centernet-master/data/coco


## Setup model softlinks:
if [ ! -d "$EXPERIMENTS_DIR"/centernet ]; then mkdir -p "$EXPERIMENTS_DIR"/centernet; fi
if [ ! -d "$EXPERIMENTS_DIR"/base_models/centernet ]; then mkdir -p "$EXPERIMENTS_DIR"/base_models/centernet; fi
if [ ! -d centernet-master/models ]; then mkdir -p centernet-master/models; fi
ln -sfvn "$EXPERIMENTS_DIR"/base_models/centernet centernet-master/models/pretrained

#EXPERIMENTS:
if [ -d centernet-master/exp ]; then rm centernet-master/exp; fi
ln -sfv "${EXPERIMENTS_DIR}"/centernet centernet-master/exp

# Note, centernet's pretrained models must be manually downloaded from google drive:
# see MODEL_ZOO.md for list of pretrained models and their URL's
# (save them to centernet-master/models/pretrained/)