#!/bin/bash
. utils.sh
# source /home/gbiamby/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate h4d_env

function selcompiler() {
	if [[ "${OS}" == "Linux" ]]; then
		python build.py build_ext develop
	else
		CFLAGS='-stdlib=libc++' python build.py build_ext develop
	fi
}

function abs_path() {
    cd "$1"
    loc="$(pwd)"
    echo "$loc"
}

DATA_TARGET_VERSION="v2"

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

# Build step:
eval "$(conda shell.bash hook)"
conda activate h4d_env
cd ssd-master/ext
selcompiler
cd ../..

# Symlink data:
if [ ! -d ssd-master/datasets ]; then mkdir ssd-master/datasets; fi
# Delete and recreate the datasets symlinks:
if [ -d ssd-master/datasets/annotations ]; then rm ssd-master/datasets/annotations; fi
if [ -d ssd-master/datasets/train2014 ]; then rm ssd-master/datasets/train2014; fi
if [ -d ssd-master/datasets/val2014 ]; then rm ssd-master/datasets/val2014; fi
ln -sfv "$DATA_DIR"/Xview/coco_chipped/ ssd-master/datasets/annotations
ln -sfv "$DATA_DIR"/Xview/coco_chipped/ ssd-master/datasets/train2014
ln -sfv "$DATA_DIR"/Xview/coco_chipped/ ssd-master/datasets/val2014

# Setup model softlinks:
mkdir -p ssd-master/weights
mkdir -p "$EXPERIMENTS_DIR"/ssd
# ln -sfn "$EXPERIMENTS_DIR"/ssd ssd-master/weights/checkpoints
mkdir -p "$EXPERIMENTS_DIR"/base_models/ssd
ln -sfn "$EXPERIMENTS_DIR"/base_models/ssd ssd-master/weights/pretrained

# Download pretrained weights:
if [ ! -f "ssd-master/weights/pretrained/vgg16_reducedfc.pth" ]; then
    dlcmd https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth -O ssd-master/weights/pretrained/vgg16_reducedfc.pth
fi
