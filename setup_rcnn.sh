#!/bin/bash
. utils.sh
eval "$(conda shell.bash hook)"
conda activate h4d_env

function selcompiler() {
        if [[ "${OS}" == "Linux" ]]; then
		python setup.py build develop
        else
                CFLAGS='-stdlib=libc++' python setup.py build develop
        fi
}

function abs_path() {
	cd "$1"
	loc="$(pwd)"
	echo "$loc"
}


# Expand relative paths to absolute ones (required for symlink creation to work):

if [ ! "$H4D_DATA_DIR" ] && [ ! "$H4D_EXPERIMENTS_DIR" ]; then
    #relative path for local repo
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

# Setup python libs
cd faster-rcnn.pytorch/lib
# adding CFLAGS arg to fix g++ compiling issue
selcompiler
cd ../..

# Data dir
mkdir -p faster-rcnn.pytorch/data
mkdir -p faster-rcnn.pytorch/data/Xview

rm -rf faster-rcnn.pytorch/data/cache

# Setup data softlinks
ln -sfnv "$DATA_DIR"/Xview/coco_chipped "./faster-rcnn.pytorch/data/Xview/annotations"
ln -sfnv "$DATA_DIR"/Xview/coco_chipped "./faster-rcnn.pytorch/data/Xview/images"
ln -sfnv "$DATA_DIR"/coco "./faster-rcnn.pytorch/data/coco"


# Setup model softlinks
mkdir -p "$EXPERIMENTS_DIR"/faster_rcnn

mkdir -p "$EXPERIMENTS_DIR"/base_models/faster_rcnn
ln -sfnv "$EXPERIMENTS_DIR"/base_models/faster_rcnn faster-rcnn.pytorch/data/pretrained_model
if [ ! -f "faster-rcnn.pytorch/data/pretrained_model/resnet101_caffe.pth" ]; then
	dlcmd https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth -O ./faster-rcnn.pytorch/data/pretrained_model/resnet101_caffe.pth
fi
