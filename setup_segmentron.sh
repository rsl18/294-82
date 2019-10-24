#!/bin/bash

function abs_path {
	cd "$1"
	loc="$(pwd)"
	echo "$loc"
}

#relative path for local repo
DATA_DIR=$(abs_path "$DATA_DIR")
EXPERIMENTS_DIR=$(abs_path "$EXPERIMENTS_DIR")

# Expand relative paths to absolute ones (required for symlink creation to work):
export DATA_DIR=$(abs_path "$DATA_DIR")
export EXPERIMENTS_DIR=$(abs_path "$EXPERIMENTS_DIR")

echo "Are you setting up a local repo on the BID machine (y/n)"
read REPO

#absolute path for bid repos
if [ "$REPO" = "y" ]; then
	#if server
	echo "BID machine, ok, using absolute paths"
	DATA_DIR="/home/data"
	EXPERIMENTS_DIR="/home/experiments"
fi

echo "Using DATA_DIR: " $DATA_DIR
echo "Using EXPERIMENTS_DIR: " $EXPERIMENTS_DIR

# Note: this creates a broken symlink right now because sec_coco_chipped folder doesn't
# currently get created by the setup_datasets.sh script.
mkdir -p segmentron-master/data/coco
ln -sfn "$DATA_DIR"/Xview/seg_coco_chipped segmentron-master/data/coco

mkdir -p "$EXPERIMENTS_DIR"/segmentron
ln -sfn "$EXPERIMENTS_DIR"/segmentron segmentron-master/experiments

