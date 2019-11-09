#!/bin/bash

set -e
eval "$(conda shell.bash hook)"
conda activate h4d_env

echo "setup datasets"

function abs_path {
	cd "$1"
	loc="$(pwd)"
	echo "$loc"
}

DATA_TARGET_VERSION="v2"

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


ln -sfn "$DATA_DIR" datasets
ln -sfn "$EXPERIMENTS_DIR" experiments

# XVIEW-COCO CHIPS

# Creates xview_coco_v0.json
if [ ! -f "$DATA_DIR"/Xview/coco_vanilla/xview_coco_v0.json ]; then
	echo "Converting from geojson to coco format."
	python ./scripts/geojson_to_json.py
else
	echo "Skipping Xview->COCO, v0 json exists already"
fi

# Creates xview_coco_v2.json
if [ ! -f "$DATA_DIR"/Xview/coco_vanilla/xview_coco_"$DATA_TARGET_VERSION".json ]; then
	echo "Remapping xview classes."
	cp ./xview_"$DATA_TARGET_VERSION".csv "$DATA_DIR"/Xview/coco_vanilla/
	cd scripts
	python ./create_remap.py
	cd ..
else
	echo "Skipping remap - $DATA_TARGET_VERSION json exists already"
fi

# Creates xview_coco_v2_train.json and xview_coco_v2_val.json:
if [ ! -f "$DATA_DIR"/Xview/coco_vanilla/xview_coco_"$DATA_TARGET_VERSION"_val.json ]; then
	echo "Creating train/val splits."
	python ./scripts/trainvalsplit.py
else
	echo "Skipping train/val split - train/val jsons exist already"
fi

# Creates xview_coco_v2_{split}_chipped.json's
if [[ ! -f "$DATA_DIR"/Xview/coco_chipped/xview_coco_"$DATA_TARGET_VERSION"_val_chipped.json \
		&& ! -f "$DATA_DIR"/Xview/coco_chipped/xview_coco_"$DATA_TARGET_VERSION"_train_chipped.json ]]; then
	echo "Chipping images."
	python ./scripts/coco_chip.py
	python ./scripts/create_tiny_dataset.py --dataset_name xview_coco_"$DATA_TARGET_VERSION"
else
	echo "Skipping chipping - coco_chipped folder exists already"
fi

# # # XVIEW-COCO SEGMENTATIONS
# if [ ! -d $DATA_DIR/Xview/seg_coco_chipped ]; then
# 	mkdir -p $DATA_DIR/Xview/seg_coco_chipped/annotations
# 	ln -sfn $DATA_DIR/Xview/coco_chipped/xview_coco_v1_train_chipped.json $DATA_DIR/Xview/seg_coco_chipped/annotations/xview_coco_v1_train_chipped.json
# 	ln -sfn $DATA_DIR/Xview/coco_chipped/xview_coco_v1_val_chipped.json $DATA_DIR/Xview/seg_coco_chipped/annotations/xview_coco_v1_val_chipped.json
# 	ln -sfn $DATA_DIR/Xview/coco_chipped/train $DATA_DIR/Xview/seg_coco_chipped/train
# 	ln -sfn $DATA_DIR/Xview/coco_chipped/val $DATA_DIR/Xview/seg_coco_chipped/val

# 	if [ ! -d "segmentron-master/scripts/coco" ]; then
# 		git clone https://github.com/cocodataset/cocoapi.git segmentron-master/scripts/coco
# 	fi
# 	rm -rf segmentron-master/scripts/coco/.git
# 	make -C segmentron-master/scripts/coco/PythonAPI
# 	rm -rf segmentron-master/scripts/pycocotools
# 	mv segmentron-master/scripts/coco/PythonAPI/pycocotools segmentron-master/scripts/pycocotools
# 	rm -rf segmentron-master/scripts/coco

# 	python segmentron-master/scripts/convert_coco.py $DATA_DIR/Xview/coco_chipped/ 'val' $DATA_DIR/Xview/seg_coco_chipped/annotations/seg_val
# 	python segmentron-master/scripts/convert_coco.py $DATA_DIR/Xview/coco_chipped/ 'train' $DATA_DIR/Xview/seg_coco_chipped/annotations/seg_train
# 	ls $DATA_DIR/Xview/coco_chipped/train >  $DATA_DIR/Xview/seg_coco_chipped/train.txt
# 	ls $DATA_DIR/Xview/coco_chipped/val >  $DATA_DIR/Xview/seg_coco_chipped/val.txt
# fi

# # COCO_TO_YOLO
# if [ ! -f "yolov3/data/xview_coco_${DATA_TARGET_VERSION}/train.txt" ] || [ ! -f "yolov3/data/xview_coco_${DATA_TARGET_VERSION}_tiny/train.txt" ]; then
# 	echo "Converting xview_coco_${DATA_TARGET_VERSION} to VOC format for use by yolov3."
# 	mkdir -p yolov3/data/xview_coco_"$DATA_TARGET_VERSION"

# 	ln -sfn "$DATA_DIR"/Xview/coco_chipped yolov3/data/xview_coco_"$DATA_TARGET_VERSION"/annotations
# 	ln -sfn "$DATA_DIR"/Xview/coco_chipped yolov3/data/xview_coco_"$DATA_TARGET_VERSION"/images
# 	mkdir -p yolov3/data/xview_coco_"$DATA_TARGET_VERSION"/labels

# 	python scripts/coco_to_yolo.py --dataset_name "xview_coco_${DATA_TARGET_VERSION}"
# else
# 	echo "Skipping yolo dataset setup - yolov3/data/xview_coco_${DATA_TARGET_VERSION} exists already"
# fi
