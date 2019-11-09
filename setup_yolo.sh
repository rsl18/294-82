#!/bin/bash
. utils.sh

function abs_path() {
    if [ ! -d "$1" ]; then
        echo "Relative path doesn't exist: ${1}"
        exit 1
    fi
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

mkdir -p "$EXPERIMENTS_DIR"/base_models/yolo
ln -sfn "$EXPERIMENTS_DIR"/base_models/yolo yolov3/weights

# Download weights for vanilla YOLOv3
# dlcmd calls either wget or curl with the appropriate option based on OS as defined at top of file
if [ ! -f "yolov3/weights/yolov3.weights" ]; then
    dlcmd https://pjreddie.com/media/files/yolov3.weights -O yolov3/weights/yolov3.weights
fi
# # Download weights for tiny YOLOv3
if [ ! -f "yolov3/weights/yolov3-tiny.weights" ]; then
    dlcmd https://pjreddie.com/media/files/yolov3-tiny.weights -O yolov3/weights/yolov3-tiny.weights
fi
# Download weights for backbone network
if [ ! -f "yolov3/weights/darknet53.conv.74" ]; then
    dlcmd https://pjreddie.com/media/files/darknet53.conv.74 -O yolov3/weights/darknet53.conv.74
fi

bash yolov3/config/create_custom_model.sh $(wc -l yolov3/data/xview_coco_$DATA_TARGET_VERSION.names | awk '{ print $1 }') 512

mkdir -p "$EXPERIMENTS_DIR"/yolo

# Point the coco folders to h4d_data_dir:
mkdir -p ./yolov3/data/coco
COCO_DIR="$DATA_DIR/coco"
ln -sfnv "$COCO_DIR/images" ./yolov3/data/coco/images
ln -sfnv "$COCO_DIR/annotations" ./yolov3/data/coco/annotations

# Don't need to download this file, it's part of MS COCO dataset, which we download in
# setup_coco.sh:
# # if [ ! -f instances_train-val2014.zip ]; then
# # dlcmd https://pjreddie.com/media/files/instances_train-val2014.zip -O ./yolov3/data/coco/instances_train-val2014.zip
# # fi

# Download additional VOC-formatted annotations for MS COCO:
if [ ! -f ./yolov3/data/coco/5k.part ]; then
    dlcmd https://pjreddie.com/media/files/coco/5k.part -O ./yolov3/data/coco/5k.part
fi
if [ ! -f ./yolov3/data/coco/trainvalno5k.part ]; then
    dlcmd https://pjreddie.com/media/files/coco/trainvalno5k.part -O ./yolov3/data/coco/trainvalno5k.part
fi
if [ ! -f ./yolov3/data/coco/labels.tgz ]; then
    dlcmd https://pjreddie.com/media/files/coco/labels.tgz -O ./yolov3/data/coco/labels.tgz
fi
tar xzf ./yolov3/data/coco/labels.tgz --directory ./yolov3/data/coco/

# Set Up Image Lists:
yolo_coco_dir="$(abs_path "./yolov3/data/coco")"
echo "yolo_coco_dir: $yolo_coco_dir"
paste <(awk "{print \"$yolo_coco_dir\"}" <"$yolo_coco_dir/5k.part") "$yolo_coco_dir/5k.part" | tr -d '\t' >"$yolo_coco_dir/5k.txt"
paste <(awk "{print \"$yolo_coco_dir\"}" <"$yolo_coco_dir/trainvalno5k.part") "$yolo_coco_dir/trainvalno5k.part" | tr -d '\t' >"$yolo_coco_dir/trainvalno5k.txt"

echo "setup_yolo.sh finished"
