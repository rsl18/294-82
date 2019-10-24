#!/bin/bash

TIMESTAMP=`date +%F_%T`
ARCH='simple'
BACKBONE='res101'
HEAD='k1'
DATASET='coco'
MAX_EPOCH=200
SPLIT='train'
PREPROC='torch'
GPU=0
OUTPUT='experiments'

python train.py --user $USER --timestamp $TIMESTAMP --arch $ARCH --backbone $BACKBONE --head $HEAD --dataset $DATASET --max-epoch $MAX_EPOCH --split $SPLIT --preproc $PREPROC --gpu $GPU --output $OUTPUT
