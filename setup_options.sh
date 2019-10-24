#!/bin/bash

# Set variables here before running script to pre-empt scripts from prompting for user
# input. These vars are "enabled" as long as they are not commented out. To disable,
# comment them out. the 0 or 1 value doesn't act like a boolean:

# Enable headless mode, only high level steps are displayed on command line, all script
# outputs and errors are saved in ./setup_log.txt:
export H4D_HEADLESS_MODE=0

# Tell conda which environment-*.yml to use if it detects nvidia:
export H4D_PLATFORM="gpu"

# Path to where all the data is stored on this computer. 2 options (pick one or the
# other, not both):
#
# Option 1: Local setups usually use relative paths:
# export H4D_DATA_DIR="$(abs_path "../data")"
# export H4D_EXPERIMENTS_DIR="$(abs_path "../experiments")"
# Option 2: BiD machine type setup, where we use absolute paths:
export H4D_DATA_DIR="../data"
export H4D_EXPERIMENTS_DIR="../experiments"

# Shouldn't need to mess with this; derived from earlier values:
export H4D_COCO_DIR="$H4D_DATA_DIR/coco"
export H4D_VOC_DIR="${H4D_DATA_DIR}/voc"

# Lets you disable/enable ./setup_coco.sh
# 1= skip coco setup (big download, so if you do it once probably can set this to zero
# unless ome part of the coco setup process changed since you last ran it)
# 0=don't skip coco setup
export H4D_DISABLE_COCO_SETUP=0
