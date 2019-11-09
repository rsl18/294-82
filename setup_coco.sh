#!/bin/bash
. ./utils.sh
. setup_options.sh

# Make script fail on first error:
set -e

# check if proper vars are set:
if [[ ! "$H4D_COCO_DIR" ]]; then
    echo "H4D_COCO_DIR is not set"
    exit 1
else
    echo "using H4D_COCO_DIR: $H4D_COCO_DIR"
fi

if [[ "$H4D_DISABLE_COCO_SETUP" -eq 1 ]]; then
    echo "H4D_DISABLE_COCO_SETUP was set, skipping setup_coco.sh is skipping."
    exit 0
fi

# If you are copying the coco data from a thumb drive or some other offline location,
# just make sure your $H4D_COCO_DIR has the following structure and run this script. The
# script will bypass the downloads if it sees these files on disk and it will do the
# unzipping for you. $H4D_COCO_DIR
#   - images
#       - train2014.zip
#       - val2014.zip
#   - annotations
#   - annotations_trainval2014.zip

# After setup is done you can throw away the zips if you need to preserve disk space,
# and you should have a structure like this:
#   - images
#       - train2014
#           - *.jpg
#       - val2014
#           - *.jpg
#   - annotations
#       - *.json

# ...But beware that if you delete the zips and run setup again, it will try
#         to download the .zip's again over your internet connection (around 20GB)

function download_and_unzip() {
    local -r DEST_DIR="${1}"
    shift                # shift all args to the left, so we can handle array param:
    declare -a urls=( "${@:2:$1}" ); shift "$(( $1 + 1 ))"
    declare -a extract_dirs=( "${@:2:$1}" ); shift "$(( $1 + 1 ))"

    cd "$DEST_DIR"
    for i in ${!urls[*]}; do
        echo "url: ${urls[$i]}"
        filename="$(basename ${urls[$i]})"
        dest_path="$(pwd)/${filename}"
        echo "filename: ${filename}"
        echo "dest_path: $dest_path"
        if [ ! -f "${dest_path}" ]; then
            echo "Downloading ${urls[$i]}"
            curl -O "${urls[$i]}"
        fi
        # Unzip:
        echo "Unzipping '${dest_path}'"
        if [ ${filename: -4} == ".zip" ]; then
            unzip -qo "${filename}"
        else
            tar xvzf "${filename}" -C "${extract_dirs[$i]}"
        fi
    done
    cd - # this line outputs the name of dir you land in, i'd like to suppress that if possible.
}

# Declare image and annotation URL's, and folder they should extract to (relative):
coco_img_urls=(
    "http://images.cocodataset.org/zips/train2014.zip"
    "http://images.cocodataset.org/zips/val2014.zip"
    "http://images.cocodataset.org/zips/train2017.zip"
    "http://images.cocodataset.org/zips/val2017.zip"
)
coco_img_urls_extract_dirs=("./" "./" "./" "./")
coco_ann_urls=(
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    "https://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/coco_minival2014.tar.gz"
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
coco_ann_urls_extract_dirs=(
    "./"
    "./annotations"
    "./"
)

# Create target folders:
mkdir -p "${H4D_COCO_DIR}/images"
mkdir -p "${H4D_COCO_DIR}/annotations"

# Download:
# Weird technique necessary to pass two arrays to a function for bash < 4.3:
# https://stackoverflow.com/questions/10953833/passing-multiple-distinct-arrays-to-a-shell-function
download_and_unzip "$H4D_COCO_DIR/images" \
    "${#coco_img_urls[@]}" "${coco_img_urls[@]}" \
    "${#coco_img_urls_extract_dirs[@]}" "${coco_img_urls_extract_dirs[@]}"
download_and_unzip "$H4D_COCO_DIR" \
    "${#coco_ann_urls[@]}" "${coco_ann_urls[@]}" \
    "${#coco_ann_urls_extract_dirs[@]}" "${coco_ann_urls_extract_dirs[@]}"