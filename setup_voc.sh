#!/bin/bash
. ./utils.sh
. setup_options.sh

# Make script fail on first error:
set -e

# check if proper vars are set:
if [[ ! "$H4D_VOC_DIR" ]]; then
    echo "H4D_VOC_DIR is not set"
    exit 1
else
    echo "using H4D_VOC_DIR: $H4D_VOC_DIR"
fi

if [[ "$H4D_DISABLE_VOC_SETUP" -eq 1 ]]; then
    echo "H4D_DISABLE_VOC_SETUP was set, skipping setup_voc.sh is skipping."
    exit 0
fi

# TODO: Update these comments for VOC (this file was created based off of setup_coco.sh,
# so these instructions haven't been updated yet):

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
            tar xvf "${filename}" -C "${extract_dirs[$i]}"
        fi
    done
    cd - # this line outputs the name of dir you land in, i'd like to suppress that if possible.
}

# Declare image and annotation URL's, and folder they should extract to (relative):
voc_urls=(
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar"
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar"
)
voc_urls_extract_dirs=("./" "./" "./" "./" "./")

# Create target folders:
mkdir -p "${H4D_VOC_DIR}"

# Download:
# Weird technique necessary to pass two arrays to a function for bash < 4.3:
# https://stackoverflow.com/questions/10953833/passing-multiple-distinct-arrays-to-a-shell-function
download_and_unzip "$H4D_VOC_DIR" \
    "${#voc_urls[@]}" "${voc_urls[@]}" \
    "${#voc_urls_extract_dirs[@]}" "${voc_urls_extract_dirs[@]}"

# VOC Combined (primarily for centernet, but maybe used by others later):
if [ -d "${H4D_VOC_DIR}/voc_combined" ]; then
    rm -rf "${H4D_VOC_DIR}/voc_combined";
fi
mkdir -p "${H4D_VOC_DIR}/voc_combined/images"
cd "${H4D_VOC_DIR}/voc_combined"
cp "${H4D_VOC_DIR}"/VOCdevkit/VOC2007/JPEGImages/* "${H4D_VOC_DIR}/voc_combined/images/"
cp "${H4D_VOC_DIR}"/VOCdevkit/VOC2012/JPEGImages/* "${H4D_VOC_DIR}/voc_combined/images/"
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
unzip PASCAL_VOC.zip
rm PASCAL_VOC.zip
mv PASCAL_VOC annotations/
cd -

CENTERNET_ENV="h4d_env"
eval "$(conda shell.bash hook)"
conda activate $CENTERNET_ENV
conda info --envs
python centernet-master/src/tools/merge_pascal_json.py
conda deactivate

# Separate VOC's (for SSD):
rm -rf "${H4D_VOC_DIR}/VOC2012"
rm -rf "${H4D_VOC_DIR}/VOC2007"
mkdir -p "${H4D_VOC_DIR}/VOC2012"
mkdir -p "${H4D_VOC_DIR}/VOC2007"
cp -r "${H4D_VOC_DIR}"/VOCdevkit/VOC2012/* "${H4D_VOC_DIR}/VOC2012"
cp -r "${H4D_VOC_DIR}"/VOCdevkit/VOC2007/* "${H4D_VOC_DIR}/VOC2007"

# VOCdevkit is needed by centernet:
mv "${H4D_VOC_DIR}"/VOCdevkit "${H4D_VOC_DIR}/voc_combined/VOCdevkit"