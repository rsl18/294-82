#!/bin/bash

# Force exit on first error:
set -e

# use environment-gpu.yml if you have nvidia gpu's on your system
# echo "If you have Nvidia GPU(s) on your system type 'gpu', else type 'cpu'"
# read platform

. utils.sh

# source /home/gbiamby/anaconda3/etc/profile.d/conda.sh

function program_exists() {
    command -v "$1" >/dev/null 2>&1
}

function get_platform() {
# removed local declaration of outvar_platform to avoid bash version issues with -n flag
    outvar_platform=$1 # use an output var
    outvar_platform="cpu"
    set +e
    if program_exists nvcc; then
        if [[ "$H4D_PLATFORM" ]]; then
            outvar_platform="$H4D_PLATFORM"
        else
            ask_for_confirmation \
                "CUDA detected! Do you want to setup the conda environment with GPU support? (recommended)"
            if answer_is_yes; then
                outvar_platform="gpu"
            fi
        fi
    fi
    platform=$outvar_platform
    set -e
}

platform="gpu"
get_platform $platform

echo "Installing env. for platform: $platform"

if [[ "${platform}" == "cpu" ]]; then
    echo "using cpu env"
    OS="`uname`"
    cp environment-cpu.yml{,.bak}
    if [[ "${OS}" == "Linux" ]]; then
        echo "Using linux-cpu mode"
        # There is no '-cpu' version of pytorch or torchvision for macos, but for the CPU
        # env on linux we need to use '-cpu' versions of those packages, so replace that in the yml:
        sed -i 's/torchvision=/torchvision-cpu=/g' environment-cpu.yml
        sed -i 's/pytorch=/pytorch-cpu=/g' environment-cpu.yml

    else
        echo "Using mac-cpu mode"
        # Mac requires that libgcc-ng and libstdcxx-ng be removed altogether
        sed -i '' '/libgcc-ng/g' environment-cpu.yml    #'should' delete the dash with the text, might cause new line
        sed -i '' '/libstdcxx-ng/g' environment-cpu.yml #'should' delete the dash with the text, might cause new line
    fi
fi

# Delete and recreate conda env:
eval "$(conda shell.bash hook)"
conda deactivate
conda env remove -n h4d_env
conda env create -f environment-$platform.yml
if [[ "${platform}" == "cpu" ]]; then
    mv environment-cpu.yml.bak environment-cpu.yml
fi

# Enable using the h4d_env conda environment from jupyter notebooks:
conda activate h4d_env
python -m ipykernel install --user --name h4d_env --display-name "h4d_env"
# Install pycocotools:
cd cocoapi/PythonAPI
python3 setup.py build_ext install
cd ../..
# Install def_conv:
cd vendor/def_conv
python3 setup.py build_ext install
cd -

##
## Centernet env:
if [[ "${platform}" == "gpu" ]]; then
    ##
    ## Hardcoding this is hacky, but for now I just want to see if this can work. Ideally we'd detech the torch version
    ## in the centernet conda environment programmatically.
    TORCH_VERSION="1.1"

    echo "Installing h4d_centernet conda environment"
    eval "$(conda shell.bash hook)"
    conda deactivate
    conda info --envs

    ## Revert the batch_norm disable hack, to allow conda env setup to work without errors:
    if [[ -f ~/anaconda3/envs/h4d_centernet/lib/python3.7/site-packages/torch/nn/functional.py ]]; then
        if [[ "${TORCH_VERSION}" == "0.4" ]]; then
        sed -i "1254s/False/torch\.backends\.cudnn\.enabled/g" ~/anaconda3/envs/h4d_centernet/lib/python3.7/site-packages/torch/nn/functional.py
    fi
        if [[ "${TORCH_VERSION}" == "1.1" ]]; then
            sed -i "1697s/False/torch\.backends\.cudnn\.enabled/g" ~/anaconda3/envs/h4d_centernet/lib/python3.7/site-packages/torch/nn/functional.py
        fi
    fi

    # Remove and re-create h4d_centernet conda env:
    CENTERNET_ENV="h4d_env"
    echo "Using centernet environment: ${CENTERNET_ENV}"
    if [[ "$CENTERNET_ENV" == "h4d_centernet" ]]; then
        conda env remove -n $CENTERNET_ENV
        conda env create -f environment-centernet-gpu.yml
        conda activate h4d_centernet

        ## Enable using h4d_centernet conda environment from jupyter notebooks:
        python -m ipykernel install --user --name h4d_centernet --display-name "h4d_centernet"
    fi

    ## Install pycocotools:
    cd cocoapi/PythonAPI
    python3 setup.py build_ext install
    cd -

    ## Compile NMS? It wasn't in current CenterNet readme, so I'm not enabling it if we're running with torch 0.4. We can un-comment-out this
    ## conditional block if we end up going with 0.4 and find out that we need to compile NMS:
    # if [[ "${TORCH_VERSION}" == "0.4" ]]; then
    #     cd centernet-master\src\lib\external
    #     python setup.py build_ext --inplace
    #     cd -
    # fi
    ## ...but another resource about getting CenterNet to run on torch 1.1 / CUDA 10 mentioned this step:
    ##  (see: https://www.gitmemory.com/issue/xingyizhou/CenterNet/7/486653333)
    if [[ "${TORCH_VERSION}" == "1.1" ]]; then
        cd centernet-master/src/lib/external
        python setup.py build_ext --inplace
        cd -
    fi

    ## Install DCNv2:
    if [[ "${TORCH_VERSION}" == "0.4" ]]; then
        cd centernet-master/src/lib/models/networks/DCNv2
        ./make.sh
        cd -
    fi
    if [[ "${TORCH_VERSION}" == "1.1" ]]; then
        ## Note: if we get torch 1.1 to work reliably we can update the repo to include the 1.1 version of DCNv2 instead
        ## of removing the old one and cloning the newer one:
        rm -rf centernet-master/src/lib/models/networks/DCNv2
        git clone https://github.com/CharlesShang/DCNv2.git centernet-master/src/lib/models/networks/DCNv2
        cd centernet-master/src/lib/models/networks/DCNv2
        python setup.py build develop
        cd -
    fi

    ## Enable batch_norm disable hack:
    # Changes functional.py line 1254; the last param should be "False",
    # instead of "torch.backends.cudnn.enabled" (for more details see centernet-master/INSTALL.md, part 1):
    if [[ -f ~/anaconda3/envs/h4d_env/lib/python3.7/site-packages/torch/nn/functional.py ]]; then
        if [[ "${TORCH_VERSION}" == "0.4" ]]; then
            sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ~/anaconda3/envs/h4d_env/lib/python3.7/site-packages/torch/nn/functional.py
        fi
    else
        if [[ "${TORCH_VERSION}" == "1.1" ]]; then
            sed -i "1697s/torch\.backends\.cudnn\.enabled/False/g" ~/anaconda3/envs/h4d_env/lib/python3.7/site-packages/torch/nn/functional.py
        fi
    fi
else
    echo "WARNING: Skipped h4d_centernet environment creation because platform is CPU. Check setup_options.sh if this machine has a GPU."
fi
