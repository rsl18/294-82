# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~

    GB note: No need to do anything. I updated our h4d setup process to create h4d_centernet conda env with all the required dependencies.


1. Install pytorch0.4.1:

    ~~~
    conda install pytorch=0.4.1 torchvision -c pytorch
    ~~~
    
    And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
    
     ~~~
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
     ~~~
     
     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 


    GB Note: it's on line 1254 in our h4d_centernet environment. I added the correct sed command to setup_env.sh, so no need to do anything.
    ~~~
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

    GB note: No need to do anything here, I added this installation to the h4d_centernet conda environment.

3. Clone this repo:

    ~~~
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
    ~~~

    GB note: we have this as a folder in our repo: centernet-master, no need to do anything.

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    GB note: added the requirements to our h4d_centernet conda env, no need to do anything here. Note, had to create a separate environment, parallel to h4d_env because centernet has special requirements like pytorch 0.4.1. Maybe we can upgrade it to torch 1.x later on. It seems to require 0.4.1 because of DCNv2, but there is a version of DCNv2 for torck 1.x, maybe we can switch to that.
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

    GB note: Added this to the setup for the new h4d_centernet conda env, in setup_env.sh.

6. [Optional] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

    ~~~
    cd $CenterNet_ROOT/src/lib/external
    make
    ~~~

    GB: Skipping this for now, don't plan to use multi-scale initially. May revisit later.


7. Download pertained models for [detection]() or [pose estimation]() and move them to `$CenterNet_ROOT/models/`. More models can be found in [Model zoo](MODEL_ZOO.md).
