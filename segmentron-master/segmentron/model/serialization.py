import os
import sys
import pickle
import re

import torch
from torch.utils import model_zoo


FAIR_URLS = {
    50: "https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/"
        "MSRA/R-50.pkl",
    101: "https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/"
         "MSRA/R-101.pkl",
}


def load_fair_params(depth):
    fair_state = load_fair_url(FAIR_URLS[depth])
    fair_keys = sorted(fair_state.keys())
    model_keys = parse_fair_as_torchvision(fair_keys)
    model_state = {model_key: torch.from_numpy(fair_state[fair_key])
                   for model_key, fair_key in zip(model_keys, fair_keys)}
    return model_state


def load_fair_url(url, model_dir=None, progress=True):
    # note: like torch.utils.model_zoo.load_url but
    # - hash is disabled
    # - loads pickle format
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO',
                              os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = model_zoo.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        model_zoo._download_url_to_file(url, cached_file, hash_prefix,
                                        progress=progress)
    return pickle.load(open(cached_file, 'rb'), encoding='latin')


def parse_fair_as_torchvision(keys):
    # translate convention from Caffe2 to PyTorch
    keys = [k.replace('_', '.') for k in keys]
    keys = [re.sub('\.w$', '.weight', k) for k in keys]
    keys = [re.sub('\.b$', '.bias', k) for k in keys]

    # batch norm statistics and affine parameters are collapsed into combined
    # scale/weight and shift/bias parameters, and the scale 's' -> 'weight'
    keys = [re.sub('\.s$', '.weight', k) for k in keys]

    # make ResNet names compatible with torchvision
    keys = [k.replace('res.conv1.', 'conv1.') for k in keys]

    keys = [k.replace('res2.', 'layer1.') for k in keys]
    keys = [k.replace('res3.', 'layer2.') for k in keys]
    keys = [k.replace('res4.', 'layer3.') for k in keys]
    keys = [k.replace('res5.', 'layer4.') for k in keys]

    keys = [k.replace('.branch2a.', '.conv1.') for k in keys]
    keys = [k.replace('.branch2b.', '.conv2.') for k in keys]
    keys = [k.replace('.branch2c.', '.conv3.') for k in keys]

    keys = [re.sub('conv(\d)\.bn\.', r'bn\1.', k) for k in keys]

    keys = [k.replace('.branch1.bn.', '.downsample.1.') for k in keys]
    keys = [k.replace('.branch1.', '.downsample.0.') for k in keys]

    return keys
