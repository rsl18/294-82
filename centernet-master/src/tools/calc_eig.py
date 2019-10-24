# Standard Library imports:
import argparse
from pathlib import Path

# 3rd Party imports:
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from pycocotools.coco import COCO
import skimage.io as io
import skimage.transform as transform
from tqdm import tqdm

# 1st Party imports:
import _init_paths

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: E402, E501, F401
from h4dlib.config import h4dconfig


def get_img_paths(coco, dataDir, split, dataset):
    img_ids = coco.getImgIds()
    print("num images: ", len(img_ids))
    img_infos = [coco.loadImgs(img_id)[0] for img_id in img_ids]
    if dataset == "coco":
        img_paths = [
            dataDir / f"images/{split}/{img_info['file_name']}" for img_info in img_infos
        ]
    else:
        img_paths = [
            dataDir / f"{split}/{img_info['file_name']}" for img_info in img_infos
        ]
    return img_paths


def get_channel_means(img_paths, resize=False, img_size=512):
    """Calculates per-channel means for set of image paths.
    Performs both styles of mean calculation at once:
    get_channel_means_broadcast_gray() and get_channel_means_skip()

    Args:
        img_paths ([type]): [description]
        resize (bool, optional): [description]. Defaults to False.
        img_size (int, optional): [description]. Defaults to 512.

    Returns:
        [type]: [description]
    """
    total_broadcast_gray = np.zeros(3)
    num_images_broadcast_gray = 0
    total_skip = np.zeros(3)
    num_images_skip = 0
    for img_path in tqdm(img_paths):
        if resize:
            img = io.imread(img_path)
            img = (
                cv2.resize(
                    img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC
                )
                / 255.0
            )
        else:
            img = io.imread(img_path).astype(np.float) / 255.0
        if len(img.shape) == 3:
            assert img.shape[2] == 3, "Image should be three channels"
            img_means = (img.sum(axis=0).sum(axis=0)) / (img.shape[0] * img.shape[1])
            total_broadcast_gray += img_means
            num_images_broadcast_gray += 1
            total_skip += img_means
            num_images_skip += 1
        else:
            img_means = img.sum() / (img.shape[0] * img.shape[1])
            total_broadcast_gray += img_means
            num_images_broadcast_gray += 1
            # print(img_path, img.shape)
    print(
        "broadcast_gray: ",
        total_broadcast_gray / num_images_broadcast_gray,
        total_broadcast_gray,
        num_images_broadcast_gray,
    )
    print("skip: ", total_skip / num_images_skip, total_skip, num_images_skip)
    return (
        total_broadcast_gray / num_images_broadcast_gray,
        total_broadcast_gray,
        num_images_broadcast_gray,
    )


def get_channel_means_broadcast_gray(img_paths, resize=False, img_size=512):
    """Calculates per-channel means for set of image paths. Broadcasts the single
    channel pixel means to all 3 channels in case of single-channel images.
    Effectively same as converting the image to 3-channel grayscale before including
    it in the population mean.

    Args:
        img_paths ([type]): [description]
        resize (bool, optional): [description]. Defaults to False.
        img_size (int, optional): [description]. Defaults to 512.

    Returns:
        [type]: [description]
    """
    total = np.zeros(3)
    num_images = 0
    for img_path in tqdm(img_paths):
        if resize:
            img = io.imread(img_path)
            img = (
                cv2.resize(
                    img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC
                )
                / 255.0
            )
        else:
            img = io.imread(img_path).astype(np.float) / 255.0
        if len(img.shape) == 3:
            assert img.shape[2] == 3, "Image should be three channels"
            img_means = (img.sum(axis=0).sum(axis=0)) / (img.shape[0] * img.shape[1])
            total += img_means
            num_images += 1
        else:
            img_means = img.sum() / (img.shape[0] * img.shape[1])
            total += img_means
            num_images += 1
            # print(img_path, img.shape)
    return total / num_images, total, num_images


def get_channel_means_skip(img_paths, resize=False, img_size=512):
    """Calculates per-channel means for set of image paths. Skips any images that are
    not 3-channel

    Args:
        img_paths ([type]): [description]
        resize (bool, optional): [description]. Defaults to False.
        img_size (int, optional): [description]. Defaults to 512.

    Returns:
        [type]: [description]
    """
    total = np.zeros(3)
    num_images = 0
    for img_path in tqdm(img_paths):
        if resize:
            img = io.imread(img_path)
            img = (
                cv2.resize(
                    img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC
                )
                / 255.0
            )
        else:
            img = io.imread(img_path).astype(np.float) / 255.0
        if len(img.shape) == 3:
            assert img.shape[2] == 3, "Image should be three channels"
            img_means = (img.sum(axis=0).sum(axis=0)) / (img.shape[0] * img.shape[1])
            total += img_means
            num_images += 1
    return total / num_images, total, num_images


def fancyPCA(img_paths, img_size=256):
    """
    Calculates 3 eigenvectors/values according to Fancy_PCA. Not really working if
    we try to do on entirety of MSCOCO2017, uses too much memory.

    Turns out CenterNet uses the same eigen-v's for all four datasets, so we'll skip
    having to call this function at all, just reuse same eigen-v's that CenterNet
    is using for the other datsets.
    """
    # Calculate Fancy PCA
    # Using six sample images.
    img_paths = img_paths[:50000]

    # Read collection of images with imread_collection
    imlist = io.imread_collection(img_paths)

    # Convert all images to standard size:
    # imglist_resized = []
    # for i in range(len(imlist)):
    #     # Using the skimage.transform function-- resize image (m x n x dim).
    #     m = transform.resize(imlist[i], (img_size, img_size, 3))
    #     imglist_resized.append(m)

    # initializing with zeros.
    res = np.zeros(shape=(1, 3))

    print("Resizing images to: ", img_size)
    imlist_resized = []
    for i in tqdm(range(len(imlist))):
        m = transform.resize(imlist[i], (img_size, img_size, 3))
        # Reshape the matrix to a list of rgb values.
        arr = m.reshape((img_size * img_size), 3)
        imlist_resized.append(arr)
        # # concatenate the vectors for every image with the existing list.
        # res = np.concatenate((res, arr), axis=0)
    # # delete initial zeros' row:
    # res = np.delete(res, (0), axis=0)

    # print list of vectors - 3 columns (rgb)
    # res = np.stack(imlist_resized, axis=0)
    # res = np.array(imlist_resized)
    res = np.concatenate(imlist_resized, axis=0)

    print(res.shape)
    # print(res)

    # Subtract means:
    print("res.shape: ", res.shape)
    m = res.mean(axis=0)
    print("mean: ", m)
    res = res - m
    print("res: ", res)
    R = np.cov(res, rowvar=False)
    print(R)

    # Calculate eigenvectors and values:
    evals, evecs = LA.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # sort eigenvectors according to same index
    evals = evals[idx]

    # select the best 3 eigenvectors (3 is desired dimension
    # of rescaled data array)
    evecs = evecs[:, :3]

    # make a matrix with the three eigenvectors as its columns.
    evecs_mat = np.column_stack((evecs))

    print("eigenvalues: ", evals)
    print("eigenvectors: ", evecs_mat)
    return evals, evecs_mat


# def normalize_meanstd(a, axis=None):
#     # axis param denotes axes along which mean & std reductions are to be performed
#     mean = np.mean(a, axis=axis, keepdims=True)
#     std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
#     return mean, std


# def calc_rgb_means_and_std(img_paths, img_size):
#     # imlist = io.imread_collection(img_paths)
#     res = np.zeros(shape=(1, 3))

#     print("Resizing images to: ", img_size)
#     imlist_resized = []
#     for img in tqdm(img_paths):
#         m = transform.resize(io.imread(img), (img_size, img_size, 3))
#         # Reshape the matrix to a list of rgb values.
#         arr = m.reshape((img_size * img_size), 3)
#         imlist_resized.append(arr)
#     res = np.concatenate(imlist_resized, axis=0)
#     print("res.shape: ", res.shape)
#     return normalize_meanstd(res, axis=(0))


# def calc_rgb_means_and_std2(img_paths, img_size):
#     imlist_resized = []
#     for img in tqdm(img_paths):
#         m = transform.resize(io.imread(img), (img_size, img_size, 3))
#         imlist_resized.append(m)
#     res = np.concatenate(imlist_resized, axis=0)
#     print("res.shape: ", res.shape)
#     return normalize_meanstd(res, axis=(0, 1))


class StatsRecorder:
    """Class that performs per-channel (RGB) mean and stdev calculation.
    Computation is done online, one image at a time, thus memory efficient to compute
    RGB means/stdev for an entire dataset such as MSCOCO.

    """
    def __init__(self, data=None, img_size=512):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        self.img_size = img_size
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, img_path):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        data = transform.resize(
            io.imread(img_path), (self.img_size, self.img_size, 3)
        ).reshape(self.img_size * self.img_size, 3)
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)
            m = self.nobservations * 1.0
            n = data.shape[0]
            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = (
                m / (m + n) * self.std ** 2
                + n / (m + n) * newstd ** 2
                + m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            )
            self.std = np.sqrt(self.std)
            self.nobservations += n


def get_rgb_means_and_stdev(img_paths, img_size):
    """
    Calculate per-channel means and standard deviations for entire set of specified
    image paths.
    """
    mystats = StatsRecorder()
    for img in tqdm(img_paths):
        mystats.update(img)
    # Reorder status from RGB order to BGR, since data loader uses OpenCV
    # and Centernet seems to expect BGR order:
    return np.flip(mystats.mean), np.flip(mystats.std)


def main(opt):
    ##
    ## Get COCO:
    if opt.dataset == "coco":
        dataDir = h4dconfig.DATA_DIR / "coco"
        split = "train2017"
        annFile = dataDir / f"annotations/instances_{split}.json"
        print("dataDir: ", dataDir)
        print("annFile: ", annFile)
        coco = COCO(annFile)
    elif opt.dataset == "xview":
        dataDir = h4dconfig.DATA_DIR / "Xview/coco_chipped"
        split = "train"
        annFile = dataDir / f"xview_coco_v2_{split}_chipped.json"
        print("dataDir: ", dataDir)
        print("annFile: ", annFile)
        coco = COCO(annFile)
    ##
    ## Get image paths:
    img_paths = get_img_paths(coco, dataDir, split, opt.dataset)

    ##
    ## Calculate RGB means:
    # print(get_rgb_means_and_stdev(img_paths, img_size=256))
    print(get_rgb_means_and_stdev(img_paths, img_size=512))

    # print(calc_rgb_means_and_std2(img_paths, img_size=256))
    # print(calc_rgb_means_and_std2(img_paths, img_size=512))
    # means, total, num_images = get_channel_means(img_paths)
    # print("Final means: ", means)
    # means, total, num_images = get_channel_means(img_paths, resize=True, img_size=256)
    # print("Final means (resize): ", means)

    # means, total, num_images = get_channel_means_broadcast_gray(img_paths)
    # print("Final means: ", means)
    # means, total, num_images = get_channel_means_skip(img_paths)
    # print("Final means: ", means)
    # means, total, num_images = get_channel_means_broadcast_gray(
    #     img_paths, resize=True, img_size=512
    # )
    # print("Final means: ", means)
    # means, total, num_images = get_channel_means_skip(
    #     img_paths, resize=True, img_size=512
    # )
    # print("Final means: ", means)

    # Calculate eigen-stuff:
    # evals, evecs = fancyPCA(img_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="xview",
        help="coco | xview",
    )
    opt = parser.parse_args()
    main(opt)
