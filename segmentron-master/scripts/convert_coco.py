import os

import click
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


IGNORE = 255

@click.command()
@click.argument('coco_dir', type=click.Path(exists=True))
@click.argument('split', type=click.Choice(('train', 'val')))
@click.argument('output', type=click.Path())
@click.option('--ignore_crowd', is_flag=True)
def convert_json_to_seg(coco_dir, split, output, ignore_crowd):
    """Convert COCO dataset masks from JSON format into semantic segmentation
    targets in the style of PASCAL VOC. Note that ignoring crowd is advised,
    since these annotations can be somewhat erratic.

    Take
        coco_dir: path to the COCO dataset dir
        split: the split to convert
        output: the directory to save the target images (PNG)
        ignore_crowd: mark crowd annotations for ignoring (True), or keep them
    """
    print(f"converting COCO {split} at {coco_dir}")
    coco = COCO(f'{coco_dir}/annotations/instances_{split}2014.json')
    coco_to_target_mapping = {v: i for i, v in enumerate(coco.getCatIds(), 1)}

    os.makedirs(output, exist_ok=True)

    image_ids = coco.getImgIds()
    for i, im_idx in enumerate(image_ids, 1):
        if i % 1000 == 0:
            print(f"converted {i}/{len(image_ids)} target images...")
        im = coco.loadImgs(im_idx)[0]
        # 0: background
        seg = np.zeros((im['height'], im['width']), dtype=np.uint8)
        used = seg.copy()
        annos = coco.loadAnns(coco.getAnnIds(imgIds=im_idx))
        for anno in annos:
            mask = coco.annToMask(anno) == 1
            mask = np.logical_and(mask, np.logical_not(used))
            class_ = coco_to_target_mapping[anno['category_id']]
            if ignore_crowd and anno['iscrowd']:
                class_ = IGNORE
            seg[mask] = class_
            used = np.logical_or(used, mask)
        seg_file = im['file_name'].replace('jpg', 'png').replace("jpeg", "png")
        Image.fromarray(seg).save(f'{output}/{seg_file}')
    print(f"converted {i} target images... done!")


if __name__ == '__main__':
    convert_json_to_seg()
