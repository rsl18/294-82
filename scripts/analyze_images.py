from PIL import Image
from collections import namedtuple
from pathlib import Path
import os
from typing import Any
import numpy as np
import sys


def analyze(DS: str) -> None:
    """
    Prints various data about the images located in the path specified.
    Returns None.
    :param DS: Absolute path of (flattened) folder where images are stored to analyze (DS - dataset).
    """
    os.chdir(DS)
    total_imgs = 0
    smallest_width = float("inf")
    smallest_height = float("inf")
    largest_width = float("-inf")
    largest_height = float("-inf")
    widths = []
    heights = []
    smallest_img_area = float("inf")
    largest_img_area = float("-inf")
    areas = []

    for dirpath, dirnames, files in os.walk(DS):
        for file_name in files:
            if not file_name[0] == ".":
                im = Image.open(file_name)
                width, height = im.size
                smallest_width = width if width < smallest_width else smallest_width
                smallest_height = (
                    height if height < smallest_height else smallest_height
                )
                largest_width = width if width > largest_width else largest_width
                largest_height = height if height > largest_height else largest_height
                widths.append(width)
                heights.append(height)
                total_imgs += 1
                img_area = width * height
                areas.append(img_area)
                smallest_img_area = (
                    img_area if img_area < smallest_img_area else smallest_img_area
                )
                largest_img_area = (
                    img_area if img_area > largest_img_area else largest_img_area
                )
                widths_std_dev = np.std(widths)
                heights_std_dev = np.std(heights)
                area_std_dev = np.std(areas)

    avg_width = sum(widths) / total_imgs
    avg_height = sum(heights) / total_imgs
    print("INFO:\n")
    print(f"total images: {total_imgs}")
    print(f"smallest width: {smallest_width}")
    print(f"smallest height: {smallest_height}")
    print(f"largest width: {largest_width}")
    print(f"largest height: {largest_height}")
    print(f"smallest image area: {smallest_img_area}")
    print(f"largest image area: {largest_img_area}")
    print(f"widths standard deviation: {widths_std_dev}")
    print(f"heights standard deviation: {heights_std_dev}")
    print(f"areas standard deviation: {area_std_dev}")


if __name__ == "__main__":
    analyze(*sys.argv[1:])
