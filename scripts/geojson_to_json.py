
# Standard Library imports:
import argparse
from collections import Counter
import io
import json
import os

# 3rd Party imports:
from PIL import Image
import numpy as np
from tqdm import tqdm

# h4dlib imports:
import _import_helper  # pylint: disable=unused-import # noqa: F401
import h4dlib.data.wv_util as wv


def generate_images_and_annotations(class_ids, source_imgs_dir, dest_imgs_dir, img_tag, xview_json, catLookup):
	# coco_images = {
	# 	"license": 5,
	# 	"file_name": "COCO_train2014_000000057870.jpg",
	# 	"coco_url": "http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg",
	# 	"height": 480,
	# 	"width": 640,
	# 	"date_captured": "2013-11-14 16:28:13",
	# 	"flickr_url": "http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg",
	# 	"id": 57870
	# }

	# coco_annotations = {
	# 	"segmentation": [[312.29,562.89,402.25,511.49,400.96,425.38,398.39,372.69,388.11,332.85,318.71,325.14,295.58,305.86,269.88,314.86,258.31,337.99,217.19,321.29,182.49,343.13,141.37,348.27,132.37,358.55,159.36,377.83,116.95,421.53,167.07,499.92,232.61,560.32,300.72,571.89]],
	# 	"area": 54652.9556,
	# 	"iscrowd": 0,
	# 	"image_id": 480023,
	# 	"bbox": [116.95,305.86,285.3,266.03],
	# 	"category_id": 58,
	# 	"id": 86
	# 	}
	# all of these arrays are the same length and correspond to all the entries in the geojson
	# so each image file appears multiple times in all_imgs.
	print(F"Getting xview labels from {xview_json}")
	all_coords, all_imgs, all_classes = wv.get_labels(xview_json)

	# not all images in geojson are in directory...
	existing_imgs = set(os.listdir(source_imgs_dir))

	images = []
	annotations = []
	file_index = 0
	annotation_index = 0

	converted_images = set()
	drop_counter = Counter()

	print(F"Converting to coco format...")
	for image in tqdm(all_imgs):
		# perform appropriate checks from earlier
		if not image in existing_imgs:
			continue
	
		if image in converted_images:
			continue

		arr = wv.get_image(os.path.join(source_imgs_dir, image))
		coords = all_coords[all_imgs == image]
		classes = all_classes[all_imgs == image]
		width = len(arr[0])
		height = len(arr)
		if not os.path.exists(dest_imgs_dir):
			os.makedirs(dest_imgs_dir)	
		image_dict = {}
		image_dict['license'] = 1
		image_dict['file_name'] = 'COCO_' +  img_tag + '_' + str(file_index).zfill(12) + ".jpeg"
		image_dict['coco_url'] = ""
		image_dict['width'] = width
		image_dict['height'] = height
		image_dict['date_captured'] = "2018-02-22 00:00:00"
		image_dict['flickr_url'] = ""
		image_dict['id'] = file_index
		images.append(image_dict)
		new_image = convertToJpeg(arr)
		with open(os.path.join(dest_imgs_dir,'COCO_' +  img_tag + '_' + str(file_index).zfill(12) + ".jpeg"), 'wb') as image_file:
			image_file.write(new_image)

		for i in range(len(coords)):
			class_id = int(classes[i])
			drop_annotation = False

			if not class_id in class_ids:
				continue

			box = coords[i]
			x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
			assert(x_max >= x_min and y_max >= y_min)
			
			# This case is never hit, but leaving it here anyway:
			if x_min == y_min == x_max == y_max == 0:
				drop_annotation = True
			
			x,y = int(x_min), int(y_min)
			w,h = int(x_max - x_min), int(y_max - y_min)

			# if w < 20 or h < 20:
			# 	continue
			if x_min < 0 or x_min > width:
				drop_annotation = True
			if x_max < 0 or x_max > width:
				drop_annotation = True
			if y_min < 0 or y_min > height:
				drop_annotation = True
			if y_max < 0 or y_max > height:
				drop_annotation = True

			if drop_annotation:
    			# Clip any bbox coords that exceed image bounds:
				x_min_cl, x_max_cl = min(max(x_min, 0), width-1), min(max(x_max, 0), width-1)
				y_min_cl, y_max_cl = min(max(y_min, 0), width-1), min(max(y_max, 0), height-1)
				# Calc. original and clipped bbox areas:
				original_area = np.abs(x_max-x_min)*np.abs(y_max-y_min)
				clipped_area = np.abs(x_max_cl-x_min_cl)*np.abs(y_max_cl-y_min_cl)
				# If the truncated bbox area is less than 2/3 the full bbox area, drop it:
				if clipped_area/original_area < 0.666666:
					drop_counter[catLookup[int(classes[i])]] += 1
					continue

			annotation_dict = {}
			annotation_dict['bbox'] = [x, y, w, h]				
			annotation_dict['segmentation'] = [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]]
			annotation_dict['area'] = w*h
			annotation_dict['iscrowd'] = 0
			annotation_dict['image_id'] = file_index
			annotation_dict['category_id'] = class_id
			annotation_dict['id'] = str(annotation_index)
			annotations.append(annotation_dict)
			annotation_index += 1
		file_index += 1
		converted_images.add(image)

	print(F"Finished converting to COCO format. Total dropped annotations: {sum(drop_counter.values())}")
	print(F"Drop counts by category: {drop_counter}")
	return images, annotations

def generate_info():
	info_json = {
		"description": "XView Dataset",
		"url": "http://xviewdataset.org/",
		"version": "1.0",
		"year": 2018,
		"contributor": "Defense Innovation Unit Experimental (DIUx)",
		"date_created": "2018/02/22"
	}
	return info_json

def generate_licenses():
	licenses = []
	license = {
		"url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
		"id": 1,
		"name": "Attribution-NonCommercial-ShareAlike License"
	}
	licenses.append(license)
	return licenses

def generate_categories():

	class_dict = {
	"Fixed-wing Aircraft": "Fixed-wing Aircraft",
	"Small Aircraft": "Fixed-wing Aircraft",
	"Cargo Plane": "Fixed-wing Aircraft",
	"Helicopter": "None",
	"Passenger Vehicle": "Passenger Vehicle",
	"Small Car": "Passenger Vehicle",
	"Bus": "Passenger Vehicle",
	"Pickup Truck": "Truck",
	"Utility Truck": "Truck",
	"Truck": "Truck",
	"Cargo Truck": "Truck",
	"Truck w/Box": "Truck",
	"Truck Tractor": "Truck",
	"Trailer": "Truck",
	"Truck w/Flatbed": "Truck",
	"Truck w/Liquid": "Truck",
	"Crane Truck": "Engineering Vehicle",
	"Railway Vehicle": "Railway Vehicle",
	"Passenger Car": "Railway Vehicle",
	"Cargo Car": "Railway Vehicle",
	"Flat Car": "Railway Vehicle",
	"Tank car": "Railway Vehicle",
	"Locomotive": "Railway Vehicle",
	"Maritime Vessel": "Maritime Vessel",
	"Motorboat": "Maritime Vessel",
	"Sailboat": "Maritime Vessel",
	"Tugboat": "Maritime Vessel",
	"Barge": "Maritime Vessel",
	"Fishing Vessel": "Maritime Vessel",
	"Ferry": "Maritime Vessel",
	"Yacht": "Maritime Vessel",
	"Container Ship": "Maritime Vessel",
	"Oil Tanker": "Maritime Vessel",
	"Engineering Vehicle": "Engineering Vehicle",
	"Tower crane": "Engineering Vehicle",
	"Container Crane": "Engineering Vehicle",
	"Reach Stacker": "Engineering Vehicle",
	"Straddle Carrier": "Engineering Vehicle",
	"Mobile Crane": "Engineering Vehicle",
	"Dump Truck": "Engineering Vehicle",
	"Haul Truck": "Engineering Vehicle",
	"Scraper/Tractor": "Engineering Vehicle",
	"Front loader/Bulldozer": "Engineering Vehicle",
	"Excavator": "Engineering Vehicle",
	"Cement Mixer": "Engineering Vehicle",
	"Ground Grader": "Engineering Vehicle",
	"Hut/Tent": "Building",
	"Shed": "Building",
	"Building": "Building",
	"Aircraft Hangar": "Building",
	"Damaged Building": "Building",
	"Facility": "Building",
	"Construction Site": "None",
	"Vehicle Lot": "None",
	"Helipad": "None",
	"Storage Tank": "None",
	"Shipping container lot": "None",
	"Shipping Container": "None",
	"Pylon": "None",
	"Tower": "None",
	}

	categories = []
	class_ids = []
	class_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../xview_class_labels.txt")
	with open(class_labels_path) as class_labels_file:
		while True:
			line = class_labels_file.readline()
			if not line:
				break
			class_id, name = line.split(":")
			if not int(class_id) in class_ids:
				class_ids.append(int(class_id))
			name = name.split('\n')[0]
			class_entry = {}
			class_entry['supercategory'] = class_dict[name]
			class_entry['id'] = int(class_id)
			class_entry['name'] = name
			categories.append(class_entry)
	return categories, class_ids

def convertToJpeg(im):
    """
	(copied from tfr_util.py, so we don't have to import tensorflow)
    Converts an image array into an encoded JPEG string.

    Args:
        im: an image array

    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format='JPEG')
        return f.getvalue()

def main():
	proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	DATA_DIR = "datasets/Xview/vanilla/"
	XVIEW_JSON = os.path.join(proj_root, 'datasets/Xview/vanilla/xView_train.geojson')
	IMG_TAG = '2014'
	SOURCE_IMGS_DIR = os.path.join(DATA_DIR, F"train_images")
	DEST_IMGS_DIR = os.path.join(proj_root, F'datasets/Xview/coco_vanilla')
	DEST_JSON = os.path.join(proj_root, 'datasets/Xview/coco_vanilla', F'xview_coco_v0.json')
    
	if not os.path.exists(DEST_IMGS_DIR):
		os.makedirs(DEST_IMGS_DIR)

	with open(DEST_JSON, 'w') as coco_file:
		root_json = {}
		categories_json, class_ids = generate_categories()
		catLookup = {c["id"]:c["name"] for c in categories_json}
		root_json['categories'] = categories_json
		root_json['info'] = generate_info()
		root_json['licenses'] = generate_licenses()
		images, annotations = generate_images_and_annotations(
			class_ids, SOURCE_IMGS_DIR, DEST_IMGS_DIR, IMG_TAG, XVIEW_JSON, catLookup)
		root_json['images'] = images
		root_json['annotations'] = annotations
		coco_file.write(json.dumps(root_json))

if __name__ == "__main__":
	main()
